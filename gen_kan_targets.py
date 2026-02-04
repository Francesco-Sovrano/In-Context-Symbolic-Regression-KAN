#!/usr/bin/env python3
"""
Generate random target functions f(x) similar to the example, but restricted to:

- Atoms/functions ONLY from `lib` (plus numeric constants if you allow them).
- Binary operators ONLY: + and -
- Function nesting depth <= 2 (i.e., at most ONE level of nesting), like:
    exp( sin(x0) + (x1)^2 )
  but NOT:
    exp(log(abs(x0)))   # depth 3 along a path

It prints:
  1) A human-readable symbolic string
  2) A Torch implementation: `def f(x): return ...`

Notes on "x^2", "1/x", "1/x^2":
- Treated as unary atoms that can wrap an expression.
- For numerical safety, log/sqrt/recip use patterns like log(abs(z)+eps), 1/(abs(z)+eps), etc.
  These still only use atoms in lib (+/- and constants).

Example:
  python3 gen_kan_targets.py --n_var 5 --num_funcs 100 --seed 0

CSV example:
  python3 gen_kan_targets.py --n_var 5 --num_funcs 100 --seed 0 --emit_csv data/targets.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import torch


# ---------- Library (as in your example) ----------
LIB = [
	"0", "1", 
	"x",
	"x^2", "x^3", "x^4", "x^5",
	"1/x", "1/x^2", "1/x^3",
	"sqrt", "1/sqrt(x)",
	"log", "exp",
	"sin", "cos", "tan", "tanh",
	"abs", "sgn",
	"arctan", "arcsin", "arccos", 
	"arctanh",
	"gaussian",
]

# Unary ops we can emit (must correspond to items in LIB)
UNARY_OPS = [
	"x^2", "x^3", "x^4", "x^5",
	"1/x", "1/x^2", "1/x^3",
	"sqrt", "1/sqrt(x)",
	"log", "exp",
	"sin", "cos", "tan", "tanh",
	"abs", "sgn",
	"arctan", "arcsin", "arccos", 
	"arctanh",
	"gaussian",
]

# "Always safe" (no domain restriction on R)
SAFE_UNARY = [
	"x^2", "x^3", "x^4", "x^5",
	"exp",
	"sin", "cos", "tan", "tanh",
	"abs", "sgn",
]


# ---------- AST ----------
@dataclass(frozen=True)
class Var:
    i: int


@dataclass(frozen=True)
class Const:
    v: float


@dataclass(frozen=True)
class AddSub:
    # terms with signs: (+1 or -1, expr)
    terms: Tuple[Tuple[int, "Expr"], ...]


@dataclass(frozen=True)
class Unary:
    op: str
    arg: "Expr"


Expr = Union[Var, Const, AddSub, Unary]


def max_unary_depth(e: Expr) -> int:
    """Maximum number of Unary nodes along any root-to-leaf path."""
    if isinstance(e, (Var, Const)):
        return 0
    if isinstance(e, AddSub):
        return max((max_unary_depth(t) for _, t in e.terms), default=0)
    if isinstance(e, Unary):
        return 1 + max_unary_depth(e.arg)
    raise TypeError(e)


def used_ops(e: Expr) -> set:
    """Return set of unary op strings used."""
    if isinstance(e, (Var, Const)):
        return set()
    if isinstance(e, AddSub):
        s = set()
        for _, t in e.terms:
            s |= used_ops(t)
        return s
    if isinstance(e, Unary):
        return {e.op} | used_ops(e.arg)
    raise TypeError(e)


def expr_to_symbolic(e: Expr, var_names: Sequence[str]) -> str:
    if isinstance(e, Var):
        return var_names[e.i]
    if isinstance(e, Const):
        if e.v == int(e.v):
            return str(int(e.v))
        return repr(float(e.v))
    if isinstance(e, AddSub):
        parts = []
        for j, (sgn, t) in enumerate(e.terms):
            s = expr_to_symbolic(t, var_names)
            if j == 0:
                parts.append(s if sgn > 0 else f"-({s})")
            else:
                parts.append((" + " if sgn > 0 else " - ") + f"({s})")
        return "".join(parts) if parts else "0"
    if isinstance(e, Unary):
        a = expr_to_symbolic(e.arg, var_names)
        op = e.op
        if op == "x^2":
            return f"({a})^2"
        if op == "1/x":
            return f"1/({a})"
        if op == "1/x^2":
            return f"1/(({a})^2)"
        return f"{op}({a})"
    raise TypeError(e)


def expr_to_torch(e: Expr, n_var: int) -> str:
    """
    Emit Torch code as a Python expression string.
    Assumes input `x` has shape [N, n_var] and we always keep column vectors x[:, [i]].
    """
    if isinstance(e, Var):
        if not (0 <= e.i < n_var):
            raise ValueError("Var index out of range")
        return f"x[:, [{e.i}]]"
    if isinstance(e, Const):
        return repr(float(e.v))
    if isinstance(e, AddSub):
        if not e.terms:
            return "0.0"
        out = []
        for j, (sgn, t) in enumerate(e.terms):
            ts = expr_to_torch(t, n_var)
            if j == 0:
                out.append(ts if sgn > 0 else f"-({ts})")
            else:
                out.append((" + " if sgn > 0 else " - ") + f"({ts})")
        return "".join(out)
    if isinstance(e, Unary):
        a = expr_to_torch(e.arg, n_var)
        op = e.op
        if op == "x^2":
            return f"(({a}) ** 2)"
        if op == "exp":
            return f"torch.exp({a})"
        if op == "log":
            return f"torch.log({a})"
        if op == "sqrt":
            return f"torch.sqrt({a})"
        if op == "tanh":
            return f"torch.tanh({a})"
        if op == "sin":
            return f"torch.sin({a})"
        if op == "abs":
            return f"torch.abs({a})"
        if op == "1/x":
            return f"(1.0 / ({a}))"
        if op == "1/x^2":
            return f"(1.0 / (({a}) ** 2))"
        raise ValueError(f"Unknown op: {op}")
    raise TypeError(e)


# ---------- Generation ----------
def default_var_names(n: int) -> List[str]:
    base = ["x", "y", "z", "u", "v", "w", "a", "b", "c", "d"]
    if n <= len(base):
        return base[:n]
    return base + [f"x{i}" for i in range(len(base), n)]


def rand_const(rng: random.Random, const_pool: Sequence[float]) -> Const:
    return Const(rng.choice(const_pool))


def rand_var(rng: random.Random, n_var: int) -> Var:
    return Var(rng.randrange(n_var))


def make_sum(rng: random.Random, terms: List[Expr], allow_zero_sum: bool = False) -> Expr:
    """Create an additive expression from given terms with random +/- signs."""
    signed: List[Tuple[int, Expr]] = []
    for t in terms:
        sgn = 1 if rng.random() < 0.5 else -1
        signed.append((sgn, t))
    if not signed:
        return Const(0.0)
    e = AddSub(tuple(signed))
    if not allow_zero_sum and all(isinstance(t, Const) for _, t in e.terms):
        return e
    return e


def safeify_for_domain(rng: random.Random, op: str, arg: Expr, eps: float) -> Expr:
    """
    For domain-limited ops (log/sqrt/recip), turn `arg` into something safe
    using only allowed ops (+/-) and allowed unary atoms.

    Important: this adds at most ONE inner unary (abs) plus + eps constant,
    so it respects the nesting constraint if you budget for it.
    """
    if op in ("log", "sqrt"):
        return AddSub(((+1, Unary("abs", arg)), (+1, Const(eps))),)
    if op in ("1/x", "1/x^2"):
        return AddSub(((+1, Unary("abs", arg)), (+1, Const(eps))),)
    return arg


def random_term(
    rng: random.Random,
    n_var: int,
    max_term_depth: int,
    allow_constants: bool,
    const_pool: Sequence[float],
    eps: float,
) -> Expr:
    """
    Build a single term with unary nesting depth <= max_term_depth.
    Depth counts Unary nodes only (AddSub doesn't increase depth).
    """
    if allow_constants and rng.random() < 0.2:
        leaf: Expr = rand_const(rng, const_pool)
    else:
        leaf = rand_var(rng, n_var)

    if max_term_depth <= 0:
        return leaf

    depth = rng.randrange(max_term_depth + 1)
    if depth == 0:
        return leaf

    if depth == 1:
        op = rng.choice(SAFE_UNARY)
        return Unary(op, leaf)

    op_outer = rng.choice(UNARY_OPS)

    if rng.random() < 0.6:
        inner = leaf
    else:
        op_inner = rng.choice(SAFE_UNARY)
        inner = Unary(op_inner, leaf)

    if op_outer in ("log", "sqrt", "1/x", "1/x^2"):
        inner = safeify_for_domain(rng, op_outer, leaf, eps=eps)

    return Unary(op_outer, inner)


def random_expr(
    rng: random.Random,
    n_var: int,
    max_terms: int,
    outer_wrapper_prob: float,
    allow_constants: bool,
    const_pool: Sequence[float],
    eps: float,
) -> Expr:
    """
    Build an expression:
      - sum of K terms, K in [1, max_terms]
      - optionally wrapped in ONE outer safe unary (exp/sin/tanh/abs/x^2)
    If wrapped, each term is limited to depth<=1 so total depth<=2.
    If not wrapped, terms can reach depth<=2.
    """
    use_wrapper = (rng.random() < outer_wrapper_prob)

    k = rng.randint(1, max_terms)
    term_depth = 1 if use_wrapper else 2
    terms = [
        random_term(
            rng,
            n_var=n_var,
            max_term_depth=term_depth,
            allow_constants=allow_constants,
            const_pool=const_pool,
            eps=eps,
        )
        for _ in range(k)
    ]
    s: Expr = make_sum(rng, terms)

    if use_wrapper:
        wrapper = rng.choice(SAFE_UNARY)
        s = Unary(wrapper, s)

    if max_unary_depth(s) > 2:
        raise RuntimeError("Bug: produced depth > 2")

    return s


def is_valid_over_range(expr: Expr, n_var: int, lo: float, hi: float, samples: int) -> bool:
    """Quick finite-check on random samples in [lo, hi]."""
    x = (hi - lo) * torch.rand(samples, n_var) + lo
    try:
        code = expr_to_torch(expr, n_var)
        y = eval(code, {"torch": torch, "x": x, "math": math})
        return torch.isfinite(y).all().item() and y.shape == (samples, 1)
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_var", type=int, default=2)
    p.add_argument("--num_funcs", type=int, default=5)
    p.add_argument("--max_terms", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--outer_wrapper_prob", type=float, default=0.6)
    p.add_argument("--allow_constants", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--const_pool",
        type=float,
        nargs="*",
        default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        help="Constants allowed in expressions (if --allow-constants).",
    )
    p.add_argument("--eps", type=float, default=1e-3, help="Small epsilon used in safe log/sqrt/recip patterns.")

    p.add_argument("--range_min", type=float, default=-1.0)
    p.add_argument("--range_max", type=float, default=1.0)
    p.add_argument("--check_samples", type=int, default=1024)
    p.add_argument("--max_tries_per_func", type=int, default=500)

    p.add_argument(
        "--emit_python_module",
        type=str,
        default="",
        help="If set, write a .py file containing generated functions + a list `GENERATED`.",
    )

    # NEW: CSV output
    p.add_argument(
        "--emit_csv",
        type=str,
        default="",
        help="If set, write a CSV with one row per generated function.",
    )
    return p.parse_args()


def write_csv(path: str, generated: List[Expr], var_names: Sequence[str], args: argparse.Namespace) -> None:
    rows = []
    for idx, e in enumerate(generated):
        sym = expr_to_symbolic(e, var_names)
        code = expr_to_torch(e, args.n_var)
        ops = sorted(list(used_ops(e)))
        rows.append(
            {
                "idx": idx,
                "symbolic": sym,
                "torch": code,
                "ops": " ".join(ops),
                "max_unary_depth": max_unary_depth(e),
                "n_var": args.n_var,
                "max_terms": args.max_terms,
                "seed": args.seed,
                "range_min": args.range_min,
                "range_max": args.range_max,
                "check_samples": args.check_samples,
                "outer_wrapper_prob": args.outer_wrapper_prob,
                "allow_constants": args.allow_constants,
                "const_pool": " ".join([str(x) for x in args.const_pool]),
                "eps": args.eps,
            }
        )

    fieldnames = list(rows[0].keys()) if rows else [
        "idx","symbolic","torch","ops","max_unary_depth",
        "n_var","max_terms","seed","range_min","range_max",
        "check_samples","outer_wrapper_prob","allow_constants","const_pool","eps"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    var_names = default_var_names(args.n_var)

    generated: List[Expr] = []
    seen_sym = set()

    for _ in range(args.num_funcs):
        ok = False
        for _try in range(args.max_tries_per_func):
            e = random_expr(
                rng,
                n_var=args.n_var,
                max_terms=args.max_terms,
                outer_wrapper_prob=args.outer_wrapper_prob,
                allow_constants=args.allow_constants,
                const_pool=args.const_pool,
                eps=args.eps,
            )

            ops = used_ops(e)
            if not ops.issubset(set(UNARY_OPS)):
                continue

            sym = expr_to_symbolic(e, var_names)
            if sym in seen_sym:
                continue

            if not is_valid_over_range(e, args.n_var, args.range_min, args.range_max, args.check_samples):
                continue

            generated.append(e)
            seen_sym.add(sym)
            ok = True
            break

        if not ok:
            raise RuntimeError("Failed to generate a valid function within max tries. Try relaxing constraints.")

    # If CSV requested: write it and exit without printing blocks
    if args.emit_csv:
        write_csv(args.emit_csv, generated, var_names, args)
        print(f"Wrote CSV: {args.emit_csv}")
        return

    # Original stdout printing
    print("lib =", LIB)
    print()
    for idx, e in enumerate(generated):
        sym = expr_to_symbolic(e, var_names)
        code = expr_to_torch(e, args.n_var)
        print(f"=== f{idx} ===")
        print("symbolic:", sym)
        print("torch:   ", code)
        print()

    # Optionally emit a python module
    if args.emit_python_module:
        lines = []
        lines.append("# Auto-generated by gen_kan_targets.py\n")
        lines.append("import torch\n\n")
        lines.append(f"LIB = {LIB!r}\n\n")
        lines.append(f"N_VAR = {args.n_var}\n\n")
        for idx, e in enumerate(generated):
            code = expr_to_torch(e, args.n_var)
            sym = expr_to_symbolic(e, var_names)
            lines.append(f"def f{idx}(x):\n")
            lines.append(f"    # {sym}\n")
            lines.append(f"    return {code}\n\n")
        fnames = ", ".join([f"f{i}" for i in range(len(generated))])
        lines.append(f"GENERATED = [{fnames}]\n")
        with open(args.emit_python_module, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        print(f"Wrote: {args.emit_python_module}")


if __name__ == "__main__":
    main()
