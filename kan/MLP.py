import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .LBFGS import LBFGS

seed = 0
torch.manual_seed(seed)

class MLP(nn.Module):
    
    def __init__(self, width, act='silu', save_act=True, seed=0, device='cpu'):
        super(MLP, self).__init__()
        
        torch.manual_seed(seed)
        
        linears = []
        self.width = width
        self.depth = depth = len(width) - 1
        for i in range(depth):
            linears.append(nn.Linear(width[i], width[i+1]))
        self.linears = nn.ModuleList(linears)
        
        #if activation == 'silu':
        self.act_fun = torch.nn.SiLU()
        self.save_act = save_act
        self.acts = None
        
        self.cache_data = None
        
        self.device = device
        self.to(device)
        
        
    def to(self, device):
        super(MLP, self).to(device)
        self.device = device
        
        return self
        
        
    def get_act(self, x=None):
        if isinstance(x, dict):
            x = x['train_input']
        if x == None:
            if self.cache_data != None:
                x = self.cache_data
            else:
                raise Exception("missing input data x")
        save_act = self.save_act
        self.save_act = True
        self.forward(x)
        self.save_act = save_act
        
    @property
    def w(self):
        return [self.linears[l].weight for l in range(self.depth)]
        
    def forward(self, x):
        
        # cache data
        self.cache_data = x
        
        self.acts = []
        self.acts_scale = []
        self.wa_forward = []
        self.a_forward = []
        
        for i in range(self.depth):
            
            if self.save_act:
                act = x.clone()
                act_scale = torch.std(x, dim=0)
                wa_forward = act_scale[None, :] * self.linears[i].weight
                self.acts.append(act)
                if i > 0:
                    self.acts_scale.append(act_scale)
                self.wa_forward.append(wa_forward)
            
            x = self.linears[i](x)
            if i < self.depth - 1:
                x = self.act_fun(x)
            else:
                if self.save_act:
                    act_scale = torch.std(x, dim=0)
                    self.acts_scale.append(act_scale)
                
        return x
    
    def attribute(self):
        if self.acts == None:
            self.get_act()

        node_scores = []
        edge_scores = []

        # back propagate from the last layer
        node_score = torch.ones(self.width[-1]).requires_grad_(True).to(self.device)
        node_scores.append(node_score)

        for l in range(self.depth,0,-1):

            edge_score = torch.einsum('ij,i->ij', torch.abs(self.wa_forward[l-1]), node_score/(self.acts_scale[l-1]+1e-4))
            edge_scores.append(edge_score)

            # this might be improper for MLPs (although reasonable for KANs)
            node_score = torch.sum(edge_score, dim=0)/torch.sqrt(torch.tensor(self.width[l-1], device=self.device))
            #print(self.width[l])
            node_scores.append(node_score)

        self.node_scores = list(reversed(node_scores))
        self.edge_scores = list(reversed(edge_scores))
        self.wa_backward = self.edge_scores
    
    def plot(self, beta=3, scale=1., metric='w'):
        # metric = 'w', 'act' or 'fa'
        
        if metric == 'fa':
            self.attribute()
        
        depth = self.depth
        y0 = 0.5
        fig, ax = plt.subplots(figsize=(3*scale,3*y0*depth*scale))
        shp = self.width
        
        min_spacing = 1/max(self.width)
        for j in range(len(shp)):
            N = shp[j]
            for i in range(N):
                plt.scatter(1 / (2 * N) + i / N, j * y0, s=min_spacing ** 2 * 5000 * scale ** 2, color='black')
                
        plt.ylim(-0.1*y0,y0*depth+0.1*y0)
        plt.xlim(-0.02,1.02)

        linears = self.linears
        
        for ii in range(len(linears)):
            linear = linears[ii]
            p = linear.weight
            p_shp = p.shape
            
            if metric == 'w':
                pass
            elif metric == 'act':
                p = self.wa_forward[ii]
            elif metric == 'fa':
                p = self.wa_backward[ii]
            else:
                raise Exception('metric = \'{}\' not recognized. Choices are \'w\', \'act\', \'fa\'.'.format(metric))
            for i in range(p_shp[0]):
                for j in range(p_shp[1]):
                    plt.plot([1/(2*p_shp[0])+i/p_shp[0], 1/(2*p_shp[1])+j/p_shp[1]], [y0*(ii+1),y0*ii], lw=0.5*scale, alpha=np.tanh(beta*np.abs(p[i,j].cpu().detach().numpy())), color="blue" if p[i,j]>0 else "red")
                    
        ax.axis('off')
        
    def reg(self, reg_metric, lamb_l1, lamb_entropy):
        
        if reg_metric == 'w':
            acts_scale = self.w
        if reg_metric == 'act':
            acts_scale = self.wa_forward
        if reg_metric == 'fa':
            acts_scale = self.wa_backward
        if reg_metric == 'a':
            acts_scale = self.acts_scale
        
        if len(acts_scale[0].shape) == 2:
            reg_ = 0.

            for i in range(len(acts_scale)):
                vec = acts_scale[i]
                vec = torch.abs(vec)

                l1 = torch.sum(vec)
                p_row = vec / (torch.sum(vec, dim=1, keepdim=True) + 1)
                p_col = vec / (torch.sum(vec, dim=0, keepdim=True) + 1)
                entropy_row = - torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1))
                entropy_col = - torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0))
                reg_ += lamb_l1 * l1 + lamb_entropy * (entropy_row + entropy_col)
                
        elif len(acts_scale[0].shape) == 1:
            
            reg_ = 0.

            for i in range(len(acts_scale)):
                vec = acts_scale[i]
                vec = torch.abs(vec)

                l1 = torch.sum(vec)
                p = vec / (torch.sum(vec) + 1)
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy

        return reg_
    
    def get_reg(self, reg_metric, lamb_l1, lamb_entropy):
        return self.reg(reg_metric, lamb_l1, lamb_entropy)
        
    def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2.,
        loss_fn=None, lr=1., batch=-1, metrics=None, in_vars=None, out_vars=None, beta=3,
        device='cpu', reg_metric='w', display_metrics=None):

        # ---- device/dtype & data on device once
        dev = getattr(self, "device", device)
        if isinstance(dev, str):
            dev = torch.device(dev)
        self.to(dev)
        dtype = next(self.parameters()).dtype

        train_X = dataset['train_input'].to(dev)
        train_y = dataset['train_label'].to(dev)
        test_X  = dataset['test_input'].to(dev)
        test_y  = dataset['test_label'].to(dev)

        # ---- guard: lamb only works when activations are saved
        if lamb > 0. and not self.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')
        old_save_act = self.save_act
        if lamb == 0.:
            self.save_act = False

        # ---- helpers
        def _safe_pred(x):
            # forward, but guarantee finite outputs
            y = self.forward(x)
            return torch.nan_to_num(y, nan=0.0, posinf=1e30, neginf=-1e30)

        def _safe_loss(pred, target, fn):
            l = fn(pred, target)
            # clamp absurd magnitudes; remove NaN/Inf
            l = torch.nan_to_num(l, nan=1e30, posinf=1e30, neginf=1e30)
            return l

        def _safe_reg():
            if not self.save_act or lamb == 0.:
                return torch.zeros((), device=dev, dtype=dtype)
            r = self.get_reg(reg_metric, lamb_l1, lamb_entropy)
            return torch.nan_to_num(r, nan=0.0, posinf=1e12, neginf=1e12)

        def _safe_sqrt(x):
            return torch.sqrt(torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=1e30, neginf=0.0), min=0.0))

        def _finite_or_raise(name, t):
            if not torch.isfinite(t).all():
                bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5].squeeze(-1).tolist()
                raise RuntimeError(f"{name} contains non-finite values at indices {bad}")

        if loss_fn is None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        # ---- optimizer
        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = torch.optim.LBFGS(
                self.parameters(), lr=lr, history_size=10,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        # ---- bookkeeping
        results = {'train_loss': [], 'test_loss': [], 'reg': []}
        if metrics is not None:
            for m in metrics: results[m.__name__] = []

        # ---- batch sizes
        Ntr = train_X.shape[0]; Nte = test_X.shape[0]
        if batch == -1 or batch > Ntr:
            batch_size = Ntr
            batch_size_test = Nte
        else:
            batch_size = batch
            batch_size_test = min(batch, Nte)  # never exceed test size

        # ---- closure (shared by LBFGS; reused by Adam branch)
        def make_batch_ids():
            tr_id = np.random.choice(Ntr, batch_size, replace=False)
            te_id = np.random.choice(Nte, batch_size_test, replace=False)
            return tr_id, te_id

        # NOTE: LBFGS will call closure multiple times; train_id must stay fixed within a step
        train_id, test_id = make_batch_ids()

        def closure():
            nonlocal train_id
            optimizer.zero_grad(set_to_none=True)
            pred = _safe_pred(train_X[train_id])
            tr_loss = _safe_loss(pred, train_y[train_id], loss_fn)
            reg_ = _safe_reg()
            obj = tr_loss + lamb * reg_
            # if still non-finite, bail with a clear error (prevents NaN step)
            _finite_or_raise("objective", obj)
            obj.backward()
            # tame post-prune spikes
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            return obj

        # ---- training loop
        pbar = tqdm(range(steps), desc='description', ncols=100)
        for it in pbar:
            if it == steps - 1 and old_save_act:
                self.save_act = True  # original behavior preserved

            # fresh batch ids each iteration (LBFGS needs a fixed one inside closure)
            train_id, test_id = make_batch_ids()

            if opt == "LBFGS":
                # run with current train_id captured by closure
                try:
                    optimizer.step(closure)
                except RuntimeError as e:
                    # give a friendlier hint if numerical failure
                    raise RuntimeError(f"LBFGS step failed (non-finite objective/grad). "
                                       f"Try lower lr after pruning, or check atoms.") from e
                # recompute losses for logging
                with torch.no_grad():
                    pred_tr = _safe_pred(train_X[train_id])
                    train_loss = _safe_loss(pred_tr, train_y[train_id], loss_fn_eval)
                    reg_ = _safe_reg()
            else:  # Adam
                pred_tr = _safe_pred(train_X[train_id])
                train_loss = _safe_loss(pred_tr, train_y[train_id], loss_fn)
                reg_ = _safe_reg()
                loss = train_loss + lamb * reg_
                _finite_or_raise("loss", loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

            # test loss (no grad)
            with torch.no_grad():
                pred_te = _safe_pred(test_X[test_id])
                test_loss = _safe_loss(pred_te, test_y[test_id], loss_fn_eval)

            # optional custom metrics
            if metrics is not None:
                for m in metrics:
                    results[m.__name__].append(m().item())

            # logging (safe sqrt of MSE-like losses)
            tr_rmse = _safe_sqrt(train_loss).detach().cpu().item()
            te_rmse = _safe_sqrt(test_loss).detach().cpu().item()
            reg_val = torch.nan_to_num(reg_, nan=0.0, posinf=1e12, neginf=1e12).detach().cpu().item()

            results['train_loss'].append(tr_rmse)
            results['test_loss'].append(te_rmse)
            results['reg'].append(reg_val)

            if it % log == 0:
                if display_metrics is None:
                    pbar.set_description(f"| train_loss: {tr_rmse:.2e} | test_loss: {te_rmse:.2e} | reg: {reg_val:.2e} | ")
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        if metric not in results:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)

        return results

    
    @property
    def connection_cost(self):

        with torch.no_grad():
            cc = 0.
            for linear in self.linears:
                t = torch.abs(linear.weight)
                def get_coordinate(n):
                    return torch.linspace(0,1,steps=n+1, device=self.device)[:n] + 1/(2*n)

                in_dim = t.shape[0]
                x_in = get_coordinate(in_dim)

                out_dim = t.shape[1]
                x_out = get_coordinate(out_dim)

                dist = torch.abs(x_in[:,None] - x_out[None,:])
                cc += torch.sum(dist * t)

        return cc
    
    def swap(self, l, i1, i2):

        def swap_row(data, i1, i2):
            data[i1], data[i2] = data[i2].clone(), data[i1].clone()

        def swap_col(data, i1, i2):
            data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()

        swap_row(self.linears[l-1].weight.data, i1, i2)
        swap_row(self.linears[l-1].bias.data, i1, i2)
        swap_col(self.linears[l].weight.data, i1, i2)
    
    def auto_swap_l(self, l):

        num = self.width[l]
        for i in range(num):
            ccs = []
            for j in range(num):
                self.swap(l,i,j)
                self.get_act()
                self.attribute()
                cc = self.connection_cost.detach().clone()
                ccs.append(cc)
                self.swap(l,i,j)
            j = torch.argmin(torch.tensor(ccs))
            self.swap(l,i,j)

    def auto_swap(self):
        depth = self.depth
        for l in range(1, depth):
            self.auto_swap_l(l)
            
    def tree(self, x=None, in_var=None, style='tree', sym_th=1e-3, sep_th=1e-1, skip_sep_test=False, verbose=False):
        if x == None:
            x = self.cache_data
        plot_tree(self, x, in_var=in_var, style=style, sym_th=sym_th, sep_th=sep_th, skip_sep_test=skip_sep_test, verbose=verbose)