import torch
import copy
from typing import List


@torch.no_grad()
def flatten_grad(grads):
    return torch.concat([g.view(-1) for g in grads])

class Sorter:
    def sort(self):
        raise NotImplementedError()


class GraB(Sorter):
    def __init__(self, n, d, device=None, dtype=torch.float32):
        self.n = n
        self.d = d
        self.avg_grad = torch.zeros(d, device=device, dtype=dtype)
        self.cur_sum = torch.zeros_like(self.avg_grad)
        self.next_epoch_avg_grad = torch.zeros_like(self.avg_grad)
        self.orders = torch.randperm(self.n, device=device, dtype=torch.int64)
        self.next_orders = torch.randperm(
            self.n, device=device, dtype=torch.int64)
        self.left_ptr = 0
        self.right_ptr = self.n - 1

    @torch.no_grad()
    def sort(self):
        self.avg_grad.copy_(self.next_epoch_avg_grad)
        self.next_epoch_avg_grad.zero_()
        self.cur_sum.zero_()
        self.left_ptr = 0
        self.right_ptr = self.n - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        return self.orders.clone()

    @torch.no_grad()
    def single_step(self, g, idx):
        self.next_epoch_avg_grad.add_(g / self.n)
        g -= self.avg_grad
        plus_res = self.cur_sum + g
        minus_res = self.cur_sum - g
        if torch.norm(plus_res, p=2) <= torch.norm(minus_res, p=2):
            self.next_orders[self.left_ptr] = self.orders[idx]
            self.left_ptr += 1
            self.cur_sum = plus_res
        else:
            self.next_orders[self.right_ptr] = self.orders[idx]
            self.right_ptr -= 1
            self.cur_sum = minus_res

    @torch.no_grad()
    def step(self, batch_grads, batch_idx):  # assume batch_grads: tuple of (B, d)
        for i, idx in enumerate(batch_idx):
            g = torch.cat([gi[i].reshape(-1) for gi in batch_grads])
            self.single_step(g, idx)

# class GraBOptimized(Sorter):
#     def __init__(self, n, params, device=None):
#         self.n = n
#         self.avg_grad = [torch.zeros_like(p, device=device) for p in params]
#         self.cur_sum = [torch.zeros_like(p, device=device) for p in params]
#         self.next_epoch_avg_grad = [torch.zeros_like(p, device=device) for p in params]
#         self.orders = torch.randperm(self.n, device=device, dtype=torch.int64)
#         self.next_orders = torch.randperm(self.n, device=device, dtype=torch.int64)
#         self.left_ptr = 0
#         self.right_ptr = self.n - 1

#     @torch.no_grad()
#     def sort(self):
#         for g in self.avg_grad:
#             g.zero_()
#         for g in self.next_epoch_avg_grad:
#             g.zero_()
#         for g in self.cur_sum:
#             g.zero_()
#         self.left_ptr = 0
#         self.right_ptr = self.n - 1
#         self.orders = self.next_orders
#         self.next_orders = torch.zeros_like(self.next_orders)
#         return self.orders.clone()

#     @torch.no_grad()
#     def single_step(self, g: List[torch.Tensor], idx):
#         inner_prod = 0
#         for i, gi in enumerate(g):
#             self.next_epoch_avg_grad[i].add_(gi / self.n)
#             gi.sub_(self.avg_grad[i])
#             inner_prod += torch.inner(self.cur_sum[i].view(-1), gi.view(-1))
#         if inner_prod <= 0:
#             for i, gi in enumerate(g):
#                 self.cur_sum[i].add_(gi)
#             self.next_orders[self.left_ptr] = self.orders[idx]
#             self.left_ptr += 1
#         else:
#             for i, gi in enumerate(g):
#                 self.cur_sum[i].sub_(gi)
#             self.next_orders[self.right_ptr] = self.orders[idx]
#             self.right_ptr -= 1

#     @torch.no_grad()
#     def step(self, batch_grads, batch_idx):  # assume batch_grads: tuple of (B, d)
#         for i, idx in enumerate(batch_idx):
#             g_list = [gi[i] for gi in batch_grads]
#             self.single_step(g_list, idx)


class GraBOptimized(Sorter):
    def __init__(self, n, d, device=None, dtype=torch.float32):
        self.n = n
        self.d = d
        self.avg_grad = torch.zeros(d, device=device, dtype=dtype)
        self.cur_sum = torch.zeros_like(self.avg_grad)
        self.next_epoch_avg_grad = torch.zeros_like(self.avg_grad)
        self.orders = torch.randperm(self.n, device=device, dtype=torch.int64)
        self.next_orders = torch.randperm(
            self.n, device=device, dtype=torch.int64)
        self.left_ptr = 0
        self.right_ptr = self.n - 1

    @torch.no_grad()
    def sort(self):
        self.avg_grad.copy_(self.next_epoch_avg_grad / self.n)
        self.next_epoch_avg_grad.zero_()
        self.cur_sum.zero_()
        self.left_ptr = 0
        self.right_ptr = self.n - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        return self.orders.clone()

    @torch.no_grad()
    def single_step(self, g, idx):
        self.next_epoch_avg_grad.add_(g)
        g.sub_(self.avg_grad)
        if torch.inner(self.cur_sum, g) <= 0:
            self.next_orders[self.left_ptr] = self.orders[idx]
            self.left_ptr += 1
            self.cur_sum.add_(g)
        else:
            self.next_orders[self.right_ptr] = self.orders[idx]
            self.right_ptr -= 1
            self.cur_sum.sub_(g)

    @torch.no_grad()
    def step(self, batch_grads, batch_idx):  # assume batch_grads: tuple of (B, d)
        for i, idx in enumerate(batch_idx):
            g = torch.cat([gi[i].data.view(-1) for gi in batch_grads])
            self.single_step(g, idx)
            del g


class RecursiveGraB(Sorter):
    next_epoch_avg_grad: torch.Tensor
    avg_grad: torch.Tensor
    n: int

    def __init__(self, d, device=None):
        self.left = None
        self.right = None
        self.d = d
        self.device = device
        self.lst = []
        self.run_sum = torch.zeros(d, device=device)

    @torch.no_grad()
    def single_step(self, g_error, idx):
        if self.left is None and self.right is None:
            self.lst.append(idx)
            return
        if torch.inner(self.run_sum, g_error) <= 0:
            self.run_sum.add_(g_error)
            self.left.single_step(g_error, idx)
        else:
            self.run_sum.sub_(g_error)
            self.right.single_step(g_error, idx)

    @torch.no_grad()
    def step(self, batch_grads, batch_idx):  # assume batch_grads: tuple of (B, d)
        for i, idx in enumerate(batch_idx):
            g = torch.cat([gi[i].data.view(-1) for gi in batch_grads])
            self.next_epoch_avg_grad.add_(g)
            g.sub_(self.avg_grad)
            self.single_step(g, idx)
            del g
        
    @torch.no_grad()
    def _get_recursive_orders(self):
        if self.left is None and self.right is None:
            ret = torch.tensor(self.lst, device=self.device, dtype=torch.int64)
            self.lst = []
            return ret
        else:
            left_lst = self.left._get_recursive_orders()
            right_lst = self.right._get_recursive_orders()
            self.run_sum.zero_()
            return torch.cat([left_lst, torch.flip(right_lst, dims=(0,))])

    @torch.no_grad()
    def sort(self):
        orders = self._get_recursive_orders()
        self.avg_grad.copy_(self.next_epoch_avg_grad / self.n)
        self.next_epoch_avg_grad.zero_()
        return orders

    @staticmethod
    @torch.no_grad()
    def _create_balance_trees_helper(recurse_times, d, device=None, tree=None):
        tree = RecursiveGraB(d, device=device) if tree is None else tree
        if recurse_times != 0:
            tree.left = RecursiveGraB(d, device=device)
            tree.right = RecursiveGraB(d, device=device)
            RecursiveGraB._create_balance_trees_helper(recurse_times - 1, d, device=device, tree=tree.left)
            RecursiveGraB._create_balance_trees_helper(recurse_times - 1, d, device=device, tree=tree.right)
        return tree

    @staticmethod
    @torch.no_grad()
    def create_balance_trees(recurse_times, n, d, device=None):
        tree = RecursiveGraB._create_balance_trees_helper(
            recurse_times, 
            d, device=device, tree=None
        )
        tree.next_epoch_avg_grad = torch.zeros(d, device=device)
        tree.avg_grad = torch.zeros(d, device=device)
        tree.n = n
        return tree


class RandomReshuffling(Sorter):
    def __init__(self, n, device) -> None:
        super().__init__()
        self.n = n
        self.device = device
    
    def step(self, *args, **kw):
        pass

    def sort(self, *args, **kw): 
        return torch.randperm(self.n, device=self.device, dtype=torch.int64)


class PairBalance_Sorter(Sorter):
    def __init__(self, n:int, d:int, device=None):
        assert n % 2 == 0, "pair balance only supports even number"
        self.pair_diff = torch.zeros(d, device=device)
        self.n = n 
        self.d = d
        self.run_pair_diff_sum = torch.zeros(d, device=device)
        self.pair_cache = torch.zeros(d, device=device)
        self.next_orders = torch.arange(n, dtype=torch.int64)
        self.orders = self.next_orders.clone()
        self.left_ptr, self.right_ptr = 0, self.n - 1

    def reorder_online(self, grad_vecs, i):
        # grad at even step subtract grad at odd step
        # equivalent to vecs[i] - vecs[i + 1] 
        self.pair_cache -= grad_vecs
        plus_res, minus_res = self.run_pair_diff_sum + self.pair_cache, self.run_pair_diff_sum - self.pair_cache
        if torch.norm(plus_res, p=2) <= torch.norm(minus_res, p=2):
            self.next_orders[self.left_ptr]  = self.orders[i - 1]
            self.next_orders[self.right_ptr] = self.orders[i]
            self.run_pair_diff_sum = plus_res
        else:
            self.next_orders[self.right_ptr] = self.orders[i - 1]
            self.next_orders[self.left_ptr]  = self.orders[i]        
            self.run_pair_diff_sum = minus_res
    
        self.left_ptr += 1
        self.right_ptr -= 1
        self.pair_cache.zero_()

    def store_grad(self, grad_vecs):
        self.pair_cache += grad_vecs

    def step(self, optimizer, i: int):
        # d, n
        grad_vecs = flatten_grad(optimizer)
        if i % 2 == 0:
            # store gradients to use in next step
            self.store_grad(grad_vecs)
        else:
            # perform pair balance reorder online
            self.reorder_online(grad_vecs, i)

    def sort(self):
        self.pair_diff = 0
        self.left_ptr  = 0
        self.right_ptr = self.n - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        return self.orders.clone()
