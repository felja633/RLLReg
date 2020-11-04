import functools
import torch


class TensorList(list):
    """Container mainly used for lists of torch tensors. Extends lists with pytorch functionality."""

    def __init__(self, list_of_tensors = None):
        if list_of_tensors is None:
            list_of_tensors = list()
        super(TensorList, self).__init__(list_of_tensors)

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(TensorList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return TensorList([super(TensorList, self).__getitem__(i) for i in item])
        else:
            return TensorList(super(TensorList, self).__getitem__(item))

    def __add__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 + e2 for e1, e2 in zip(self, other)])
        return TensorList([e + other for e in self])

    def __radd__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 + e1 for e1, e2 in zip(self, other)])
        return TensorList([other + e for e in self])

    def __iadd__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] += e2
        else:
            for i in range(len(self)):
                self[i] += other
        return self

    def __sub__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 - e2 for e1, e2 in zip(self, other)])
        return TensorList([e - other for e in self])

    def __rsub__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 - e1 for e1, e2 in zip(self, other)])
        return TensorList([other - e for e in self])

    def __isub__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] -= e2
        else:
            for i in range(len(self)):
                self[i] -= other
        return self

    def __mul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 * e2 for e1, e2 in zip(self, other)])
        return TensorList([e * other for e in self])

    def __rmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 * e1 for e1, e2 in zip(self, other)])
        return TensorList([other * e for e in self])

    def __imul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] *= e2
        else:
            for i in range(len(self)):
                self[i] *= other
        return self

    def __truediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 / e2 for e1, e2 in zip(self, other)])
        return TensorList([e / other for e in self])

    def __rtruediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 / e1 for e1, e2 in zip(self, other)])
        return TensorList([other / e for e in self])

    def __itruediv__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] /= e2
        else:
            for i in range(len(self)):
                self[i] /= other
        return self

    def __matmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 @ e2 for e1, e2 in zip(self, other)])
        return TensorList([e @ other for e in self])

    def __rmatmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 @ e1 for e1, e2 in zip(self, other)])
        return TensorList([other @ e for e in self])

    def __imatmul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] @= e2
        else:
            for i in range(len(self)):
                self[i] @= other
        return self

    def __mod__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 % e2 for e1, e2 in zip(self, other)])
        return TensorList([e % other for e in self])

    def __rmod__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 % e1 for e1, e2 in zip(self, other)])
        return TensorList([other % e for e in self])

    def __pos__(self):
        return TensorList([+e for e in self])

    def __neg__(self):
        return TensorList([-e for e in self])

    def __le__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 <= e2 for e1, e2 in zip(self, other)])
        return TensorList([e <= other for e in self])

    def __ge__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 >= e2 for e1, e2 in zip(self, other)])
        return TensorList([e >= other for e in self])

    #def T(self):
    #    return TensorList([e.T for e in self])

    def sqe(self, other):
        if TensorList._iterable(other):
            l = []
            for i, e2 in enumerate(other):
                d = self[i].unsqueeze(-1).permute(1, 2, 0) - e2.unsqueeze(-1).permute(1, 0, 2)
                l.append((d * d).sum(dim=0))

            return TensorList(l)
        else:
            l = []
            for i in range(len(self)):
                d = self[i].unsqueeze(-1).permute(1, 2, 0) - other.unsqueeze(-1).permute(1, 0, 2)
                l.append((d * d).sum(dim=0))

            return TensorList(l)

    def sqe2(self, other):
        l = []
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                d = (self[i]**2).sum(dim=1,keepdim=True).permute(1,0)+(e2**2).sum(dim=1, keepdim=True)-2*(e2 @ self[i].permute(1,0))
                l.append(d)
        else:
            othre_sqr = (other**2).sum(dim=1,keepdim=True)
            for i in range(len(self)):
                d = (self[i]**2).sum(dim=1,keepdim=True).permute(1,0)+othre_sqr-2*(other @ self[i].permute(1,0))
                l.append(d)


        return TensorList(l)

    def sumd(self, dim=-1, keepdim=False):
        return TensorList([e.sum(dim=dim, keepdim=keepdim) for e in self])

    def detach(self):
        return TensorList([e.detach() for e in self])

    def to(self, device):
        return TensorList([e.to(device) for e in self])

    def norm(self, dim, keepdim=False):
        return TensorList([e.norm(dim=dim, keepdim=keepdim) for e in self])

    def concat(self, other):
        return TensorList(super(TensorList, self).__add__(other))

    def copy(self):
        return TensorList(super(TensorList, self).copy())

    def unroll(self):
        if not any(isinstance(t, TensorList) for t in self):
            return self

        new_list = TensorList()
        for t in self:
            if isinstance(t, TensorList):
                new_list.extend(t.unroll())
            else:
                new_list.append(t)
        return new_list

    def cat_tensors(self):
        t = torch.cat(self)
        b = torch.cat([i*torch.ones((e.shape[0],), dtype=torch.long) for i, e in enumerate(self)]).to(t[0].device)
        return t,b

    def any_nan(self):
        return sum([torch.isnan(e).any() for e in self])>0

    def list(self):
        return list(self)

    def attribute(self, attr: str, *args):
        return TensorList([getattr(e, attr, *args) for e in self])

    def apply(self, fn):
        return TensorList([fn(e) for e in self])

    def __getattr__(self, name):
        if not hasattr(torch.Tensor, name):
            raise AttributeError('\'TensorList\' object has no attribute \'{}\''.format(name))

        def apply_attr(*args, **kwargs):
            return TensorList([getattr(e, name)(*args, **kwargs) for e in self])

        return apply_attr

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorList, list))



class TensorListList(list):
    def __init__(self, list_of_tensors = None, repeat=None):

        if list_of_tensors is None:
            list_of_tensors = list()
        elif not repeat is None:
            list_of_list_tensors2 = list()
            for i, e in enumerate(list_of_tensors):
                list_of_tensors2 = TensorList()
                if isinstance(repeat, list):
                    rep = repeat[i]
                else:
                    rep = repeat

                for i in range(rep):
                    list_of_tensors2.append(e)

                list_of_list_tensors2.append(list_of_tensors2)

            list_of_tensors = list_of_list_tensors2

        super(TensorListList, self).__init__(list_of_tensors)

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(TensorListList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return TensorList([super(TensorListList, self).__getitem__(i) for i in item])
        else:
            return TensorList(super(TensorListList, self).__getitem__(item))

    def __add__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e1 + e2 for e1, e2 in zip(self, other)])
        return TensorList([e + other for e in self])

    def __radd__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e2 + e1 for e1, e2 in zip(self, other)])
        return TensorListList([other + e for e in self])

    def __iadd__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            for i, e2 in enumerate(other):
                self[i] += e2
        else:
            for i in range(len(self)):
                self[i] += other
        return self

    def __sub__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e1 - e2 for e1, e2 in zip(self, other)])
        return TensorListList([e - other for e in self])

    def __rsub__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e2 - e1 for e1, e2 in zip(self, other)])
        return TensorListList([other - e for e in self])

    def __isub__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            for i, e2 in enumerate(other):
                self[i] -= e2
        else:
            for i in range(len(self)):
                self[i] -= other
        return self

    def __mul__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e1 * e2 for e1, e2 in zip(self, other)])
        return TensorListList([e * other for e in self])

    def __rmul__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e2 * e1 for e1, e2 in zip(self, other)])
        return TensorListList([other * e for e in self])

    def __imul__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            for i, e2 in enumerate(other):
                self[i] *= e2
        else:
            for i in range(len(self)):
                self[i] *= other
        return self

    def __truediv__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e1 / e2 for e1, e2 in zip(self, other)])
        return TensorListList([e / other for e in self])

    def __rtruediv__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e2 / e1 for e1, e2 in zip(self, other)])
        return TensorListList([other / e for e in self])

    def __itruediv__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            for i, e2 in enumerate(other):
                self[i] /= e2
        else:
            for i in range(len(self)):
                self[i] /= other
        return self

    def __matmul__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e1 @ e2 for e1, e2 in zip(self, other)])
        return TensorListList([e @ other for e in self])

    def __rmatmul__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e2 @ e1 for e1, e2 in zip(self, other)])
        return TensorListList([other @ e for e in self])

    def __imatmul__(self, other):
        if TensorListList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] @= e2
        else:
            for i in range(len(self)):
                self[i] @= other
        return self

    def __mod__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e1 % e2 for e1, e2 in zip(self, other)])
        return TensorListList([e % other for e in self])

    def __rmod__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e2 % e1 for e1, e2 in zip(self, other)])
        return TensorListList([other % e for e in self])

    def __pos__(self):
        return TensorListList([+e for e in self])

    def __neg__(self):
        return TensorListList([-e for e in self])

    def __le__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e1 <= e2 for e1, e2 in zip(self, other)])
        return TensorListList([e <= other for e in self])

    def __ge__(self, other):
        if TensorListList._iterable(other):
            assert len(self) == len(other)
            return TensorListList([e1 >= e2 for e1, e2 in zip(self, other)])
        return TensorListList([e >= other for e in self])

    def sqe(self, other):
        return TensorListList([e1.sqe(e2) for e1, e2 in zip(self, other)])

    def sqe2(self, other):
        return TensorListList([e1.sqe2(e2) for e1, e2 in zip(self, other)])

    def detach(self):
        return TensorListList([e.detach() for e in self])

    def to(self, device):
        return TensorListList([e.to(device) for e in self])

    def sum_list(self):
        out = TensorList()
        for e in self:
            sum = 0
            for ee in e:
                sum = sum + ee
            out.append(sum)
        return out

    def sumd(self, dim=-1, keepdim=False):
        return TensorListList([e.sum(dim=dim, keepdim=keepdim) for e in self])

    def concat(self, other):
        return TensorListList(super(TensorListList, self).__add__(other))

    def copy(self):
        return TensorListList(super(TensorListList, self).copy())

    def unroll(self):
        if not any(isinstance(t, TensorListList) for t in self):
            return self

        new_list = TensorListList()
        for t in self:
            if isinstance(t, TensorListList):
                new_list.extend(t.unroll())
            else:
                new_list.append(t)
        return new_list

    def any_nan(self):
        return sum([e.any_nan() for e in self])>0

    def list(self):
        return list(self)

    def attribute(self, attr: str, *args):
        return TensorListList([e.attribute(attr, *args) for e in self])

    def apply(self, fn):
        return TensorListList([e.apply(fn) for e in self])

    def tot_num_tensors(self):
        return sum([len(e) for e in self])

    def flatten_out(self):
        return TensorList([item for sublist in self for item in sublist])

    def cat_tensors(self):
        cnt=0
        l = []
        b = []
        for e in self:
            ee, bb = e.cat_tensors()
            l.append(ee)
            b.append(bb+cnt)
            cnt+=len(e)

        return torch.cat(l), torch.cat(b)

    def __getattr__(self, name):
        if not hasattr(torch.Tensor, name):
            raise AttributeError('\'TensorList\' object has no attribute \'{}\''.format(name))

        def apply_attr(*args, **kwargs):
            return TensorListList([getattr(e, name)(*args, **kwargs) for e in self])

        return apply_attr

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorListList, list))


def tensor_operation(op):
    def islist(a):
        return isinstance(a, TensorList)

    @functools.wraps(op)
    def oplist(*args, **kwargs):
        if len(args) == 0:
            raise ValueError('Must be at least one argument without keyword (i.e. operand).')

        if len(args) == 1:
            if islist(args[0]):
                return TensorList([op(a, **kwargs) for a in args[0]])
        else:
            # Multiple operands, assume max two
            if islist(args[0]) and islist(args[1]):
                return TensorList([op(a, b, *args[2:], **kwargs) for a, b in zip(*args[:2])])
            if islist(args[0]):
                return TensorList([op(a, *args[1:], **kwargs) for a in args[0]])
            if islist(args[1]):
                return TensorList([op(args[0], b, *args[2:], **kwargs) for b in args[1]])

        # None of the operands are lists
        return op(*args, **kwargs)

    return oplist
