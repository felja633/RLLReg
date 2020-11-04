import torch
from torch_geometric.nn import knn
from lib.tensorlist import TensorList, TensorListList


def empirical_estimate(points, num_neighbors):
    ps, batch = points.permute(1,0).cat_tensors()

    N = ps.shape[0]
    val = knn(ps.contiguous(), ps.contiguous(), batch_x=batch, batch_y=batch, k=num_neighbors)
    A = ps[val[1,:]].reshape(N, num_neighbors, 3)
    A = A - A.mean(dim=1, keepdim=True)
    Asqr = A.permute(0, 2, 1).bmm(A)
    sigma,_ = Asqr.cpu().symeig()
    w = (sigma[:,2] * sigma[:,1]).sqrt()
    val = val[1,:].reshape(N, num_neighbors)
    w,_ = torch.median(w[val].to(ps.device), dim=1, keepdim=True)

    weights = TensorListList()
    bi = 0
    for point_list in points:
        ww = TensorList()
        for p in point_list:
            ww.append(w[batch==bi])
            bi = bi + 1

        weights.append(ww)

    return weights

