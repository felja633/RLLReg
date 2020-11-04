# resampling ##
import numpy as np

def random_sampling(pts, num_points_left):
    def rnd_samp(nump, s):

        arr = np.arange(0, nump)
        if nump < s:
            return arr

        arr = np.random.permutation(arr)
        return arr[0:s]

    if isinstance(pts, (list,)):
        sh = pts[0].shape
        arr = rnd_samp(sh[1], num_points_left)
        return [p[:, arr] for p in pts]
    else:
        sh = pts.shape
        arr = rnd_samp(sh[1], num_points_left)
        return pts[:, arr], arr


def random_indices(tot_num_points, rate=None, num_points=None, max_num_points=None):
    if not rate is None:
        if rate >0.0 and rate < 1.0:
            n_points = int(tot_num_points * rate)
        else:
            n_points = tot_num_points

    elif num_points is None:
        n_points = tot_num_points
    else:
        n_points = num_points

    if not max_num_points is None:
        if n_points > max_num_points:
            n_points = max_num_points

    arr = np.random.permutation(np.arange(tot_num_points))

    return arr[:n_points]