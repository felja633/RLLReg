import torch
import torch.nn as nn
import lib.utils as utils
from lib.tensorlist import TensorListList

class RLLActor(nn.Module):
    def __init__(self, model, weight=1, dist_thresh=0.25, c=0.4, alpha=-2.0,
                 compute_loss_iter=0, vis=None, min_corresponence_rate=0.3,
                 eval_rot_err_thresh=0.15,eval_trans_err_thresh=0.3):
        super().__init__()
        self.model = model
        self.th = dist_thresh
        self.weight= weight
        self.c = torch.tensor(c, device=model.params.device)
        self.eval_rot_err_thresh = eval_rot_err_thresh
        self.eval_trans_err_thresh = eval_trans_err_thresh
        self.vis = vis
        self.iter_cnt = 0
        self.epoch = 0
        self.num_success_acc = 0
        self.compute_loss_iter = compute_loss_iter
        self.min_corresponence_rate = min_corresponence_rate
        self.alpha=torch.tensor(alpha, device=model.params.device)

    def train(self, mode=True):
        self.model.train(mode)

    def __call__(self, batch, epoch):
        if self.epoch != epoch:
            self.epoch = epoch
            self.iter_cnt = 0
            self.num_success_acc=0

        data, info = batch
        batch_size = len(data['coordinates'])
        device = self.model.params.device
        loss_iter = 0
        loss = 0
        final_loss = 0

        out_feat = self.model.extract_features(data)
        features = out_feat["features"]
        info["R_gt"] = info["R_gt"].to(device)
        info["t_gt"] = info["t_gt"].to(device)

        num_pairs = 0
        for fb in features:
            M = len(fb)
            for ind1 in range(M - 1):
                for ind2 in range(ind1 + 1, M):
                    num_pairs = num_pairs + 1

        # check valid pairs wrt number of correspondences
        # invalid pairs are ignored in the computation of the loss
        valid_pairs_LL = []
        if self.min_corresponence_rate > 0.0:
            correspondences = []
            for coord_ds, r, t in zip(out_feat["coordinates_ds"], info["R_gt"], info["t_gt"]):
                correspondences.append(utils.extract_correspondences_gpu(coord_ds, r, t))

            for fb, corr in zip(features, correspondences):
                db = corr["distances"].to(device)
                M = len(fb)
                cnt = 0
                valid_pairs_L = []
                for ind1 in range(M - 1):
                    for ind2 in range(ind1 + 1, M):
                        mask = db[cnt] < self.th
                        corrs_rate = mask.sum().float() / mask.shape[0]
                        valid_pairs_L.append(corrs_rate > self.min_corresponence_rate)
                        cnt = cnt + 1

                valid_pairs_LL.append(sum(valid_pairs_L))

        else:
            for fb in features:
                cnt=0
                M = len(fb)
                for ind1 in range(M - 1):
                    for ind2 in range(ind1 + 1, M):
                        cnt = cnt + 1

                valid_pairs_LL.append(cnt)

        Rgt, tgt = info["R_gt"], info["t_gt"]

        num_samples = sum(v > 0.0 for v in valid_pairs_LL)
        if self.min_corresponence_rate > 0.0:
            self.iter_cnt += num_samples.item()
        else:
            self.iter_cnt += num_samples

        # only compute loss is there is at least one valid pair in the batch
        if sum(valid_pairs_LL) == 0:
            return loss, self.num_success_acc / (self.iter_cnt)

        if sum(valid_pairs_LL) < num_pairs:
            out_feat_filt = dict()
            for k in out_feat.keys():
                out_feat_filt[k] = TensorListList(
                    [out_feat[k][i] for i in range(len(out_feat[k])) if valid_pairs_LL[i]])

            Rgt = TensorListList([Rgt[i] for i in range(batch_size) if valid_pairs_LL[i]])
            tgt = TensorListList([tgt[i] for i in range(batch_size) if valid_pairs_LL[i]])
            out_feat = out_feat_filt

        out_reg = self.model.register(out_feat)
        if not self.vis is None:
            self.vis(out_reg, info, data)

        Rs, ts = out_reg["Rs"], out_reg["ts"]
        Riter, titer = out_reg["Riter"], out_reg["titer"]
        Rgt = Rgt.to(Rs[0][0].device)
        tgt = tgt.to(ts[0][0].device)

        rot_errs = utils.L2sqrList(Rs.detach(), Rgt).sqrt()
        trans_errs = utils.L2sqrTransList(ts.detach(), tgt, Rs.detach(), Rgt).sqrt()

        # check number of successful registrations
        num_success = 0
        valid_list = []
        for rs_err,ts_err in zip(rot_errs, trans_errs):
            for rerr, terr in zip(rs_err,ts_err):
                val = (rerr<self.eval_rot_err_thresh)*(terr<self.eval_trans_err_thresh)
                valid_list.append(val.item())
                num_success = num_success + val.item()

        self.num_success_acc += num_success

        # compute registration error per iteration
        for Rit, tit, w in zip(Riter, titer, self.weight["Vs_iter"][self.compute_loss_iter:]):
            if w > 0:
                trans_errs2 = utils.GARPointList(tit, tgt, Rit, Rgt, V=out_feat["coordinates_ds"],
                                                   c=self.c, alpha=self.alpha)
                for terr in trans_errs2:
                    for t in terr:
                        loss_iter = loss_iter + w * t

        # compute final registration error
        if self.weight["Vs"] > 0.0:
            trans_errs2 = utils.GARPointList(ts, tgt, Rs, Rgt, V=out_feat["coordinates_ds"],
                                               c=self.c, alpha=self.alpha)

            for terr in trans_errs2:
                for t in terr:
                    final_loss = final_loss + self.weight["Vs"] * t

        print("num valid: ", num_success, "num_success_acc rate: ",
              self.num_success_acc / (self.iter_cnt))

        if sum(valid_pairs_LL):
            loss = loss + final_loss + loss_iter
            loss=loss/num_samples

        return loss, self.num_success_acc / (self.iter_cnt)
