import torch.nn as nn
import time
from registration.observation_weights import empirical_estimate
import registration.feature_models as feature_models
from registration.model_initializaton import *
from lib.tensorlist import TensorList, TensorListList
from lib.visdom import Visdom


class PSREG_features_list(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.show_progress = True
        if self.params.debug:
            visdom_info = dict(port=8103)
            self.visdom = Visdom(debug=2, visdom_info=visdom_info)

    def __call__(self, point_clouds):
        return self.register_point_sets(point_clouds)

    def register_point_sets(self, x):
        Vs = x["coordinates_ds"]
        features = x["features"]
        features_w = x["att"]
        repeat_list = [len(V) for V in Vs]
        init_R, init_t = get_init_transformation_list(Vs, self.params.get("mean_init", True))
        TVs = init_R @ Vs + init_t

        X = TensorList()
        Q = TensorList()
        mu = TensorList()
        for TV, Fs in zip(TVs, features):
            if self.params.cluster_init == "box":
                Xi = get_randn_box_cluster_means_list(TV, self.params.K)
            else:
                Xi = get_randn_sphere_cluster_means_list(TV, self.params.K, self.params.get("cluster_mean_scale", 1.0))
            Q.append(get_scaled_cluster_precisions_list(TV, Xi, self.params.cluster_precision_scale))
            X.append(Xi.T)


        feature_distr = feature_models.MultinomialModel(self.params.feature_distr_parameters, self.params.K,
                                                        features, repeat_list=repeat_list)

        feature_distr.to(self.params.device)

        X = TensorListList(X, repeat=repeat_list)
        self.betas = get_default_beta(Q, self.params.gamma)


        Vs = Vs
        TVs = TVs

        # Compute the observation weights
        if self.params.use_dare_weighting:
            observation_weights = empirical_estimate(Vs, self.params.ow_args)
            ow_reg_factor = 8.0
            ow_mean = observation_weights.mean(dim=0, keepdim=True)
            for idx in range(len(observation_weights)):
                for idxx in range(len(observation_weights[idx])):
                    observation_weights[idx][idxx][observation_weights[idx][idxx] > ow_reg_factor * ow_mean[idx][idxx]] \
                        = ow_reg_factor * ow_mean[idx][idxx]

        else:
            observation_weights = 1.0

        ds = TVs.permute(1,0).sqe(X).permute(1,0)

        if self.params.debug:
            self.visdom.register(dict(pcds=Vs[0].cpu(), X=X[0][0].cpu(), c=None), 'point_clouds', 2, 'init')
            time.sleep(1)

        Rs = init_R.to(self.params.device)
        ts = init_t.to(self.params.device)

        self.betas = TensorListList(self.betas, repeat=repeat_list)
        QL = TensorListList(Q, repeat=repeat_list)
        Riter = TensorListList()
        titer = TensorListList()
        TVs_iter = TensorListList()
        for i in range(self.params.num_iters):
            Qt = QL.permute(1,0)

            ap =  (-0.5 * ds * QL).exp() * QL.pow(1.5)

            if i < 1000:
                pyz_feature = feature_distr.posteriors(features)
            else:
                pyz_feature = 1.0

            a = ap * pyz_feature

            ac_den = a.sum(dim=-1, keepdim=True) + self.betas
            a =  a / ac_den  # normalize row-wise
            a = a * observation_weights

            L =  a.sum(dim=-2, keepdim=True).permute(1,0)
            W = (Vs @ a) * QL

            b = L * Qt  # weights, b
            mW = W.sum(dim=-1, keepdim=True)
            mX = (b.permute(1,0) @ X).permute(1,0)
            z = L.permute(1,0) @ Qt
            P = (W @ X).permute(1,0) - mX @ mW.permute(1,0) / z

            # Compute R and t
            svd_list_list = P.cpu().svd()
            Rs = TensorListList()
            for svd_list in svd_list_list:
                Rs_list = TensorList()
                for svd in svd_list:
                    uu, vv = svd.U, svd.V
                    vvt = vv.permute(1,0)
                    detuvt = uu @ vvt
                    detuvt = detuvt.det()
                    S=torch.ones(1,3)
                    S[:,-1]=detuvt
                    Rs_list.append((uu * S)  @  vvt)

                Rs.append(Rs_list)

            Rs = Rs.to(self.params.device)
            Riter.append(Rs)
            ts = (mX - Rs @ mW) / z
            titer.append(ts)
            TVs = Rs @ Vs + ts


            TVs_iter.append(TVs.clone())
            if self.params.debug:
                self.visdom.register(dict(pcds=TVs[0].cpu(), X=X[0][0].cpu(), c=None), 'point_clouds', 2,
                                     'registration-iter')
                time.sleep(0.2)
            # Update X
            den = L.sum_list()

            if self.params.fix_cluster_pos_iter < i:
                X = (TVs @ a).permute(1,0)
                X = TensorListList(X.sum_list() / den, repeat_list)

            # Update Q
            ds = TVs.permute(1,0).sqe(X).permute(1,0)

            wn = (a * ds).sum(dim=-2, keepdim=True).sum_list()
            Q = (3 * den / (wn.permute(1,0) + 3 * den * self.params.epsilon)).permute(1,0)
            QL = TensorListList(Q, repeat=repeat_list)

            feature_distr.maximize(ap=ap, ow=observation_weights, y=features, den=ac_den)


        out = dict(Rs=Rs, ts=ts, X=X, Riter=Riter[:-1], titer=titer[:-1], Vs=TVs, Vs_iter=TVs_iter[:-1], ow=observation_weights)
        return out
