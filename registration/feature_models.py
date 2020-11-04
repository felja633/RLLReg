from registration.model_initializaton import *
from lib.tensorlist import TensorList, TensorListList

class BaseFeatureModel:
    def to(self, y):
        return
    def posteriors(self, y):
        return 1.0
    def maximize(self, a, y, den):
        return
    def detach(self):
        return

class MultinomialModel(BaseFeatureModel):
    def __init__(self, parameters, K, features, repeat_list):
        self.params = parameters
        self.repeat_list = repeat_list
        self.initialize(K, features)

    def initialize(self, K, features):
        init_method = self.params.get("init_method", "dirichlet")
        if init_method=="uniform":
            init_distr_list = []
            for i in range(len(features)):
                init_distr_list.append(
                    torch.ones(K, self.params.num_feature_clusters).float() / self.params.num_feature_clusters)
        elif init_method == "dirichlet":
            init_distr_list = TensorList()
            relative_distr = features.sum(dim=1).sum_list()
            relative_distr = relative_distr / relative_distr.sum()
            for r in relative_distr:
                dir = torch.distributions.dirichlet.Dirichlet(r)
                init_distr_list.append(dir.sample((K,)))
        else:
            init_distr_list = []
            for i in range(len(features)):
                init_distr_list.append(
                    torch.ones(K, self.params.num_feature_clusters).float() / self.params.num_feature_clusters)

        self.distr = TensorListList(init_distr_list, repeat=self.repeat_list)

    def to(self, dev):
        self.distr = self.distr.to(dev)

    def posteriors(self, y):
        p = y.permute(1,0) @ self.distr.permute(1,0)  # marginalization
        return p

    def maximize(self, ap, ow, y, den):
        tmp = y/den.permute(1,0)
        as_sum = (tmp @ (ow * ap)).sum_list()
        self.distr = self.distr * TensorListList(as_sum.permute(1,0), repeat=self.repeat_list)
        self.distr = self.distr / self.distr.sum(dim=1, keepdims=True)


class VonMisesModelList(BaseFeatureModel):
    def __init__(self, parameters, K, features, s, mu, repeat_list):
        self.params = parameters
        self.mu = self.initialize_mu(K, features)
        if len(mu) > 0:
            self.mu = TensorListList(mu, repeat=repeat_list)

        self.K = K
        self.repeat_list = repeat_list
        self.s2 = s*s
        self.local_posterior = 1

    def to(self, dev):
        self.mu = self.mu.to(dev)

    def initialize_mu(self, K, features):
        X = TensorList()
        for TV in features:
            Xi = np.random.randn(TV[0].shape[0], K).astype(np.float32)
            Xi = torch.from_numpy(Xi).to(TV[0].device)
            Xi = Xi / torch.norm(Xi, dim=0, keepdim=True)
            X.append(Xi.permute(1,0))

        return X

    def posteriors(self, y):
        log_p = y.permute(1,0) @ TensorListList(self.mu.permute(1,0), repeat=self.repeat_list)
        p = log_p/self.s2
        return p.exp()

    def maximize(self, a, y, den):
        self.mu = ((y @ a).sum_list()).permute(1, 0)
        self.mu = self.mu/self.mu.norm(dim=-1,keepdim=True)
        return

    def detach(self):
        self.mu = self.mu.detach()
