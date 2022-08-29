import math
import os
import time

import numpy as np
import torch
from torchvision.utils import save_image



class MFA(torch.nn.Module):
    """
    A class representing a Mixture of Factor Analyzers [1] / Mixture of Probabilistic PCA [2] in pytorch.
    MFA/MPPCA are Gaussian Mixture Models with low-rank-plus-diagonal covariance, enabling efficient modeling
    of high-dimensional domains in which the data resoides near lower-dimensional subspaces.
    The implementation is based on [3] (please quote this if you are using this package in your research).

    Attributes (model parameters):
    ------------------------------
    MU: Tensor shaped [n_components, n_features]
        The component means.
    A: Tensor shaped [n_components, n_features, n_factors]
        The component subspace directions / factor loadings. These should be orthogonal (but not orthonormal)
    lod_D: Tensor shaped [n_components, n_features]
        Log of the component diagonal variance values. Note that in MPPCA, all values along the diagonal are the same.
    PI_logits: Tensor shaped [n_components]
        Log of the component mixing-coefficients (probabilities). Apply softmax to get the actual PI values.

    Main Methods:
    -------------
    fit:
        Fit the MPPCA model parameters to pre-loaded training data using EM

    batch_fit:
        Fit the MPPCA model parameters to a (possibly large) pytorch dataset using EM in batches

    sample:
        Generate new samples from the trained model

    per_component_log_likelihood, log_prob, log_likelihood:
        Probability query methods

    responsibilities, log_responsibilities, map_component:
        Responsibility (which component the sample comes from) query methods

    reconstruct, conditional_reconstruct:
        Reconstruction and in-painting

    [1] Tipping, Michael E., and Christopher M. Bishop. "Mixtures of probabilistic principal component analyzers."
        Neural computation 11.2 (1999): 443-482.
    [2] Ghahramani, Zoubin, and Geoffrey E. Hinton. "The EM algorithm for mixtures of factor analyzers."
        Vol. 60. Technical Report CRG-TR-96-1, University of Toronto, 1996.
    [3] Richardson, Eitan, and Yair Weiss. "On gans and gmms."
        Advances in Neural Information Processing Systems. 2018.

    """
    def __init__(self, n_components, n_features, n_factors):
        super(MFA, self).__init__()
        self.n_components = n_components
        self.n_features = n_features

        self.MU = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False)
        self.A = torch.nn.Parameter(torch.zeros(n_components, n_features, n_factors), requires_grad=False)
        self.log_D = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False)
        self.PI_logits = torch.nn.Parameter(torch.log(torch.ones(n_components)/float(n_components)), requires_grad=False)

    def fit(self, x, max_iterations=20):
        """
        Estimate Maximum Likelihood MPPCA parameters for the provided data using EM per
        Tipping, and Bishop. Mixtures of probabilistic principal component analyzers.
        :param x: training data (arranged in rows), shape = (<numbr of samples>, n_features)
        :param max_iterations: number of iterations
        :param feature_sampling: allows faster responsibility calculation by sampling data coordinates
        """
        K, d, l = self.A.shape
        N = x.shape[0]
        x = x.to(self.MU.device)

        init_samples_per_component = len(x) // self.n_components
        self._init_from_data(x, samples_per_component=init_samples_per_component)

        def per_component_m_step(i):
            mu_i = torch.sum(r[:, [i]] * x, dim=0) / r_sum[i]
            s2_I = torch.exp(self.log_D[i, 0]) * torch.eye(l, device=x.device)
            inv_M_i = torch.inverse(self.A[i].T @ self.A[i] + s2_I)
            x_c = x - mu_i.reshape(1, d)
            SiAi = (1.0/r_sum[i]) * (r[:, [i]]*x_c).T @ (x_c @ self.A[i])
            invM_AT_Si_Ai = inv_M_i @ self.A[i].T @ SiAi
            A_i_new = SiAi @ torch.inverse(s2_I + invM_AT_Si_Ai)
            t1 = torch.trace(A_i_new.T @ (SiAi @ inv_M_i))
            trace_S_i = torch.sum(N/r_sum[i] * torch.mean(r[:, [i]]*x_c*x_c, dim=0))
            sigma_2_new = (trace_S_i - t1)/d
            return mu_i, A_i_new, torch.log(sigma_2_new) * torch.ones_like(self.log_D[i])

        for it in range(max_iterations):
            r = self.responsibilities(x)
            r_sum = torch.sum(r, dim=0)
            new_params = [torch.stack(t) for t in zip(*[per_component_m_step(i) for i in range(K)])]
            self.MU.data = new_params[0]
            self.A.data = new_params[1]
            self.log_D.data = new_params[2]
            self.PI_logits.data = torch.log(r_sum / torch.sum(r_sum))
            ll = round(torch.mean(self.log_prob(x)).item(), 1) if it % 5 == 0 else '.....'

    def _init_from_data(self, x, samples_per_component):
        n = x.shape[0]
        K, d, l = self.A.shape

        m = samples_per_component
        o = np.random.choice(n, m*K, replace=False) if m*K < n else np.arange(n)
        assert n >= m*K
        component_samples = [[o[i*m:(i+1)*m]] for i in range(K)]

        params = [torch.stack(t) for t in zip(*[MFA._small_sample_ppca(x[component_samples[i]], n_factors=l) for i in range(K)])]

        self.MU.data = params[0]
        self.A.data = params[1]
        self.log_D.data = params[2]

    def sample(self, n, with_noise=False):
        """
        Generate random samples from the trained MFA / MPPCA
        :param n: How many samples
        :param with_noise: Add the isotropic / diagonal noise to the generated samples
        :return: samples [n, n_features], c_nums - originating component numbers
        """
        K, d, l = self.A.shape
        c_nums = np.random.choice(K, n, p=torch.softmax(self.PI_logits, dim=0).detach().cpu().numpy())
        z_l = torch.randn(n, l, device=self.A.device).double()
        z_d = torch.randn(n, d, device=self.A.device).double() if with_noise else torch.zeros(n, d, device=self.A.device).double()
        samples = torch.stack([self.A[c_nums[i]] @ z_l[i] + self.MU[c_nums[i]] + z_d[i] * torch.exp(0.5*self.log_D[c_nums[i]].double()) for i in range(n)])
        return samples

    def reconstruct(self, x):
        """
        Reconstruct samples from the model - find the MAP component and latent z for each sample and regenerate

        :param full_x: the full vectors (including the hidden coordinates, which can contain any values)
        :param observed_features: tensor containing a list of the observed coordinates of x
        :return: Reconstruction of full_x based on the trained model and observed features
        """
        K, d, l = self.A.shape
        c_i = self.map_component(x)

        AT = self.A.transpose(1, 2)
        iD = torch.exp(-self.log_D).unsqueeze(2)
        L = torch.eye(l, device=self.MU.device).reshape(1, l, l) + AT @ (iD*self.A)
        iL = torch.inverse(L)

        # per eq. 2 in Ghahramani and Hinton 1996 + the matrix inversion lemma (also described there).
        x_c = (x - self.MU[c_i]).unsqueeze(2)
        iD_c = iD[c_i]
        m_d_1 = (iD_c * x_c) - ((iD_c * self.A[c_i]) @ iL[c_i]) @ (AT[c_i] @ (iD_c * x_c))
        mu_z = AT[c_i] @ m_d_1
        return (self.A[c_i] @ mu_z).reshape(-1, d) + self.MU[c_i]

    def map_component(self, x):
        return torch.argmax(self.log_responsibilities(x), dim=1)

    def log_prob(self, x):
        return torch.logsumexp(self.per_component_log_likelihood(x), dim=1)

    def per_component_log_likelihood(self, x):
        """
        Calculate per-sample and per-component log-likelihood values
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: log-probability values [n, n_components]
        """
        return MFA._component_log_likelihood(x, torch.softmax(self.PI_logits, dim=0), self.MU, self.A, self.log_D)

    def log_responsibilities(self, x):
        comp_LLs = self.per_component_log_likelihood(x)
        return comp_LLs - torch.logsumexp(comp_LLs, dim=1).reshape(-1, 1)

    def responsibilities(self, x):

        return torch.exp(self.log_responsibilities(x))

    @staticmethod
    def _component_log_likelihood(x, PI, MU, A, log_D):
        K, d, l = A.shape
        AT = A.transpose(1, 2)
        iD = torch.exp(-log_D).view(K, d, 1)
        L = torch.eye(l, device=A.device).reshape(1, l, l) + AT @ (iD*A)
        iL = torch.inverse(L)

        def per_component_md(i):
            x_c = (x - MU[i].reshape(1, d)).T  # shape = (d, n)
            m_d_1 = (iD[i] * x_c) - ((iD[i] * A[i]) @ iL[i]) @ (AT[i] @ (iD[i] * x_c))
            return torch.sum(x_c * m_d_1, dim=0)

        m_d = torch.stack([per_component_md(i) for i in range(K)])
        det_L = torch.logdet(L)
        log_det_Sigma = det_L - torch.sum(torch.log(iD.reshape(K, d)), axis=1)
        log_prob_data_given_components = -0.5 * ((d*np.log(2.0*math.pi) + log_det_Sigma).reshape(K, 1) + m_d)
        return PI.reshape(1, K) + log_prob_data_given_components.T

    @staticmethod
    def _small_sample_ppca(x, n_factors):
        # See https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        mu = torch.mean(x, dim=0)
        # U, S, V = torch.svd(x - mu.reshape(1, -1))    # torch svd is less memory-efficient
        U, S, V = np.linalg.svd((x - mu.reshape(1, -1)).cpu().numpy(), full_matrices=False)
        V = torch.from_numpy(V.T).to(x.device)
        S = torch.from_numpy(S).to(x.device)
        sigma_squared = torch.sum(torch.pow(S[n_factors:], 2.0))/((x.shape[0]-1) * (x.shape[1]-n_factors))
        A = V[:, :n_factors] * torch.sqrt((torch.pow(S[:n_factors], 2.0).reshape(1, n_factors)/(x.shape[0]-1) - sigma_squared))
        return mu, A, torch.log(sigma_squared) * torch.ones(x.shape[1], device=x.device)



if __name__ == '__main__':
    data = read_data('../data/100-shot-obama_128/img')
    data = data.reshape(len(data), -1)
    num_iterations = 30
    device = torch.device("cuda:0")
    model = MFA(n_components=5, n_features=data.shape[1], n_factors=10).to(device)
    model.fit(data, max_iterations=num_iterations)

    rnd_samples = model.sample(64, with_noise=False)
    img_dim = int(np.sqrt(rnd_samples.shape[1] / 3))
    save_image(rnd_samples.reshape(-1, 3, img_dim, img_dim).clip(-1, 1), "samples.png", normalize=True)