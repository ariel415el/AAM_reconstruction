import os
import pickle

import numpy as np
import torch


class Model:
    def __init__(self, eigenvecs, eigenvals, mean, shape):
        self.eigenvecs = eigenvecs
        self.eigenvals = eigenvals
        self.mean = mean
        self.shape = shape

    def reconstruct(self, sample, n_components):
        projection_matrix = self.eigenvecs[:n_components]
        reconstruction = (sample.reshape(-1) - self.mean) @ projection_matrix.T @ projection_matrix + self.mean
        return reconstruction.reshape(self.shape)


def learn_pca_model(samples, save_path):
    if os.path.exists(save_path):
        model = pickle.load(open(save_path, 'rb'))
    else:
        shape = samples.shape[1:]
        vecs = samples.reshape(len(samples), -1)
        eigenvecs, eigenvals, mean = learn_pca(vecs)

        model = Model(eigenvecs=eigenvecs, eigenvals=eigenvals, mean=mean, shape=shape)

        pickle.dump(model, open(save_path, 'wb'))

    return model


def learn_pca(data, use_torch=True):
    print(f"Learning PCA on data of shape {data.shape} ...")
    if use_torch:
        data = torch.from_numpy(data).float().cuda()
        data_mean = data.mean(0)
        x = (data - data_mean).T @ (data - data_mean)
        U, S, V = torch.svd(x)
        eigenvalues, eigenvectors, data_mean =  V.T.cpu().numpy(), S.cpu().numpy(), data_mean.cpu().numpy()
    else:
        data_mean = data.mean(0)
        U, S, V = np.linalg.svd((data - data_mean).T @ (data - data_mean))
        eigenvalues, eigenvectors, data_mean =  U.T, S, data_mean
    print("Done")
    return eigenvalues, eigenvectors, data_mean