import os

from matplotlib import pyplot as plt


def plot_keypoint_modes(kps_model, save_dir, img=None, img_kps=None):
    scales = [-50, 0, 50]

    fig, axs = plt.subplots(3,3, figsize=(10, 15))
    keypoint_mean_reshaped = (kps_model.mean).reshape(kps_model.shape)
    for i in range(3):
        for j in range(len(scales)):
            keypoints = (kps_model.mean + scales[j] * kps_model.eigenvecs[i]).reshape(kps_model.shape)

            if img is not None:
                from utils.warp_utils import PiecewiseAffineTransform
                xmin, xmax, ymin, ymax, warped_image = PiecewiseAffineTransform(img, img_kps, keypoints)
                axs[i,j].imshow(warped_image / 255)
            else:
                axs[i, j].plot(keypoints[:, 0], keypoints[:, 1], 'o', c='r', label=f"mean {scales[j]} mode-{j}")
                axs[i, j].plot(keypoint_mean_reshaped[:, 0], keypoint_mean_reshaped[:, 1], 'o', c='b', label='mean')
                axs[i, j].invert_yaxis()

    plt.legend()
    if img is not None:
        plt.savefig(os.path.join(save_dir, "keypoint_modes_img.png"))
    else:
        plt.savefig(os.path.join(save_dir, "keypoint_modes.png"))


def plot_texture_modes(texture_model, save_dir):
    scales = [-10000, 0, 10000]
    fig, axs = plt.subplots(3,3, figsize=(15, 15))
    for i in range(3):
        for j in range(len(scales)):
            axs[i, j].imshow((texture_model.mean + scales[j] * texture_model.eigenvecs[i]).reshape(texture_model.shape) / 255)

    plt.legend()
    plt.savefig(os.path.join(save_dir, "texture_modes.png"))
