import os
import pickle
import numpy as np

import torch

from utils.debug_utils import plot_keypoint_modes, plot_texture_modes
from utils.pca import learn_pca_model
from utils.warp_utils import PiecewiseAffineTransform

from utils.image_utils import load_image



def load_raw_data(images_path, kps_file, start, end):
    all_kps = pickle.load(open(kps_file, 'rb'))

    keypoints = []
    images = []

    image_names = list(all_kps.keys())[start: end]
    for image_name in image_names:
        path = f'{images_path}/{image_name}.jpg'
        if not os.path.exists(path):
            path = path.replace(".jpg", ".png")
        image = load_image(path)
        kps = all_kps[image_name]
        if np.any(kps.min(0) < 0) or np.any(kps.max(0) >= image.shape[:2]):
            continue
        keypoints.append(kps)
        images.append(image)

    return np.stack(images), np.stack(keypoints)


def get_shape_normalized_data(images, keypoints, save_path):
    if os.path.exists(save_path):
        return np.load(save_path)
    mean_kps = keypoints.mean(0)
    shape_normalized_images = []
    for img, kps in zip(images, keypoints):

        xmin, xmax, ymin, ymax, warped_image = PiecewiseAffineTransform(img, kps, mean_kps)

        shape_normalized_images.append(warped_image[ymin:ymax, xmin:xmax])

    res = np.stack(shape_normalized_images)
    np.save(save_path, res)
    return res


def reconstruct_image(img, img_kps, kps_model, texture_model, n_pca_components, save_path):
    from matplotlib import pyplot as plt

    mean_kps = kps_model.mean.reshape(kps_model.shape)
    xmin, xmax, ymin, ymax, centered_face = PiecewiseAffineTransform(img, img_kps, mean_kps)

    reconstruction = texture_model.reconstruct(centered_face[ymin:ymax, xmin:xmax], n_pca_components)

    new_imag = np.zeros_like(img)
    new_imag[ymin:ymax, xmin:xmax] = reconstruction

    xmin_d, xmax_d, ymin_d, ymax_d, warped_back = PiecewiseAffineTransform(new_imag, mean_kps, img_kps)

    final = np.where(warped_back != 0 , warped_back, img)

    fig, axs = plt.subplots(3,2, figsize=(10, 10))
    axs[0,0].imshow(img / 255)
    axs[0,0].set_title("image")
    axs[0,0].axis('off')

    axs[0,1].imshow(img[ymin_d:ymax_d, xmin_d:xmax_d] / 255)
    axs[0,1].set_title("Cropped-image")
    axs[0,1].axis('off')

    axs[1, 0].imshow(centered_face[ymin:ymax, xmin:xmax] / 255)
    axs[1, 0].set_title("centered_face")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(reconstruction / 255)
    axs[1, 1].set_title("reconstruction")
    axs[1, 1].axis('off')

    axs[2, 0].imshow(new_imag / 255)
    axs[2, 0].set_title("warped_back")
    axs[2, 0].axis('off')

    axs[2, 1].imshow(final / 255)
    axs[2, 1].set_title("Final")
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()


def add_edge_keypoint(kps, img_dim):
    edge_points = np.array([[0,0], [0, img_dim / 2], [0, img_dim-1],
                   [img_dim/2, 0], [img_dim/2, img_dim/2], [img_dim/2, img_dim-1],
                   [img_dim-1, 0], [img_dim-1, img_dim / 2], [img_dim-1, img_dim-1]])

    return np.concatenate([kps, np.repeat(edge_points[None, ], len(kps), axis=0)], axis=1)


if __name__ == '__main__':
    images_dir = 'data/100-shot-obama_128/img'
    kps_file = 'data/100-shot-obama_128/face_landmarks.pkl'
    add_edge_keypoints = False
    out_dir = "Outputs" + ('Full' if add_edge_keypoints else '')

    with torch.no_grad():
        os.makedirs(out_dir, exist_ok=True)
        images, keypoints = load_raw_data(images_dir, kps_file,  0, 100)
        n = int(0.9 * len(images))
        images, keypoints = images[n:], keypoints[n:]
        test_images, test_keypoints = images[:n], keypoints[:n]
        if add_edge_keypoints:
            test_keypoints = add_edge_keypoint(test_keypoints, images.shape[1])
            keypoints = add_edge_keypoint(keypoints, images.shape[1])

        shape_normalized_textures = get_shape_normalized_data(images, keypoints, f"{out_dir}/shape_normalized_textures.npy", )

        texture_model = learn_pca_model(shape_normalized_textures, f"{out_dir}/texture_data.npy")
        kps_model = learn_pca_model(keypoints, f"{out_dir}/keypoint_data.npy")

        plot_keypoint_modes(kps_model, out_dir)
        plot_keypoint_modes(kps_model, out_dir, test_images[0], test_keypoints[0])
        plot_texture_modes(texture_model, out_dir)

        n_pca_components = 10
        for i in range(len(test_keypoints)):
            reconstruct_image(test_images[i], test_keypoints[i], kps_model, texture_model, n_pca_components, f"{out_dir}/recon-{i}.png")



