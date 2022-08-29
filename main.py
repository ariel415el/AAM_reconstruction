import argparse
import os
import numpy as np

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from models.mfa import MFA
from utils.data_utils import load_raw_data
from utils.warp_utils import PiecewiseAffineTransform


def get_shape_normalized_data(images, keypoints, save_dir):
    print("Warping and centering images...", end='', flush=True)
    file_path = os.path.join(save_dir, "shape_normalized_textures.npy")

    if os.path.exists(file_path):
        res =  np.load(file_path)
    else:
        mean_kps = keypoints.mean(0)
        shape_normalized_images = []
        for img, kps in tqdm(list(zip(images, keypoints))):

            xmin, xmax, ymin, ymax, warped_image = PiecewiseAffineTransform(img, kps, mean_kps)

            shape_normalized_images.append(warped_image[ymin:ymax, xmin:xmax])

        res = np.stack(shape_normalized_images)
        np.save(file_path, res)

    print(f"Centered images shape: {res.shape}")
    return res


def get_mppca_model(data, n_components, n_factors, file_path):
    b = data.shape[0]
    n_features = np.prod(data.shape[1:])
    model = MFA(n_components, n_features, n_factors).to(device)

    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        model.double()
    else:
        flat_data = torch.from_numpy(data).reshape(b, -1).double()
        print("Fitting MPPCA...", end='')
        model.fit(flat_data, max_iterations=30)
        torch.save(model.state_dict(), file_path)
        print("Done")

    return model


def reconstruct_image(img, img_kps, mean_kps, texture_model):

    xmin, xmax, ymin, ymax, centered_face = PiecewiseAffineTransform(img, img_kps, mean_kps)

    centered_face = centered_face[ymin:ymax, xmin:xmax]
    reconstruction = texture_model.reconstruct(torch.from_numpy(centered_face.reshape(1, -1)).to(device)).clip(0,255)
    reconstruction = reconstruction.cpu().numpy().reshape(centered_face.shape)

    new_imag = np.zeros_like(img)
    new_imag[ymin:ymax, xmin:xmax] = reconstruction

    xmin_d, xmax_d, ymin_d, ymax_d, warped_back = PiecewiseAffineTransform(new_imag, mean_kps, img_kps)

    full_reconstruction = np.where(warped_back != 0, warped_back, img)

    return centered_face, reconstruction, full_reconstruction


def morph_from_mean(centered_texture, mean_kps, target_kps, img_shape):
    xmin, ymin = mean_kps.min(0).astype(int)
    xmax, ymax = np.ceil(mean_kps.max(0)).astype(int)
    new_imag = np.zeros((img_shape[0], img_shape[1], 3))
    new_imag[ymin:ymax, xmin:xmax] = centered_texture

    _, _, _, _, new_face = PiecewiseAffineTransform(new_imag,
                                                    mean_kps,
                                                    target_kps)

    return new_face


def visualize_samples(texture_model, keypoints, org_shape, out_dir):
    mean_kps = keypoints.mean(0)

    # h,w = shape_normalized_textures.shape[1:3]

    xmin, ymin = mean_kps.min(0).astype(int)
    xmax, ymax = np.ceil(mean_kps.max(0)).astype(int)
    h, w = ymax - ymin, xmax - xmin

    sampled_textures = texture_model.sample(9, with_noise=False).reshape(-1, h, w, 3)

    nrow = int(np.sqrt(len(sampled_textures)))
    save_image(sampled_textures.permute(0, 3, 1, 2).clip(0, 255), os.path.join(out_dir, "samples.png"), normalize=True, nrow=nrow)
    faces = np.stack([
        morph_from_mean(sampled_textures[i].cpu().numpy(), mean_kps, keypoints[np.random.randint(0,len(keypoints))], org_shape)
        for i in range(len(sampled_textures))
    ])
    faces = torch.from_numpy(faces)
    save_image(faces.permute(0, 3, 1, 2).clip(0, 255), os.path.join(out_dir, "samples_morphed.png"), normalize=True, nrow=nrow)

def visualize_reconstruction(texture_model, test_images, test_keypoints, mean_kps, b, out_dir):
    centered_faces, reconstructions, full_reconstructions = [], [], []
    b = min(b, len(test_images))
    for i in range(b):
        centered_face, reconstruction, full_reconstruction = reconstruct_image(test_images[i], test_keypoints[i], mean_kps, texture_model)
        centered_faces.append(centered_face)
        reconstructions.append(reconstruction)
        full_reconstructions.append(full_reconstruction)
    nrow = int(np.sqrt(b))
    save_image(torch.from_numpy(test_images[:b]).permute(0, 3, 1, 2).double(),
               os.path.join(out_dir, "test_faces.png"), normalize=True, nrow=nrow)
    save_image(torch.from_numpy(np.stack(centered_faces)).permute(0, 3, 1, 2).double(),
               os.path.join(out_dir, "test_faces_centered.png"), normalize=True, nrow=nrow)
    save_image(torch.from_numpy(np.stack(reconstructions)).permute(0, 3, 1, 2).clip(0,255).double(),
               os.path.join(out_dir, "test_faces_reconstructions.png"), normalize=True, nrow=nrow)
    save_image(torch.from_numpy(np.stack(full_reconstructions)).permute(0, 3, 1, 2).clip(0,255).double(),
               os.path.join(out_dir, "test_faces_full_reconstructions.png"), normalize=True, nrow=nrow)

def main(args):
    out_dir = f"Outputs/{os.path.basename(args.images_dir)}" + ('Full' if args.add_edge_keypoints else '')
    os.makedirs(out_dir, exist_ok=True)

    # Load raw data
    images, keypoints = load_raw_data(args.images_dir, out_dir, args.img_size, limit_data=1000, add_edge_keypoints=args.add_edge_keypoints)
    org_shape = images.shape[1:3]

    # Split data
    n = int(0.9 * len(images))
    test_images, test_keypoints = images[n:], keypoints[n:]
    images, keypoints = images[:n], keypoints[:n]

    # Normalize shapes data
    shape_normalized_textures = get_shape_normalized_data(images, keypoints, out_dir)

    # Fit texture
    texture_model = get_mppca_model(shape_normalized_textures, args.n_components, args.n_factors, file_path=os.path.join(out_dir, "texture_model.pt"))

    # Visualizations
    visualize_samples(texture_model, keypoints, org_shape, out_dir)
    visualize_reconstruction(texture_model, test_images, test_keypoints, keypoints.mean(0), 9, out_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', required=True)
    parser.add_argument('--n_components', help='Number of PPCA components', defaulat=25)
    parser.add_argument('--n_factors', help='size of the latent space', defaulat=16)
    parser.add_argument('--img_size', help='Resize the images', defaulat=128)
    parser.add_argument('--add_edge_keypoints', help='model the entire image by adding keypoints on edges', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0")
    main(args)