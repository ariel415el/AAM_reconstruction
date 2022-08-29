import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils.extract_dlip_landmarks import FaceLandmarkDetector


def cv2pt(images):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    pt_images = []
    for img in images:
        pt_images.append(transform(img / 255))
    return torch.stack(pt_images)

def add_edge_keypoint(kps, img_dim):
    edge_points = np.array([[0,0], [0, img_dim / 2], [0, img_dim-1],
                   [img_dim/2, 0], [img_dim/2, img_dim/2], [img_dim/2, img_dim-1],
                   [img_dim-1, 0], [img_dim-1, img_dim / 2], [img_dim-1, img_dim-1]])

    return np.concatenate([kps, np.repeat(edge_points[None, ], len(kps), axis=0)], axis=1)


def load_raw_data(images_path, save_dir, img_size, limit_data=None, add_edge_keypoints=False):
    if os.path.exists(os.path.join(save_dir, "kps.npy")):
        images, keypoints = np.load(os.path.join(save_dir, "images.npy")), np.load(os.path.join(save_dir, "kps.npy"))
    else:
        keypoints = []
        images = []
        landmark_detector = FaceLandmarkDetector()

        image_names = os.listdir(images_path)
        if limit_data is not None:
            image_names = image_names[:limit_data]
        for image_name in image_names:
            img = Image.open(os.path.join(images_path, image_name))
            img = img.resize((img_size, img_size))
            np_image = np.array(img)
            kps = landmark_detector.extract_landmarks(np_image)
            if np.any(kps.min(0) < 0) or np.any(kps.max(0) >= np_image.shape[:2]):
                continue

            keypoints.append(kps)
            images.append(np_image)

        images, keypoints = np.stack(images), np.stack(keypoints)

        np.save(os.path.join(save_dir, "kps.npy"), keypoints)
        np.save(os.path.join(save_dir, "images.npy"), images)

    if add_edge_keypoints:
        keypoints = add_edge_keypoint(keypoints, images.shape[1])

    print(f"Raw data loaded {images.shape}, {keypoints.shape}")
    return images, keypoints
