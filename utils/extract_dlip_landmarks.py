import os
import pickle

import dlib
import numpy as np
from matplotlib import pyplot as plt

from utils.image_utils import load_image


def extract_landmarks(data_root):
    paths = set([os.path.join(data_root, x) for x in os.listdir(data_root)])

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    res = dict()
    for path in paths:
        img = dlib.load_rgb_image(path)

        dets, scores, idx = detector.run(img, 1, -1)

        face_landmarks = np.array([(item.x, item.y) for item in predictor(img, dets[0]).parts()])

        img_name = os.path.splitext(os.path.basename(path))[0]
        res[img_name] = face_landmarks

    pickle.dump(res, open(os.path.join(os.path.dirname(data_root), "face_landmarks.pkl"), 'wb'))

if __name__ == '__main__':
    extract_landmarks('../data/100-shot-obama_128/img')