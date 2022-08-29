import os

import dlib
import numpy as np

os.path.dirname(__file__)
class FaceLandmarkDetector:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

    def extract_landmarks(self, img):
        dets, scores, idx = self.face_detector.run(img, 1, -1)
        parts = self.landmark_predictor(img, dets[0]).parts()
        face_landmarks = np.array([(item.x, item.y) for item in parts])
        return face_landmarks