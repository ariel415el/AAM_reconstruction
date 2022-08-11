import os
import numpy as np
from PIL import Image
import pickle

def read_pts_file(path):
    with open(path) as f:
        rows = [rows.strip() for rows in f]

    """Use the curly braces to find the start and end of the point data"""
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [tuple([float(point) for point in coords]) for coords in coords_set]
    return np.array(points)

if __name__ == '__main__':
    new_data = '/cs/labs/yweiss/ariel1/data/300W/processed'
    os.makedirs(new_data, exist_ok=True)
    data_roots = ['/cs/labs/yweiss/ariel1/data/300W/01_Indoor', '/cs/labs/yweiss/ariel1/data/300W/02_Outdoor']

    all_pts = []
    all_images = []

    for data_root in data_roots:
        names = set([os.path.splitext(x)[0] for x in os.listdir(data_root)])

        for name in names:
            try:
                pts = read_pts_file(os.path.join(data_root, name + ".pts"))
                image = Image.open(os.path.join(data_root, name + ".png"))
            except Exception as e:
                print("Bad file")
                continue
            w, h = image.size


            x_min, y_min = pts.min(0)
            x_max, y_max = pts.max(0)

            size = max((x_max - x_min) , (y_max - y_min)) * 1.15 / 2

            mean = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])

            upper_left = mean - size
            bottom_right = mean + size

            if np.any(upper_left < 0) or np.any(bottom_right > np.array([h,w])):
                continue

            print("Image shape:", h, w, (x_min, y_min), (x_max, y_max), upper_left, bottom_right)
            try:
                image = image.crop((upper_left[0], upper_left[1], bottom_right[0], bottom_right[1]))
            except Exception as e:
                print("cant crop")
                continue

            assert np.all(pts.max(0) < np.array([h,w])), pts.max(0)

            pts = pts - upper_left

            assert pts.max() <= image.size[0]

            pts *= 256 / image.size[0]
            image = image.resize((256, 256))

            image.save(os.path.join(new_data, name + ".png"))
            np.save(os.path.join(new_data, name + ".npy"), pts)

            all_pts.append(pts)

    np.save(os.path.join(new_data, "../..", "mean_pts.npy"), np.mean(all_pts, axis=0))
