from PIL import Image
import numpy as np


def load_image(path):
    img = np.array(Image.open(path))
    img = img.astype(np.float64)
    # img = img * 2 - 1
    # img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    return img


