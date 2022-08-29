import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial


def GetBilinearPixel(imArr, posX, posY):
    # Get integer and fractional parts of numbers
    modXi = int(posX)
    modYi = int(posY)
    modXf = posX - modXi
    modYf = posY - modYi

    # Get pixels in four corners
    bl = imArr[modYi, modXi]
    br = imArr[modYi, min(modXi + 1, imArr.shape[1] - 1)]
    tl = imArr[min(modYi + 1, imArr.shape[0]-1), modXi]
    tr = imArr[min(modYi + 1, imArr.shape[0]-1), min(modXi + 1, imArr.shape[1]-1)]

    # Calculate interpolation
    b = modXf * br + (1. - modXf) * bl
    t = modXf * tr + (1. - modXf) * tl
    pxf = modYf * t + (1. - modYf) * b
    return (pxf + 0.5)  # Do fast rounding to integer


def PiecewiseAffineTransform(src_img, src_pts, dst_pts):
    # Split input shape into mesh

    output = np.zeros_like(src_img)

    tess = spatial.Delaunay(dst_pts)

    xmin, ymin = dst_pts.min(0).astype(int)
    xmax, ymax = np.ceil(dst_pts.max(0)).astype(int)

    # Find affine mapping from input positions to mean shape
    affine_transforms = []
    for i, tri in enumerate(tess.vertices):
        meanVertPos = np.hstack((src_pts[tri], np.ones((3, 1)))).transpose()
        shapeVertPos = np.hstack((dst_pts[tri, :], np.ones((3, 1)))).transpose()

        affine = np.dot(meanVertPos, np.linalg.inv(shapeVertPos))
        affine_transforms.append(affine)

    # Determine which tesselation triangle contains each pixel in the shape norm image
    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            dst_img_coord = np.array([i,j], dtype=np.float32)
            triangle_index = tess.find_simplex(dst_img_coord).item()

            if triangle_index != -1:
                # Calculate position in the input image
                affine = affine_transforms[triangle_index]
                dst_img_homo_coord = np.array([i, j, 1], dtype=np.float32)
                src_img_coord = np.dot(affine, dst_img_homo_coord)[:2]

                # Check destination pixel is within the image
                if np.any(src_img_coord < 0) or np.any(src_img_coord >= src_img.shape[:2]):
                    continue
                output[j, i] = GetBilinearPixel(src_img, src_img_coord[0], src_img_coord[1])
                # outArr[j, i] = src_img[int(round(outImgCoord[1])),int(round(outImgCoord[0]))]

    return xmin, xmax, ymin, ymax, output


if __name__ == "__main__":
    from PIL import Image

    id1 = 166
    id2 = 44
    ffhq_pts = np.load('../ffhq-kps.npy')
    im_1 = np.array(Image.open(f'/mnt/storage_ssd/datasets/FFHQ_128/{str(id1).zfill(5)}.png'))
    im_2 = np.array(Image.open(f'/mnt/storage_ssd/datasets/FFHQ_128/{str(id2).zfill(5)}.png'))
    pts_1 = ffhq_pts[id1]
    pts_2 = ffhq_pts[id2]

    plt.imshow(im_1)
    # plt.plot(pts_1[:,0], pts_1[:,1], 'o')
    plt.show()

    plt.imshow(im_2)
    # plt.plot(pts_2[:,0], pts_2[:,1], 'o')
    plt.show()

    out = PiecewiseAffineTransform(im_1, pts_1, pts_2)

    plt.imshow(out[-1])
    plt.show()