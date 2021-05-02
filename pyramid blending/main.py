"""
You can use this file to execute your code. You are NOT required
to use this file, and ARE ALLOWED to make ANY changes you want in
THIS file. This file will not be submitted with your assignment
or report.

DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""
import cv2
import numpy as np

import os
import errno

from os import path
from glob import glob

from blending import (gaussPyramid, laplPyramid, blend, collapse)

MIN_DEPTH = 4
IMG_EXTENSIONS = ["png", "jpeg", "jpg", "tiff", "tif", "bmp"]
SRC_FOLDER = "images/source"
OUT_FOLDER = "images/output"


def normalize(img):
    return cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def collect_files(prefix, extension_list=IMG_EXTENSIONS):
    """Return a list of all files in a directory that match the input prefix
    with one of the allowed extensions. """
    filenames = sum(map(glob, [prefix + ext for ext in extension_list]), [])
    return filenames


def viz_pyramid(stack, shape, name, norm=False):
    """Create a single image by vertically stacking the levels of a pyramid."""
    layers = [normalize(np.dstack(imgs)) if norm else np.clip(np.dstack(imgs), 0, 255) for imgs in zip(*stack)]
    stack = [cv2.resize(layer, shape, interpolation=3) for layer in layers]
    img = np.vstack(stack).astype(np.uint8)
    cv2.imwrite(name + ".png", img)
    return img


def main(black_image, white_image, mask, out_path, min_depth=MIN_DEPTH):
    """Apply pyramid blending to each color channel of the input images """

    # Convert to double and normalize the mask to the range [0..1]
    # to avoid arithmetic overflow issues
    black_image = np.atleast_3d(black_image).astype(np.float)
    white_image = np.atleast_3d(white_image).astype(np.float)
    mask_img = np.atleast_3d(mask).astype(np.float) / 255.

    shape = mask_img.shape[1::-1]
    min_size = min(black_image.shape[:2])
    depth = int(np.log2(min_size)) - min_depth

    gauss_pyr_mask = [gaussPyramid(ch, depth) for ch in np.rollaxis(mask_img, -1)]
    gauss_pyr_black = [gaussPyramid(ch, depth) for ch in np.rollaxis(black_image, -1)]
    gauss_pyr_white = [gaussPyramid(ch, depth) for ch in np.rollaxis(white_image, -1)]
    viz_pyramid(gauss_pyr_mask, shape, path.join(out_path, 'gauss_pyr_mask'), norm=True)
    viz_pyramid(gauss_pyr_black, shape, path.join(out_path, 'gauss_pyr_black'))
    viz_pyramid(gauss_pyr_white, shape, path.join(out_path, 'gauss_pyr_white'))

    lapl_pyr_black = [laplPyramid(ch) for ch in gauss_pyr_black]
    lapl_pyr_white = [laplPyramid(ch) for ch in gauss_pyr_white]
    viz_pyramid(lapl_pyr_black, shape, path.join(out_path, 'lapl_pyr_black'), norm=True)
    viz_pyramid(lapl_pyr_white, shape, path.join(out_path, 'lapl_pyr_white'), norm=True)

    outpyr = [blend(*x) for x in zip(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)]
    result = [[collapse(x)] for x in outpyr]
    viz_pyramid(outpyr, shape, path.join(out_path, 'outpyr'), norm=True)
    viz_pyramid(result, shape, path.join(out_path, 'result'))


if __name__ == "__main__":
    """Apply pyramid blending to all folders below SRC_FOLDER that contain
    a black, white, and mask image.
    """
    subfolders = os.walk(SRC_FOLDER)
    next(subfolders)  # skip the root input folder
    for dirpath, dirnames, fnames in subfolders:

        image_dir = os.path.split(dirpath)[-1]
        output_dir = os.path.join(OUT_FOLDER, image_dir)

        print("Processing files in '" + image_dir + "' folder...")

        black_names = collect_files(os.path.join(dirpath, '*black.'))
        white_names = collect_files(os.path.join(dirpath, '*white.'))
        mask_names = collect_files(os.path.join(dirpath, '*mask.'))

        if not len(black_names) == len(white_names) == len(mask_names) == 1:
            print("    Cannot proceed. There can only be one black, white, " +
                  "and mask image in each input folder.")
            continue

        try:
            os.makedirs(output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        black_img = cv2.imread(black_names[0], cv2.IMREAD_COLOR)
        white_img = cv2.imread(white_names[0], cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_names[0], cv2.IMREAD_COLOR)

        main(black_img, white_img, mask_img, output_dir)
