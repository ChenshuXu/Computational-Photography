"""
You can use this file to execute your code. You are NOT required
to use this file, and ARE ALLOWED to make ANY changes you want in
THIS file. This file will not be submitted with your assignment
or report, so if you write code for above & beyond effort, make sure
that you include important snippets in your writeup. CODE ALONE IS
NOT SUFFICIENT FOR ABOVE AND BEYOND CREDIT.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""
import cv2

import os
import errno

from os import path

import panorama as pano

NUM_MATCHES = 50
SRC_FOLDER = "images/source"
OUT_FOLDER = "images/output"
IMG_EXTS = set(["png", "jpeg", "jpg", "gif", "tiff", "tif", "raw", "bmp"])


def main(image_files, output_folder):
    """Generate a panorama from the images in the source folder.

    NOTE: This pipeline _can_ result in large voids near the panorama edges
    for some inputs that a more sophisticated pipeline could avoid.
    """

    inputs = ((name, cv2.imread(name)) for name in sorted(image_files)
              if path.splitext(name)[-1][1:].lower() in IMG_EXTS)

    # start with the first image in the folder and process each image in order
    name, pano_img = next(inputs)
    print("\n  Starting with: {}".format(name))
    for name, next_img in inputs:

        if next_img is None:
            print("\nUnable to proceed: {} failed to load.".format(name))
            return

        print("  Adding {}".format(name))
        pano_img = pano.blendImagePair(pano_img, next_img, NUM_MATCHES)

    cv2.imwrite(path.join(output_folder, "output.jpg"), pano_img)
    print("  Done!")


if __name__ == "__main__":
    """Generate panoramas from the images in each subdirectory of SRC_FOLDER"""

    subfolders = os.walk(SRC_FOLDER)
    next(subfolders)  # skip the root input folder
    for dirpath, dirnames, fnames in subfolders:

        image_dir = os.path.split(dirpath)[-1]
        output_dir = os.path.join(OUT_FOLDER, image_dir)

        try:
            os.makedirs(output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        print("Processing '" + image_dir + "' folder...")

        image_files = [os.path.join(dirpath, name) for name in fnames]

        main(image_files, output_dir)
