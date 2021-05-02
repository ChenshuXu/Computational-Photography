import cv2
import numpy as np
import scipy as sp
import unittest

from os import path

import panorama as pano

"""
You can use this file as a starting point to write your own unit tests
for this assignment. You are encouraged to discuss testing with your
peers, but you may not share code directly. Your code is scored based
on test cases performed by the autograder upon submission -- these test
cases will not be released.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""

IMG_FOLDER = "images/source/sample"


def write_mask_image(mask, file_name):
    mask_image = mask.astype(np.uint8) * 255
    cv2.imwrite(file_name, mask_image)


class Assignment8Test(unittest.TestCase):

    def setUp(self):
        images = [cv2.imread(path.join(IMG_FOLDER, "1.jpg")),
                  cv2.imread(path.join(IMG_FOLDER, "2.jpg")),
                  cv2.imread(path.join(IMG_FOLDER, "3.jpg"))]

        if any([im is None for im in images]):
            raise IOError("Error, one or more sample images not found.")

        self.images = images

    def test_getImageCorners(self):
        # getCorners_cv2 = []
        # cv2.getPerspectiveTransform(self.images[0], getCorners_cv2)
        # print(getCorners_cv2)
        print(self.images[0].shape)
        getCorners_my = pano.getImageCorners(self.images[0])
        print(getCorners_my)

    def test_createImageMask(self):
        image_1 = self.images[0]
        image_2 = self.images[1]
        num_matches = 10
        kp1, kp2, matches = pano.findMatchesBetweenImages(image_1, image_2, num_matches)
        homography = pano.findHomography(kp1, kp2, matches)
        corners_1 = pano.getImageCorners(image_1)
        corners_2 = pano.getImageCorners(image_2)
        min_xy, max_xy = pano.getBoundingCorners(corners_1, corners_2, homography)
        left_image = pano.warpCanvas(image_1, homography, min_xy, max_xy)
        right_image = np.zeros_like(left_image)
        min_xy = min_xy.astype(np.int)
        right_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],
        -min_xy[0]:-min_xy[0] + image_2.shape[1]] = image_2

        left_mask = pano.createImageMask(left_image)
        right_mask = pano.createImageMask(right_image)

        write_mask_image(left_mask, "left_mask_image.png")
        write_mask_image(right_mask, "right_mask_image.png")

    def test_createRegionMasks(self):
        image_1 = self.images[0]
        image_2 = self.images[1]
        num_matches = 10
        kp1, kp2, matches = pano.findMatchesBetweenImages(image_1, image_2, num_matches)
        homography = pano.findHomography(kp1, kp2, matches)
        corners_1 = pano.getImageCorners(image_1)
        corners_2 = pano.getImageCorners(image_2)
        min_xy, max_xy = pano.getBoundingCorners(corners_1, corners_2, homography)
        left_image = pano.warpCanvas(image_1, homography, min_xy, max_xy)
        right_image = np.zeros_like(left_image)
        min_xy = min_xy.astype(np.int)
        right_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],
        -min_xy[0]:-min_xy[0] + image_2.shape[1]] = image_2
        left_mask = pano.createImageMask(left_image)
        right_mask = pano.createImageMask(right_image)
        left_only_mask, overlap_mask, right_only_mask = pano.createRegionMasks(left_mask, right_mask)

        write_mask_image(left_only_mask, "left_only_mask.png")
        write_mask_image(overlap_mask, "overlap_mask.png")
        write_mask_image(right_only_mask, "right_only_mask.png")

    def test_findDistanceToMask(self):
        image_1 = self.images[0]
        image_2 = self.images[1]
        num_matches = 10
        kp1, kp2, matches = pano.findMatchesBetweenImages(image_1, image_2, num_matches)
        homography = pano.findHomography(kp1, kp2, matches)
        corners_1 = pano.getImageCorners(image_1)
        corners_2 = pano.getImageCorners(image_2)
        min_xy, max_xy = pano.getBoundingCorners(corners_1, corners_2, homography)
        left_image = pano.warpCanvas(image_1, homography, min_xy, max_xy)
        right_image = np.zeros_like(left_image)
        min_xy = min_xy.astype(np.int)
        right_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],
        -min_xy[0]:-min_xy[0] + image_2.shape[1]] = image_2

        left_mask = pano.createImageMask(left_image)
        right_mask = pano.createImageMask(right_image)
        left_only_mask, overlap_mask, right_only_mask = pano.createRegionMasks(left_mask, right_mask)

        left_distance_mask = pano.findDistanceToMask(left_mask)
        left_only_distance_mask = pano.findDistanceToMask(left_only_mask)
        write_mask_image(left_only_distance_mask, "left_only_distance_mask.png")
        overlap_distance_mask = pano.findDistanceToMask(overlap_mask)
        right_distance_mask = pano.findDistanceToMask(right_mask)
        right_only_distance_mask = pano.findDistanceToMask(right_only_mask)
        write_mask_image(right_only_distance_mask, "right_only_distance_mask.png")

    def test_generateAlphaWeights(self):
        image_1 = self.images[0]
        image_2 = self.images[1]
        num_matches = 10
        kp1, kp2, matches = pano.findMatchesBetweenImages(image_1, image_2, num_matches)
        homography = pano.findHomography(kp1, kp2, matches)
        corners_1 = pano.getImageCorners(image_1)
        corners_2 = pano.getImageCorners(image_2)
        min_xy, max_xy = pano.getBoundingCorners(corners_1, corners_2, homography)
        left_image = pano.warpCanvas(image_1, homography, min_xy, max_xy)
        right_image = np.zeros_like(left_image)
        min_xy = min_xy.astype(np.int)
        right_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],
        -min_xy[0]:-min_xy[0] + image_2.shape[1]] = image_2

        left_mask = pano.createImageMask(left_image)
        right_mask = pano.createImageMask(right_image)
        left_only_mask, overlap_mask, right_only_mask = pano.createRegionMasks(left_mask, right_mask)

        left_distance_mask = pano.findDistanceToMask(left_mask)
        left_only_distance_mask = pano.findDistanceToMask(left_only_mask)
        overlap_distance_mask = pano.findDistanceToMask(overlap_mask)
        right_distance_mask = pano.findDistanceToMask(right_mask)
        right_only_distance_mask = pano.findDistanceToMask(right_only_mask)
        left_ratio_mask = pano.generateAlphaWeights(left_only_distance_mask, right_only_distance_mask)
        cv2.imwrite("ratio_mask.png", left_ratio_mask * 255)

    def test_blend(self):
        image_1 = self.images[0]
        image_2 = self.images[1]
        blend_1_2_image = pano.blendImagePair(image_1, image_2, 100)
        cv2.imwrite("blend_1_2_image.png", blend_1_2_image)

        # image_3 = self.images[2]
        # blend_2_3_image = pano.blendImagePair(image_2, image_3, 100)
        # cv2.imwrite("blend_2_3_image.png", blend_2_3_image)

if __name__ == '__main__':
    unittest.main()
