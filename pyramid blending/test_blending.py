import cv2
import numpy as np
import scipy as sp
import unittest

from os import path

import blending as blend

"""
You can use this file as a starting point to write your own unit tests
for this assignment. You are encouraged to discuss testing with your
peers, but you may not share code directly. Your code is scored based
on test cases performed by the autograder upon submission -- these test
cases will not be released.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""

IMG_FOLDER = "images/source/sample"


class Assignment2Test(unittest.TestCase):

    def setUp(self):
        self.black_img = cv2.imread(path.join(IMG_FOLDER, "black.jpg"))
        self.white_img = cv2.imread(path.join(IMG_FOLDER, "white.jpg"))
        self.mask_img = cv2.imread(path.join(IMG_FOLDER, "mask.jpg"))

        if any(map(lambda x: x is None,
                   [self.black_img, self.white_img, self.mask_img])):
            raise IOError("Error, samples image not found.")

    def test_reduce_layer(self):
        black_image = np.atleast_3d(self.black_img).astype(np.float)
        one_ch = np.rollaxis(black_image, -1)[0]

        reduce_cv2 = cv2.pyrDown(one_ch)
        reduce_manual = blend.reduce_layer(one_ch)
        cv2.imwrite("reduced_image_black.png", reduce_manual)
        self.assertTrue(np.allclose(reduce_manual, reduce_cv2, 1, 1))

    def test_expand_layer(self):
        black_image = np.atleast_3d(self.black_img).astype(np.float)
        one_ch = np.rollaxis(black_image, -1)[0]

        expand_cv2 = cv2.pyrUp(one_ch)
        expand_manual = blend.expand_layer(one_ch)
        cv2.imwrite("expanded_image_black.png", expand_manual)
        self.assertTrue(np.allclose(expand_manual, expand_cv2, 1, 1))

    def test_gauss_pyramid(self):
        lower_reso_cv2 = [self.black_img]
        for i in range(4):
            lower_reso_cv2.append(cv2.pyrDown(lower_reso_cv2[-1]))
        for i in range(len(lower_reso_cv2)):
            cv2.imwrite("lower_reso_cv2_" + str(i)+".png", lower_reso_cv2[i])

        lower_reso_manual = blend.gaussPyramid(self.black_img, 4)
        for i in range(len(lower_reso_manual)):
            cv2.imwrite("lower_reso_manual_" + str(i)+".png", lower_reso_manual[i])

        for i in range(4):
            self.assertTrue(np.allclose(lower_reso_manual[i], lower_reso_cv2[i], 1, 1))

    def test_lapl_pyramid(self):
        gauss_pyr = blend.gaussPyramid(self.black_img, 4)
        for i in range(len(gauss_pyr)):
            cv2.imwrite("gauss_pyr_" + str(i) + ".png", gauss_pyr[i])

        lapl_pyr = blend.laplPyramid(gauss_pyr)
        for i in range(len(lapl_pyr)):
            cv2.imwrite("lapl_pyr_" + str(i) + ".png", lapl_pyr[i])

        for i in range(len(gauss_pyr)):
            self.assertTrue(gauss_pyr[i].shape == lapl_pyr[i].shape)

if __name__ == '__main__':
    unittest.main()
