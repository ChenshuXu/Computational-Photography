import cv2
import numpy as np
import scipy as sp
import unittest

from os import path

import hdr as hdr
from matplotlib import pyplot as plt

"""
You can use this file as a starting point to write your own unit tests
for this assignment. You are encouraged to discuss testing with your
peers, but you may not share code directly. Your code is scored based
on test cases performed by the autograder upon submission -- these test
cases will not be released.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""

IMG_FOLDER = "images/source/sample"

def plotHistogramCDF(histogram, cdf, plot_filename):
    """ Enable at end of file, default=False
        This code generates a basic histogram plot with a cdf line.
    """
    histogramNorm = minmaxNorm(histogram, newmax=1.)
    cdfNorm = minmaxNorm(cdf, newmax=1.)

    x = np.arange(256)
    hist = np.squeeze(histogramNorm)
    plt.bar(x, hist)
    plt.plot(x, cdfNorm, 'r')
    plt.title(plot_filename)

    filename = plot_filename + '.png'
    plt.savefig(filename)
    plt.close()
    return


def minmaxNorm(array, newmax=1.0):
    """ Takes a np.array and normalizes it between 0 and newmax. """
    arrmin = np.min(array)
    arrmax = np.max(array)
    return newmax*(array-arrmin)/(arrmax-arrmin)

class Assignment8Test(unittest.TestCase):

    def setUp(self):
        # images = [cv2.imread(path.join(IMG_FOLDER, "sample-00.png")),
        #           cv2.imread(path.join(IMG_FOLDER, "sample-01.png")),
        #           cv2.imread(path.join(IMG_FOLDER, "sample-02.png")),
        #           cv2.imread(path.join(IMG_FOLDER, "sample-03.png")),
        #           cv2.imread(path.join(IMG_FOLDER, "sample-04.png")),
        #           cv2.imread(path.join(IMG_FOLDER, "sample-05.png"))]
        #
        # if not all([im is not None for im in images]):
        #     raise IOError("Error, one or more sample images not found.")
        #
        # self.images = images
        # self.exposures = np.float64([1 / 160.0, 1 / 125.0, 1 / 80.0,
        #                              1 / 60.0, 1 / 40.0, 1 / 15.0])
        pass

    def test_applyHistogramEqualization_with_clip(self):
        basicHDR = cv2.imread("images/output/sample/basicHDR.png")
        histogram = hdr.computeHistogram(basicHDR)
        pdf = hdr.clip_pdf(histogram, 1000)
        cumulative_density = hdr.computeCumulativeDensity(pdf)
        plotHistogramCDF(pdf, cumulative_density, "histCDF")
        hist_eq = hdr.applyHistogramEqualization(basicHDR, cumulative_density)
        cv2.imwrite("histEQ_clip.png", hist_eq)

    def test_best_hdr(self):
        basicHDR = cv2.imread("images/output/sample/basicHDR.png")
        bestHDR = hdr.bestHDR(basicHDR)
        cv2.imwrite("bestHDR.png", bestHDR)

    def test_increase_v(self):
        best_hdr_image = cv2.imread("images/output/sample/bestHDR.png")
        hsv = cv2.cvtColor(best_hdr_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] += 10
        best_hdr_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite("bestHDR.png", best_hdr_image)

    def test_clahe(self):
        basicHDR = cv2.imread("images/output/sample/basicHDR.png")
        hsv_image = cv2.cvtColor(basicHDR, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
        clahe_v = clahe.apply(v)

        clahe_image_hsv = cv2.merge((h, s, clahe_v))
        clahe_image = cv2.cvtColor(clahe_image_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite("clahe_image.png", clahe_image)


if __name__ == '__main__':
    unittest.main()
