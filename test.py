"""
CS6475
Simple script to verify your installation
"""
import unittest

class TestInstall(unittest.TestCase):
    def test_scipy(self):
        import scipy

    def test_numpy(self):
        import numpy

    def test_cv2(self):
        import cv2

    def test_cv2_version(self):
        import cv2
        v = cv2.__version__.split(".")
        self.assertTrue(v[0] == '4' and v[1] == '0', 'Wrong OpenCV version.'
                                                     'Make sure you installed OpenCV 4.0.x.'
                                                     'All older OpenCV versions are not supported.')

    def test_ORB(self):
        import cv2
        test_orb = cv2.ORB_create()


if __name__ == '__main__':
    unittest.main()
