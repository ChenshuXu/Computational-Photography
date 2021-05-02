import unittest
import numpy as np
import scipy as sp
import cv2
from os import path
import cvxpy as cp

from matplotlib import pyplot as plt
import main
import final_project as stabilization


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.motion_video_file = 'videos/motion.avi'
        self.motion_video_output_dir = 'videos/test_output/motion'

        self.sample_video_file = 'videos/Yuna Kim Skate America 2009 [fan cam].mp4'
        self.sample_video_output_dir = 'videos/test_output/sample'
        # params for ShiTomasi corner detection
        self.sample_video_feature_params = dict(maxCorners=1000,
                                                qualityLevel=0.1,
                                                minDistance=10,
                                                blockSize=10)

        # Parameters for lucas kanade optical flow
        self.sample_video_lk_params = dict(winSize=(50, 50),
                                           maxLevel=0,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def test_write_frames_to_file(self):
        """write frames to image file"""
        frames, fps = stabilization.extract_frames_from_video(self.motion_video_file)
        for i in range(len(frames)):
            frame_filename = path.join(self.motion_video_output_dir, "frame_{}.jpg".format(i))
            cv2.imwrite(frame_filename, frames[i])

    def test_plot_draw_good_feature(self):
        """draw good features on a image"""
        frames, fps = stabilization.extract_frames_from_video(self.motion_video_file)
        image = frames[0]
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        corners = cv2.goodFeaturesToTrack(image_gray, mask=None, **feature_params)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title("good_feature_to_track with {} corners".format(100))
        plt.show()

    def test_optical_flow(self):
        frames, fps = stabilization.extract_frames_from_video(self.motion_video_file)
        homographies = []
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.1,
                              minDistance=4,
                              blockSize=4)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(5, 5),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

            curr_frame = frames[i]
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            circled = np.copy(prev_frame)
            for j in good_old:
                x, y = j.ravel()
                cv2.circle(circled, (x, y), 1, (0, 0, 255), -1)
            frame_filename = path.join(self.motion_video_output_dir, "frame_{}_1.jpg".format(i))
            cv2.imwrite(frame_filename, circled)

            circled = np.copy(curr_frame)
            for j in good_new:
                x, y = j.ravel()
                cv2.circle(circled, (x, y), 1, (0, 0, 255), -1)
            frame_filename = path.join(self.motion_video_output_dir, "frame_{}_2.jpg".format(i))
            cv2.imwrite(frame_filename, circled)

            M, _ = cv2.estimateAffine2D(good_new, good_old, method=cv2.RANSAC)
            H = np.append(M, np.array([0, 0, 1]).reshape((1, 3)), axis=0)
            homographies.append(H)

        homographies = np.array(homographies)
        print(homographies)
        stabilization.plot_homographies(homographies, self.motion_video_output_dir)

    def test_compute_homographies_with_motion_video(self):
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=1000,
                              qualityLevel=0.1,
                              minDistance=10,
                              blockSize=10)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(50, 50),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        start = 0
        end = 20
        homographies = stabilization.compute_homographies_from_video(self.motion_video_file,
                                                                     start=start,
                                                                     end=end,
                                                                     feature_params=feature_params,
                                                                     lk_params=lk_params,
                                                                     show_frames=True,
                                                                     output_dir=self.motion_video_output_dir)
        stabilization.plot_homographies(raw_homography=homographies, output_dir=self.motion_video_output_dir,
                                        start=start, end=end)

    def test_get_corners(self):
        shape = (720, 1280, 3)
        print(stabilization.get_corners(shape, ratio=0.5))

    def test_compute_homographies_with_sample_video(self):
        start = 3300
        end = 3400
        homographies = stabilization.compute_homographies_from_video(self.sample_video_file,
                                                                     start=start,
                                                                     end=end,
                                                                     feature_params=self.sample_video_feature_params,
                                                                     lk_params=self.sample_video_lk_params,
                                                                     show_frames=True,
                                                                     output_dir=self.sample_video_output_dir)
        stabilization.plot_homographies(raw_homography=homographies, output_dir=self.sample_video_output_dir,
                                        start=start, end=end)

    def test_compute_smooth_path_with_motion_video(self):
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.1,
                              minDistance=4,
                              blockSize=4)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(5, 5),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        start = 0
        end = None
        homographies = stabilization.compute_homographies_from_video(self.motion_video_file,
                                                                     start=start,
                                                                     end=end,
                                                                     feature_params=feature_params,
                                                                     lk_params=lk_params,
                                                                     show_frames=False,
                                                                     output_dir=self.motion_video_output_dir)
        frames, fps = stabilization.extract_frames_from_video(self.motion_video_file, end=2)
        shape = frames[0].shape

        smoothed = stabilization.compute_smooth_path(shape, homographies, ratio=0.8)
        stabilization.plot_homographies(raw_homography=homographies,
                                        output_dir=self.motion_video_output_dir,
                                        smoothed_homography=smoothed, start=start, end=end)

    def test_compute_smooth_path_with_sample_video(self):
        start = 0
        end = 500
        raw_homography = stabilization.compute_homographies_from_video(self.sample_video_file,
                                                                       start=start,
                                                                       end=end,
                                                                       feature_params=self.sample_video_feature_params,
                                                                       lk_params=self.sample_video_lk_params,
                                                                       show_frames=False,
                                                                       output_dir=self.sample_video_output_dir)
        frames, fps = stabilization.extract_frames_from_video(self.sample_video_file, end=2)
        shape = frames[0].shape
        smoothed = stabilization.compute_smooth_path(shape, raw_homography, ratio=0.8)
        stabilization.plot_homographies(raw_homography=raw_homography,
                                        output_dir=self.sample_video_output_dir,
                                        smoothed_homography=smoothed, start=start, end=end)

    def test_draw_box(self):
        frames, fps = stabilization.extract_frames_from_video(self.sample_video_file, end=2)
        shape = frames[0].shape
        new_image = stabilization.draw_box(frames[0], stabilization.get_corners(shape))
        cv2.imwrite("box.jpg", new_image)

    def test_apply_homography_on_frame(self):
        frames, fps = stabilization.extract_frames_from_video(self.sample_video_file, end=2)
        new_frame = stabilization.apply_homography_on_frame(frames[0], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                                            ratio=0.8)
        print(new_frame.shape)
        cv2.imwrite("cropped.jpg", new_frame)

    def test_warp_frame(self):
        start = 3300
        end = 3400
        ratio = 0.9
        frames, fps = stabilization.extract_frames_from_video(self.sample_video_file, start=start, end=end)
        raw_homography = stabilization.compute_homographies_from_frames(frames)
        shape = frames[0].shape
        smoothed = stabilization.compute_smooth_path(shape, raw_homography, ratio=ratio)
        boxed_frames = np.copy(frames)
        new_frames = stabilization.apply_homography_on_frames(frames, smoothed, ratio=ratio)
        stabilization.create_video_from_frames(new_frames,
                                               path.join(self.sample_video_output_dir,
                                                         "frame_smooth_{}_{}.mp4".format(start, end)),
                                               fps=30)
        boxed_frames = stabilization.draw_smoothed_box_on_frames(boxed_frames, smoothed, ratio=ratio)
        stabilization.create_video_from_frames(boxed_frames,
                                               path.join(self.sample_video_output_dir,
                                                         "frame_box_{}_{}.mp4".format(start, end)),
                                               fps=30)
        stabilization.plot_homographies(raw_homography, self.sample_video_output_dir,
                                        smoothed_homography=smoothed, start=start, end=end)

    def test_cvxpy(self):
        # Create two scalar optimization variables.
        x = cp.Variable()
        y = cp.Variable()

        # Create two constraints.
        constraints = [x + y == 1,
                       x - y >= 1]

        # Form objective.
        obj = cp.Minimize((x - y) ** 2)

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal var", x.value, y.value)

    def test_cvxpy2(self):
        # Problem data.
        A_wall = 100
        A_flr = 10
        alpha = 0.5
        beta = 2
        gamma = 0.5
        delta = 2

        h = cp.Variable(pos=True, name="h")
        w = cp.Variable(pos=True, name="w")
        d = cp.Variable(pos=True, name="d")

        volume = h * w * d
        wall_area = 2 * (h * w + h * d)
        flr_area = w * d
        hw_ratio = h / w
        dw_ratio = d / w
        constraints = [
            wall_area <= A_wall,
            flr_area <= A_flr,
            hw_ratio >= alpha,
            hw_ratio <= beta,
            dw_ratio >= gamma,
            dw_ratio <= delta
        ]
        problem = cp.Problem(cp.Maximize(volume), constraints)
        print(problem)
        assert not problem.is_dcp()
        assert problem.is_dgp()
        problem.solve(gp=True)
        print(problem.value)

    def test_transform(self):
        source = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
        dest = np.float32([[0, 0], [-1000, 0], [-1000, -1000], [0, -1000]])

        points = np.float32([[[50, 50]]])

        homography, _ = cv2.findHomography(source, dest)

        transformed = cv2.perspectiveTransform(points, homography)

        print(transformed)
        # => [[[-500. -500.]]]

        homography_inverse = np.linalg.inv(homography)

        detransformed = cv2.perspectiveTransform(transformed, homography_inverse)

        print(detransformed)
        # => [[[50. 50.]]]

    def test_batch_size(self):
        batch_size = 10
        n = 6
        batch_number = int(n / batch_size)
        for i in np.arange(batch_number + 1):
            start = batch_size * i
            end = min(n, batch_size * (i + 1))
            print("batch{}: frame {} - {}".format(i, start, end))

    def test_foreground_removal(self):
        pass

    def test_save(self):
        frames, fps = stabilization.extract_frames_from_video(self.sample_video_file)
        np.save(path.join(self.sample_video_output_dir, "frames.npy"), frames)


if __name__ == '__main__':
    unittest.main()
