import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
from os import path
import cvxpy as cp


def extract_frames_from_video(video_file, start=None, end=None):
    """
    get frames from video file in range [start, end)
    Parameters
    ----------
    video_file
    start: include start frame
    end: not include end frame

    Returns
    -------
    frames: np.ndarray(), dtype=np.uint8
    fps: int, frames per second
    """
    if start is None:
        start = 0
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("read video {} * {} * {}, {} fps".format(frame_count, height, width, fps))
    frames = []
    current_frame = 0
    while 1:
        ret, frame = cap.read()
        if ret:
            if current_frame >= start:
                frames.append(frame)
            if end is not None and end - 1 <= current_frame:
                break
            current_frame += 1
        else:
            break
    return np.array(frames, dtype=np.uint8), fps


def create_video_from_frames(frames, video_file, fps=30):
    """
    create video with frames
    Parameters
    ----------
    frames: np.ndarray(), dtype=np.uint8
    video_file: string, output video file name
    fps: float, frame rate
    Returns
    -------

    """
    n, r, c, ch = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, fps, (c, r))
    for i in np.arange(n):
        out.write(frames[i])
    out.release()


def compute_homography(prev_frame, curr_frame, feature_params=None, lk_params=None, show_frames=False, output_dir=None,
                       frame_name=None):
    """
    calculate homography from current frame to previous frame
    Parameters
    ----------
    prev_frame: np.ndarray() dtype=np.uint8
    curr_frame: np.ndarray() dtype=np.uint8
    feature_params: dict
    lk_params: dict
    show_frames: bool
    output_dir: str
    frame_name: str

    Returns
    -------
    H: np.ndarray(), dtype=np.float64, shape=(3,3), homography
    """
    # params for ShiTomasi corner detection
    if feature_params is None:
        feature_params = dict(maxCorners=2000,
                              qualityLevel=0.1,
                              minDistance=10,
                              blockSize=10)

    # Parameters for lucas kanade optical flow
    if lk_params is None:
        lk_params = dict(winSize=(50, 50),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # TODO: get background mask
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

    # select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # if need to show good points on frame
    if show_frames:
        circled = np.copy(prev_frame)
        for j in good_old:
            x, y = j.ravel()
            cv2.circle(circled, (x, y), 6, (0, 0, 255), -1)
        frame_filename = path.join(output_dir, "frame_{}_1.jpg".format(frame_name))
        cv2.imwrite(frame_filename, circled)

        circled = np.copy(curr_frame)
        for j in good_new:
            x, y = j.ravel()
            cv2.circle(circled, (x, y), 6, (0, 0, 255), -1)
        frame_filename = path.join(output_dir, "frame_{}_2.jpg".format(frame_name))
        cv2.imwrite(frame_filename, circled)

    if good_new.shape[0] > 0 and good_old.shape[0] > 0:
        M, _ = cv2.estimateAffine2D(good_new, good_old, method=cv2.RANSAC)
    else:
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

    if M is None:
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    H = np.append(M, np.array([0.0, 0.0, 1.0]).reshape((1, 3)), axis=0)
    return H


def compute_homographies_from_frames(frames, feature_params=None, lk_params=None, show_frames=False, output_dir=None,
                                     first_frame_idx=None):
    """
    get array of homographies from frames
    Parameters
    ----------
    frames: np.ndarray(), dtype=np.uint8, shape=(n,r,c,3)
    feature_params: dict
    lk_params: dict
    show_frames: bool
    output_dir: str
    first_frame_idx: int
    Returns
    -------
    homographies: np.ndarray(), dtype=np.float64, shape=(n-1, 3, 3)
    """
    n, r, c, ch = frames.shape
    homographies = np.zeros((n - 1, 3, 3), dtype=np.float64)
    frame_name = None
    for i in np.arange(1, n):
        if show_frames:
            frame_name = str(first_frame_idx + i)
        h = compute_homography(frames[i - 1], frames[i], feature_params, lk_params, show_frames, output_dir, frame_name)
        homographies[i - 1] = h

    return homographies


def compute_homographies_from_video(video_file, start=None, end=None, feature_params=None, lk_params=None,
                                    show_frames=False, output_dir=None):
    """
    (not in use)
    Parameters
    ----------
    video_file
    start: int, inclued start frame
    end: int, not include end frame
    feature_params
    lk_params
    show_frames
    output_dir

    Returns
    -------
    homographies: np.ndarray(), dtype=np.float64
        shape of (end-start, 3, 3)
    """
    if start is None:
        start = 0

    cap = cv2.VideoCapture(video_file)
    homographies = []
    ret, prev_frame = cap.read()
    current_frame_idx = 1
    frame_name = None
    while 1:
        ret, curr_frame = cap.read()
        if ret:
            if current_frame_idx >= start:
                if show_frames:
                    frame_name = str(current_frame_idx)
                h = compute_homography(prev_frame, curr_frame, feature_params, lk_params, show_frames, output_dir,
                                       frame_name)
                homographies.append(h)
            if end is not None and 0 < end - 1 <= current_frame_idx:
                break
            prev_frame = curr_frame
            current_frame_idx += 1
        else:
            break

    homographies = np.array(homographies, dtype=np.float64)
    return homographies


def draw_movement_on_frames(frames, raw_homography):
    n, r, c, ch = frames.shape
    raise NotImplementedError


def draw_movement_on_frame(prev_frame, curr_frame, homography, points):
    raise NotImplementedError


def get_corners(shape, ratio=0.8):
    """
    Return the x, y coordinates for the four corners bounding the input shape
    assume top left is (0,0)
    Parameters
    ----------
    shape: a tuple get from np (r, c, ch)
    ratio

    Returns
    -------
    corners: np.ndarray(), dtype=np.float64, shape=(4, 2) list of 4 elements
    """
    h, w, ch = shape
    # width is x, height is y
    center_x = w / 2.0
    center_y = h / 2.0
    d_x = w * ratio / 2.0
    d_y = h * ratio / 2.0
    top_left = (center_x - d_x, center_y - d_y)
    top_right = (center_x + d_x, center_y - d_y)
    button_left = (center_x - d_x, center_y + d_y)
    button_right = (center_x + d_x, center_y + d_y)
    corners = np.array([top_left, top_right, button_left, button_right], dtype=np.float64)
    return corners


def compute_smooth_path(shape, raw_homography, ratio=0.8, first_Bt=None):
    """
    use linear programming to compute smoothed homography
    Parameters
    ----------
    shape: tuple, a shape tuple get from np, (r, c, ch)
    raw_homography: np.ndarray(), dtype=np.float64, shape=(n, 3, 3), F_t in paper
    ratio: float
    first_Bt: np.ndarray, dtype=np.float64, shape=(3,3)
    Returns
    -------
    smoothed_homography: np.ndarray(), dtype=np.float64, shape=(n, 3, 3), B_t in paper
    """
    print("computing smooth path...")
    n = raw_homography.shape[0]
    # pt form: (dx, dy, a, b, c, d)
    # | a   b   dx  |
    # | c   d   dy  | -> B_t
    # | 0   0   1   |
    weight_1 = 10
    weight_2 = 1
    weight_3 = 100
    affine_weights = np.transpose([1, 1, 100, 100, 100, 100])
    # pt is flattened Bt, each pt has 6 values
    # shape of n * 6
    smoothed_homography = cp.Variable((n, 6))
    # slack variable, e_1 in paper
    e_1 = cp.Variable((n, 6))
    # slack variable, e_2 in paper
    e_2 = cp.Variable((n, 6))
    # slack variable, e_3 in paper
    e_3 = cp.Variable((n, 6))

    # minimize c^T dot e
    objective = cp.Minimize(cp.sum((weight_1 * (e_1 @ affine_weights)) +
                                   (weight_2 * (e_2 @ affine_weights)) +
                                   (weight_3 * (e_3 @ affine_weights)), axis=0))
    constraints = []

    # constraint the first Bt
    if first_Bt is not None:
        constraints.append(smoothed_homography[0, 0] == first_Bt[0, 2])
        constraints.append(smoothed_homography[0, 1] == first_Bt[1, 2])
        constraints.append(smoothed_homography[0, 2] == first_Bt[0, 0])
        constraints.append(smoothed_homography[0, 3] == first_Bt[0, 1])
        constraints.append(smoothed_homography[0, 4] == first_Bt[1, 0])
        constraints.append(smoothed_homography[0, 5] == first_Bt[1, 1])

    # Bt constraint
    # 0.9 <= a, d <= 1.1
    # -0.1 <= b, c <= 0.1
    # -0.05 <= b + c <= 0.05
    # -0.1 <= a - d <= 0.1
    # at all t, Bt should within this constraint
    for t in range(n):
        a = smoothed_homography[t, 2]
        b = smoothed_homography[t, 3]
        c = smoothed_homography[t, 4]
        d = smoothed_homography[t, 5]
        constraints.append(a >= 0.9)
        constraints.append(a <= 1.1)

        constraints.append(d >= 0.9)
        constraints.append(d <= 1.1)

        constraints.append(b >= -0.1)
        constraints.append(b <= 0.1)

        constraints.append(c >= -0.1)
        constraints.append(c <= 0.1)

        temp_1 = b + c
        # constraints.append(temp_1 == 0)
        constraints.append(temp_1 >= -0.05)
        constraints.append(temp_1 <= 0.05)

        temp_2 = a - d
        # constraints.append(temp_2 == 0)
        constraints.append(temp_2 >= -0.1)
        constraints.append(temp_2 <= 0.1)

    # inclusion constraint
    h = shape[0]
    w = shape[1]
    corners = get_corners(shape, ratio)
    for i in np.arange(4):
        corner = corners[i]
        x, y = corner
        for t in range(n):
            # function 8 in paper
            flatted_h = [smoothed_homography[t, 0], smoothed_homography[t, 1], smoothed_homography[t, 2],
                         smoothed_homography[t, 3], smoothed_homography[t, 4], smoothed_homography[t, 5]]
            projected_x = np.array([1, 0, x, y, 0, 0]) @ np.transpose(flatted_h)
            projected_y = np.array([0, 1, 0, 0, x, y]) @ np.transpose(flatted_h)
            constraints.append(projected_x >= 0)
            constraints.append(projected_y >= 0)
            constraints.append(projected_x <= w)
            constraints.append(projected_y <= h)

    # smooth constraint
    # e1, e2, e3 at any t >= 0
    for t in range(n):
        for i in range(6):
            constraints.append(e_1[t, i] >= 0)
            constraints.append(e_2[t, i] >= 0)
            constraints.append(e_3[t, i] >= 0)

    for t in range(n - 3):
        # convert from flattened to matrix in order to do matrix multiplication with F_t and get R_t
        #                           | a   b   dx  |
        # (dx, dy, a, b, c, d)->    | c   d   dy  |
        #                           | 0   0   1   |
        B_t = [[smoothed_homography[t, 2], smoothed_homography[t, 3], smoothed_homography[t, 0]],
               [smoothed_homography[t, 4], smoothed_homography[t, 5], smoothed_homography[t, 1]],
               [0, 0, 1]]
        B_t1 = [[smoothed_homography[t + 1, 2], smoothed_homography[t + 1, 3], smoothed_homography[t + 1, 0]],
                [smoothed_homography[t + 1, 4], smoothed_homography[t + 1, 5], smoothed_homography[t + 1, 1]],
                [0, 0, 1]]
        B_t2 = [[smoothed_homography[t + 2, 2], smoothed_homography[t + 2, 3], smoothed_homography[t + 2, 0]],
                [smoothed_homography[t + 2, 4], smoothed_homography[t + 2, 5], smoothed_homography[t + 2, 1]],
                [0, 0, 1]]
        B_t3 = [[smoothed_homography[t + 3, 2], smoothed_homography[t + 3, 3], smoothed_homography[t + 3, 0]],
                [smoothed_homography[t + 3, 4], smoothed_homography[t + 3, 5], smoothed_homography[t + 3, 1]],
                [0, 0, 1]]

        # R_t = F_t+1 dot B_t+1 - B_t
        residual_t = raw_homography[t + 1] @ B_t1 - B_t
        # R_t+1
        residual_t1 = raw_homography[t + 2] @ B_t2 - B_t1
        # R_t+2
        residual_t2 = raw_homography[t + 3] @ B_t3 - B_t2
        # flatten residuals in order to put into constraints
        # | a   b   dx  |
        # | c   d   dy  |->(dx, dy, a, b, c, d)
        # | 0   0   1   |
        residual_flatten_t = [residual_t[0, 2], residual_t[1, 2], residual_t[0, 0], residual_t[0, 1], residual_t[1, 0],
                              residual_t[1, 1]]
        residual_flatten_t1 = [residual_t1[0, 2], residual_t1[1, 2], residual_t1[0, 0], residual_t1[0, 1],
                               residual_t1[1, 0],
                               residual_t1[1, 1]]
        residual_flatten_t2 = [residual_t2[0, 2], residual_t2[1, 2], residual_t2[0, 0], residual_t2[0, 1],
                               residual_t2[1, 0],
                               residual_t2[1, 1]]

        # at any t
        # -e1 <= R_t(p) <= e1
        # -e2 <= R_t1(p) - R_t(p) <= e2
        # -e3 <= R_t2(p) - 2R_t1(p) + R_t(p) <= e3
        # each residual has 6 values, each e at time t has 6 values
        for i in range(6):
            constraints.append(residual_flatten_t[i] <= e_1[t, i])
            constraints.append(residual_flatten_t[i] >= -e_1[t, i])
            constraints.append((residual_flatten_t1[i] - residual_flatten_t[i]) <= e_2[t, i])
            constraints.append((residual_flatten_t1[i] - residual_flatten_t[i]) >= -e_2[t, i])
            constraints.append(
                (residual_flatten_t2[i] - 2 * residual_flatten_t1[i] + residual_flatten_t[i]) <= e_3[t, i])
            constraints.append(
                (residual_flatten_t2[i] - 2 * residual_flatten_t1[i] + residual_flatten_t[i]) >= -e_3[t, i])

    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    print("status:", problem.status)

    # convert from flattened to matrix
    #                           | a   b   dx  |
    # (dx, dy, a, b, c, d)->    | c   d   dy  |
    #                           | 0   0   1   |
    result = np.zeros((n, 3, 3), dtype=np.float64)
    for i in np.arange(n):
        result[i] = np.array(
            [[smoothed_homography.value[i, 2], smoothed_homography.value[i, 3], smoothed_homography.value[i, 0]],
             [smoothed_homography.value[i, 4], smoothed_homography.value[i, 5], smoothed_homography.value[i, 1]],
             [0, 0, 1]])

    return result


def draw_box(image, corners):
    """
    draw red box on image based on corner points
    Parameters
    ----------
    image: np.ndarray(), dtype=np.uint8
    corners: np.ndarray(), shape=(4,2), [top_left, top_right, button_left, button_right]

    Returns
    -------
    image: np.ndarray(), dtype=np.uint8
    """
    # result = np.copy(image)
    x1, y1 = int(corners[0, 0]), int(corners[0, 1])
    x2, y2 = int(corners[1, 0]), int(corners[1, 1])
    x3, y3 = int(corners[2, 0]), int(corners[2, 1])
    x4, y4 = int(corners[3, 0]), int(corners[3, 1])

    result = np.copy(image)
    cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)  # top
    cv2.line(result, (x3, y3), (x1, y1), (0, 0, 255), 2)  # left
    cv2.line(result, (x2, y2), (x4, y4), (0, 0, 255), 2)  # right
    cv2.line(result, (x3, y3), (x4, y4), (0, 0, 255), 2)  # bottom

    return result


def draw_smoothed_box(frame, homography, ratio=0.9):
    """
    draw red box on frame with homography
    Parameters
    ----------
    frame: np.ndarray(), dtype=np.uint8
    homography: np.ndarray(), dtype=np.float64, shape=(3, 3)
    ratio: float

    Returns
    -------
    frame: np.ndarray(), dtype=np.uint8
    """
    shape = frame.shape
    corners = get_corners(shape, ratio)
    for i in np.arange(4):
        x = corners[i, 0]
        y = corners[i, 1]
        # | a   b   dx  |
        # | c   d   dy  |->(dx, dy, a, b, c, d)
        # | 0   0   1   |
        flatten_h = [homography[0, 2], homography[1, 2], homography[0, 0], homography[0, 1], homography[1, 0],
                     homography[1, 1]]
        projected_x = np.array([1, 0, x, y, 0, 0]) @ np.transpose(flatten_h)
        projected_y = np.array([0, 1, 0, 0, x, y]) @ np.transpose(flatten_h)
        corners[i, 0] = projected_x
        corners[i, 1] = projected_y

    return draw_box(frame, corners)


def draw_smoothed_box_on_frames(frames, smoothed_homography, ratio=0.9):
    """
    draw red box on frames with homography array
    Parameters
    ----------
    frames: np.ndarray(), dtype=np.uint8, shape=(n,r,c,ch)
    smoothed_homography: np.ndarray(), dtype=np.float64, shape=(n-1,3,3)
    ratio: float

    Returns
    -------
    new_frames: np.ndarray(), dtype=np.uint8, shape=(n-1, r, c, ch)
    """
    n, r, c, ch = frames.shape
    new_frames = np.zeros((n, r, c, ch), dtype=np.uint8)
    for i in np.arange(1, n):
        new_frames[i - 1] = draw_smoothed_box(frames[i], smoothed_homography[i - 1], ratio)
    new_frames[n - 1] = draw_smoothed_box(frames[n - 1], smoothed_homography[n - 2], ratio)
    return new_frames


def draw_smoothed_box_on_video(video_file, out_video_file, smoothed_homography, start=None, end=None, ratio=0.8,
                               fps=30):
    """
    (not in use)
    Parameters
    ----------
    video_file
    out_video_file
    smoothed_homography
    start
    end
    ratio
    fps

    Returns
    -------

    """
    if start is None:
        start = 0

    cap = cv2.VideoCapture(video_file)
    ret, curr_frame = cap.read()
    current_frame_idx = 0

    r, c, ch = curr_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_video_file, fourcc, fps, (c, r))

    while 1:
        ret, curr_frame = cap.read()
        if ret:
            if current_frame_idx >= start:
                out.write(draw_smoothed_box(curr_frame, smoothed_homography[current_frame_idx], ratio))
            if end is not None and 0 < end - 1 <= current_frame_idx:
                break
            current_frame_idx += 1
        else:
            break
    out.release()


def apply_homography_on_frames(frames, homography, ratio=0.9):
    """
    apply homography on frame array
    n frames with n-1 homography
    Parameters
    ----------
    frames: np.ndarray(), dtype=np.uint8, shape=(n,r,c,ch)
    homography: np.ndarray(), dtype=np.float64, shape=(n-1,3,3)
    ratio: float
    Returns
    -------
    warped_frames: np.ndarray(), dtype=np.uint8, shape=(n,r,c,ch)
    """
    n, r, c, ch = frames.shape
    warped_r, warped_c = get_warped_shape((r, c, ch), ratio)
    warped_frames = np.zeros((n, warped_r, warped_c, ch), dtype=np.uint8)
    for i in np.arange(1, n):
        warped_frames[i - 1] = apply_homography_on_frame(frames[i], homography[i - 1], ratio)
    warped_frames[n - 1] = apply_homography_on_frame(frames[n - 1], homography[n - 2], ratio)
    return warped_frames


def apply_homography_on_frame(frame, homography, ratio=0.9):
    """
    apply homography on a frame
    Parameters
    ----------
    frame: np.ndarray(), dtype=np.uint8, shape=(r,c,ch)
    homography: np.ndarray(), dtype=np.float64, shape=(3,3)
    ratio: float

    Returns
    -------
    warped_frame: np.ndarray(), dtype=np.uint8, smaller frame
    """
    corners = get_corners(frame.shape, ratio=ratio)
    min_x = int(corners[0, 0])
    min_y = int(corners[0, 1])
    max_x = int(corners[3, 0])
    max_y = int(corners[3, 1])
    warped_frame = cv2.warpPerspective(frame, np.linalg.inv(homography), (frame.shape[1], frame.shape[0]))
    return warped_frame[min_y: max_y, min_x: max_x, :]


def apply_homography_on_video(video_file, out_video_file, homography, start=None, end=None, ratio=0.9, fps=30):
    """
    (not in use)
    Parameters
    ----------
    video_file
    out_video_file
    homography
    start
    end
    ratio
    fps

    Returns
    -------

    """
    if start is None:
        start = 0

    cap = cv2.VideoCapture(video_file)
    ret, curr_frame = cap.read()
    current_frame_idx = 0

    r, c, ch = curr_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_video_file, fourcc, fps, (c, r))

    while 1:
        ret, curr_frame = cap.read()
        if ret:
            if current_frame_idx >= start:
                out.write(apply_homography_on_frame(curr_frame, homography[current_frame_idx], ratio))
            if end is not None and 0 < end - 2 <= current_frame_idx:
                break
            current_frame_idx += 1
        else:
            break
    out.release()


def get_warped_shape(shape, ratio):
    """
    get frame size with warp ratio
    Parameters
    ----------
    shape: tuple, from np.shape, (r, c, ch)
    ratio: float
    Returns
    -------
    (r, c)
    """
    r, c, ch = shape
    corners = get_corners((r, c, ch), ratio=ratio)
    min_x = int(corners[0, 0])
    min_y = int(corners[0, 1])
    max_x = int(corners[3, 0])
    max_y = int(corners[3, 1])
    return max_y - min_y, max_x - min_x


def smoothing_video(video_file, out_video_file, ratio=0.9, batch_size=500):
    """
    main pipeline function
    Parameters
    ----------
    video_file: str
    out_video_file: str
    ratio: float
    batch_size: int

    Returns
    -------

    """
    frames, fps = extract_frames_from_video(video_file)
    n, r, c, ch = frames.shape
    boxed_frames = np.zeros((n, r, c, ch), dtype=np.uint8)
    warped_r, warped_c = get_warped_shape((r, c, ch), ratio)
    new_frames = np.zeros((n, warped_r, warped_c, ch), dtype=np.uint8)
    all_raw_homography = np.zeros((n-1, 3, 3), dtype=np.float64)
    all_smoothed = np.zeros((n-1, 3, 3), dtype=np.float64)

    batch_number = int(n / batch_size)
    print("split into {} batches".format(batch_number))
    first_Bt = None
    for i in np.arange(batch_number + 1):
        start = batch_size * i
        end = min(n, batch_size * (i + 1) + 4)
        print("running batch {}: frame {} - {}".format(i, start, end))
        raw_homography = compute_homographies_from_frames(frames[start: end])
        all_raw_homography[start:end-1] = raw_homography
        smoothed = compute_smooth_path((r, c, ch), raw_homography, ratio=ratio, first_Bt=first_Bt)
        all_smoothed[start:end-1] = smoothed
        first_Bt = smoothed[-4]
        boxed_frames[start:end] = draw_smoothed_box_on_frames(frames[start:end], smoothed, ratio=ratio)
        new_frames[start:end] = apply_homography_on_frames(frames[start:end], smoothed, ratio=ratio)
        # create_video_from_frames(new_frames[start:end],
        #                          path.join(out_video_file, "frame_smooth_{}_{}.mp4".format(start, end)),
        #                          fps=fps)
        # create_video_from_frames(boxed_frames[start:end],
        #                          path.join(out_video_file, "frame_box_{}_{}.mp4".format(start, end)),
        #                          fps=fps)
        # plot_homographies(raw_homography, out_video_file, prefix="v2", smoothed_homography=smoothed, start=start, end=end)
    print("done smoothing")
    create_video_from_frames(boxed_frames, path.join(out_video_file, "result_with_box.mp4"),
                             fps=fps)
    create_video_from_frames(new_frames, path.join(out_video_file, "result_smoothed.mp4"),
                             fps=fps)
    plot_homographies(all_raw_homography, out_video_file, smoothed_homography=all_smoothed)


def plot_homographies(raw_homography, output_dir, prefix="", smoothed_homography=None, start=None, end=None):
    """
    plot homography array
    Parameters
    ----------
    prefix
    raw_homography: np.ndarray(), dtype=np.float64, shape of (n, 3, 3)
    output_dir: string, output directory string
    smoothed_homography: np.ndarray(), dtype=np.float64, shape of (n, 3, 3)
    start: int, start frame index
    end: int, end frame index

    Returns
    -------

    """
    if start is None:
        start = 0
    if end is None:
        end = start + raw_homography.shape[0]
    else:
        end = min(end, start + raw_homography.shape[0])
    frame_idx = np.arange(start, end)
    n = end - start
    sub_prefix = prefix + " frame {} - {} ".format(start, end)
    raw_x_path = np.zeros(n)
    raw_y_path = np.zeros(n)
    raw_dx = np.zeros(n)
    raw_dy = np.zeros(n)
    smoothed_x_path = np.zeros(n)
    smoothed_y_path = np.zeros(n)
    smoothed_dx = np.zeros(n)
    smoothed_dy = np.zeros(n)
    pt = np.array([1, 1, 1], dtype=np.float64)

    for i in np.arange(n):
        pt = np.matmul(raw_homography[i], pt)
        raw_x_path[i] = pt[0]
        raw_y_path[i] = pt[1]
        raw_dx[i] = raw_homography[i, 0, 2]
        raw_dy[i] = raw_homography[i, 1, 2]
        if smoothed_homography is not None:
            smooth_pt = np.matmul(smoothed_homography[i], pt)
            smoothed_x_path[i] = smooth_pt[0]
            smoothed_y_path[i] = smooth_pt[1]
            smoothed_dx[i] = smoothed_homography[i, 0, 2]
            smoothed_dy[i] = smoothed_homography[i, 1, 2]

    plt.plot(frame_idx, raw_dx, label="raw dx")
    if smoothed_homography is not None:
        plt.plot(frame_idx, smoothed_dx, label="smoothed dx")
    plt.title("dx")
    plt.xlabel("frame")
    plt.ylabel("dx")
    plot_filename = path.join(output_dir, sub_prefix + "motion_dx.png")
    plt.legend()
    plt.savefig(plot_filename)
    # plt.show()
    plt.clf()

    plt.plot(frame_idx, raw_dy, label="raw dy")
    if smoothed_homography is not None:
        plt.plot(frame_idx, smoothed_dy, label="smoothed dy")
    plt.title("dy")
    plt.xlabel("frame")
    plt.ylabel("dy")
    plot_filename = path.join(output_dir, sub_prefix + "motion_dy.png")
    plt.legend()
    plt.savefig(plot_filename)
    # plt.show()
    plt.clf()

    plt.plot(frame_idx, raw_x_path, label="raw x path")
    if smoothed_homography is not None:
        plt.plot(frame_idx, smoothed_x_path, label="smoothed x path")
    plt.title("x transform")
    plt.xlabel("frame")
    plt.ylabel("x transform")
    plot_filename = path.join(output_dir, sub_prefix + "motion_x.png")
    plt.legend()
    plt.savefig(plot_filename)
    # plt.show()
    plt.clf()

    plt.plot(frame_idx, raw_y_path, label="raw y path")
    if smoothed_homography is not None:
        plt.plot(frame_idx, smoothed_y_path, label="smoothed y path")
    plt.title("y transform")
    plt.xlabel("frame")
    plt.ylabel("y transform")
    plot_filename = path.join(output_dir, sub_prefix + "motion_y.png")
    plt.legend()
    plt.savefig(plot_filename)
    # plt.show()
    plt.clf()
