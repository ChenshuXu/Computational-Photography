# CS6475 - Spring 2021

import numpy as np
import scipy as sp
import cv2
import scipy.signal  # option for a 2D convolution library
from matplotlib import pyplot as plt  # for optional plots

import copy

""" Project 1: Seam Carving

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

References
----------
See the following papers, available in Canvas under Files:

(1) "Seam Carving for Content-Aware Image Resizing"
    Avidan and Shamir, 2007
    
(2) "Improved Seam Carving for Video Retargeting"
    Rubinstein, Shamir and Avidan, 2008
    
FORBIDDEN:
    1. OpenCV functions SeamFinder, GraphCut, and CostFunction are
    forbidden, along with any similar functions in the class environment.
    2. Metrics functions of error or similarity. These need to be coded from their
    mathematical equations.

GENERAL RULES:
    1. ALL CODE USED IN THIS ASSIGNMENT to generate images, red-seam images,
    differential images, and comparison metrics must be included in this file.
    2. YOU MAY ADD FUNCTIONS to this file, however it is your responsibility
    to ensure that the autograder accepts your submission.
    3. DO NOT CHANGE the format of this file. You may NOT change existing function
    signatures, including named parameters with defaults.
    4. YOU MAY NOT USE any library function that essentially completes
    seam carving or metric calculations for you. If you have questions on this,
    ask on Piazza.
    5. DO NOT IMPORT any additional libraries other than the ones included in the
    original Course Setup CS6475 environment.
    You should be able to complete the assignment with the given libraries.
    6. DO NOT INCLUDE code that saves, shows, prints, displays, or writes the
    image passed in, or your results. If you have code in the functions that 
    does any of these operations, comment it out before autograder runs.
    7. YOU ARE RESPONSIBLE for ensuring that your code executes properly.
    This file has only been tested in the course environment. any changes you make
    outside the areas annotated for student code must not impact the autograder
    system or your performance.
    
FUNCTIONS:
    IMAGE GENERATION:
        beach_backward_removal
        dolphin_backward_insert + with redSeams=True
        dolphin_backward_5050
        bench_backward_removal + with redSeams=True
        bench_forward_removal + with redSeams=True
        car_backward_insert
        car_forward_insert
    COMPARISON METRICS:
        difference_image
        numerical_comparison
"""


def calc_energy_map(image):
    """
    Create energy map from image
    Parameters
    ----------
    image: numpy.ndarray (dtype=uint8)
        Three-channel image of shape (r,c,ch)
    Returns
    -------
    energy_map: numpy.ndarray (dtype=np.float64)
        Energy map of shape (r,c)
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT).astype(np.float64)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT).astype(np.float64)
    grad_sum = np.square(grad_x) + np.square(grad_y)
    grad_sum = np.sqrt(grad_sum)
    energy_map = grad_sum.sum(axis=2)
    return energy_map


def calc_energy_map_2(image):
    r, c, ch = image.shape
    grad = np.zeros_like(image, dtype=np.float64)
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT).astype(np.float64)
    for i in np.arange(r):
        for j in np.arange(c):
            # pxl = padded_image[i+1, j+1]
            left = padded_image[i + 1, j]
            right = padded_image[i + 1, j + 2]
            up = padded_image[i, j + 1]
            down = padded_image[i + 2, j + 1]
            d_x = left - right
            d_y = up - down
            sq = np.square(d_x) + np.square(d_y)
            grad[i, j] = sq
    return np.sum(grad, axis=2)


def calc_energy_map_3(image):
    r, c, ch = image.shape
    channels = np.moveaxis(image, 2, 0)
    grad_x = np.zeros((r, c), dtype=np.float64)
    grad_y = np.zeros((r, c), dtype=np.float64)

    for ch in channels:
        dy, dx = np.gradient(ch.astype(np.float64))
        grad_x += np.absolute(dx)
        grad_y += np.absolute(dy)

    grad_sum = grad_x + grad_y
    max_energy = np.max(grad_sum)
    return grad_sum / max_energy


def calc_energy_map_4(image):
    channel_b, channel_g, channel_r = np.moveaxis(image, 2, 0)
    b_energy = np.absolute(cv2.Scharr(channel_b, -1, 1, 0)) + np.absolute(cv2.Scharr(channel_b, -1, 0, 1))
    g_energy = np.absolute(cv2.Scharr(channel_g, -1, 1, 0)) + np.absolute(cv2.Scharr(channel_g, -1, 0, 1))
    r_energy = np.absolute(cv2.Scharr(channel_r, -1, 1, 0)) + np.absolute(cv2.Scharr(channel_r, -1, 0, 1))
    energy_map = b_energy + g_energy + r_energy
    return energy_map.astype(np.float64)


def calc_forward_grads(image):
    """
    calculate up left, left right, up right grads
    Parameters
    ----------
    image: numpy.ndarray (dtype=uint8)
        Three-channel image of shape (r,c,ch)

    Returns
    -------

    """
    r, c, ch = image.shape
    grad_up_left = np.zeros_like(image, dtype=np.float64)
    grad_up = np.zeros_like(image, dtype=np.float64)
    grad_up_right = np.zeros_like(image, dtype=np.float64)
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT).astype(np.float64)
    for i in np.arange(r):
        for j in np.arange(c):
            pixel = padded_image[i+1, j+1]
            left = padded_image[i + 1, j]
            up_left = padded_image[i, j]
            right = padded_image[i + 1, j + 2]
            up_right = padded_image[i + 2, j + 2]
            up = padded_image[i, j + 1]
            d_up_left = up_left - left
            d_up = left - right
            d_up_right = up_right - right
            grad_up_left[i, j] = np.absolute(d_up_left)
            grad_up[i, j] = np.absolute(d_up)
            grad_up_right[i, j] = np.absolute(d_up_right)
    return np.sum(grad_up_left, axis=2), np.sum(grad_up, axis=2), np.sum(grad_up_right, axis=2)


def calc_neighbor_grads(image, kernel):
    """
    calculate neighbor grad map with kernel
    Parameters
    ----------
    image: numpy.ndarray (dtype=uint8)
        Three-channel image of shape (r,c,ch)
    kernel

    Returns
    -------

    """
    channel_b, channel_g, channel_r = np.moveaxis(image, 2, 0)
    grad_b = np.absolute(cv2.filter2D(channel_b, -1, kernel))
    grad_g = np.absolute(cv2.filter2D(channel_g, -1, kernel))
    grad_r = np.absolute(cv2.filter2D(channel_r, -1, kernel))
    return grad_b + grad_g + grad_r


def cumulative_map_forward(energy_map, grad_up_left, grad_up, grad_up_right):
    """
    find cumulative map, use forward energy
    Parameters
    ----------
    grad_y_right
    grad_y_left
    grad_x
    energy_map: numpy.ndarray (dtype=np.float64)
        shape of (r, c)

    Returns
    -------
    cumulative_map: numpy.ndarray (dtype=np.float64)
        shape of (r, c)
    backtrack: np.ndarray (dtype=np.int64)
    """
    cumulative_map = np.copy(energy_map)
    backtrack = np.zeros_like(energy_map, dtype=np.int64)
    r, c = energy_map.shape

    for i in np.arange(1, r):
        for j in np.arange(0, c):
            # i = row idx, j = col idx
            if j == 0:
                e_r = cumulative_map[i - 1, j + 1] + grad_up[i, j] + grad_up_right[i, j]
                e_u = cumulative_map[i - 1, j] + grad_up[i, j]
                idx_in_row = np.argmin([e_u, e_r])
                min_energy = min(e_r, e_u)
                backtrack[i, j] = idx_in_row
            elif j == c - 1:
                e_l = cumulative_map[i - 1, j - 1] + grad_up[i, j] + grad_up_left[i, j]
                e_u = cumulative_map[i - 1, j] + grad_up[i, j]
                idx_in_row = j + np.argmin([e_l, e_u]) - 1
                min_energy = min(e_l, e_u)
                backtrack[i, j] = idx_in_row
            else:
                e_l = cumulative_map[i - 1, j - 1] + grad_up[i, j] + grad_up_left[i, j]
                e_r = cumulative_map[i - 1, j + 1] + grad_up[i, j] + grad_up_right[i, j]
                e_u = cumulative_map[i - 1, j] + grad_up[i, j]
                idx_in_row = j + np.argmin([e_l, e_u, e_r]) - 1
                min_energy = min(e_l, e_r, e_u)
                backtrack[i, j] = idx_in_row
            cumulative_map[i, j] = energy_map[i, j] + min_energy

    return cumulative_map, backtrack


def cumulative_map_backward(energy_map):
    """
    find n minimum path from energy map
    Parameters
    ----------
    energy_map: numpy.ndarray (dtype=np.float64)
        shape of (r, c)
    n:  int
    Returns
    -------
    path: numpy.ndarray (dtype=np.int)
        shape of (n, r)
    cumulative_map: numpy.ndarray (dtype=np.float64)
        shape of (r, c)
    """
    cumulative_map = np.copy(energy_map)
    backtrack = np.zeros_like(energy_map, dtype=np.int64)
    r, c = energy_map.shape
    for i in np.arange(1, r):
        for j in np.arange(0, c):
            if j == 0:
                # handle the left edge of the image
                # find index of the minimum value in upper i
                # middle: j, right: j+1
                idx = np.argmin(cumulative_map[i - 1, j:j + 2])
                # if choose middle, idx=0, index in i=j
                # if choose right, idx=1, index in i=j+1
                idx_in_row = j + idx
                # record the trace
                backtrack[i, j] = idx_in_row
                min_energy = cumulative_map[i - 1, idx_in_row]
            elif j == c - 1:
                # handle the right edge of the image
                # left: j-1, middle: j
                idx = np.argmin(cumulative_map[i - 1, j - 1:j + 1])
                # if choose left, idx=0, index in i=j-1
                # if choose middle, idx=1, index in i=j
                idx_in_row = j + idx - 1
                backtrack[i, j] = idx_in_row
                min_energy = cumulative_map[i - 1, idx_in_row]
            else:
                # handle other cases
                # left: j-1, middle: j, right: j+1
                idx = np.argmin(cumulative_map[i - 1, j - 1:j + 2])
                # if choose left, idx=0, index in i=j-1
                # if choose middle, idx=1, index in i=j
                # if choose right, idx=2, index in i=j+1
                idx_in_row = j + idx - 1
                backtrack[i, j] = idx_in_row
                min_energy = cumulative_map[i - 1, idx_in_row]

            cumulative_map[i, j] = energy_map[i, j] + min_energy

    return cumulative_map, backtrack


def get_path(cumulative_map, backtrack):
    """
    get seams path from cumulative map and backtrack map
    Parameters
    ----------
    cumulative_map: np.ndarray(dtype=np.float64)
        shape of (r, c)
    backtrack: np.ndarray(dtype=np.int)
        shape of (r, c)
    Returns
    -------
    path: np.ndarray(dtype=np.int)
        shape of (r, )
    """
    r, c = cumulative_map.shape
    path = np.zeros((r, ), dtype=np.int)
    # in the last i, find smallest value, it is the minimum energy line starts
    last_row = np.argsort(cumulative_map[-1, :])
    j = last_row[0]  # index of smallest value
    # from button to top
    for i in reversed(range(r)):
        path[i] = j
        j = backtrack[i, j]
    np.flip(path)
    return path


def get_path_mask(shape, path):
    """
    use minimum energy path to generate mask
    Parameters
    ----------
    shape: tuple (r, c)
    path: numpy.ndarray (dtype=np.int)
        n paths from top to button, shape of (r, )
    Returns
    -------
    mask: numpy.ndarray (dtype=np.bool)
        Mask with shape like energy_map, path=True, other=False
    """
    r, c = shape
    mask = np.zeros(shape, dtype=np.bool)
    for i in np.arange(r):
        idx = path[i]
        mask[i, idx] = True
    return mask


def remove_1(image, path):
    """
    remove one column base on index in path
    Parameters
    ----------
    image: np.ndarray (dtype=np.float64)
        shape of (r, c, ch)
    path: np.ndarray (dtype=np.int)
        shape of (r, )

    Returns
    -------
    new_image: np.ndarray (dtype=np.float64)
        shape of (r, c-1, ch)
    """
    r, c, ch = image.shape
    mask = get_path_mask((r, c), path)
    mask = np.stack([mask, mask, mask], axis=2)
    new_image = image[~mask].reshape((r, c - 1, ch))
    return new_image


def seam_remove_1_backward(image):
    """
    remove one column
    Parameters
    ----------
    image

    Returns
    -------
    new_image: new image after delete one column
    path: list of index of this deletion
    """
    r, c, ch = image.shape
    energy_map = calc_energy_map(image)
    cumulative_map, backtrack = cumulative_map_backward(energy_map)
    path = get_path(cumulative_map, backtrack)
    new_image = remove_1(image, path)
    return new_image, path


def seam_remove_1_forward(image):
    """
    remove one column
    Parameters
    ----------
    image: np.ndarray (dtype=np.float64)

    Returns
    -------
    new_image: np.ndarray (dtype=np.float64)
        new image after delete one column
    path: np.ndarray (dtype=np.int)
        list of index of this deletion, shape of (r, )
    """
    energy_map = calc_energy_map(image)
    grad_up_left, grad_up, grad_up_right = calc_forward_grads(image)
    cumulative_map, backtrack = cumulative_map_forward(energy_map, grad_up_left, grad_up, grad_up_right)
    path = get_path(cumulative_map, backtrack)
    new_image = remove_1(image, path)
    return new_image, path


def insert_1(image, path, red_seam=False):
    """
    insert one column from path
    Parameters
    ----------
    image: np.ndarray (dtype=np.float64)
        shape of (r, c, ch)
    path: np.ndarray (dtype=np.int)
        shape of (r, )
    Returns
    -------
    inserted_image: np.ndarray (dtype=np.float64)
    """
    r, c, ch = image.shape
    inserted_image = np.zeros((r, c + 1, ch), dtype=np.float64)
    for i in np.arange(r):
        j = path[i]
        if red_seam:
            new_pixel = [0, 0, 255]
        else:
            if j == 0:
                new_pixel = image[i, j]
            elif j == c - 1:
                new_pixel = image[i, j]
            else:
                new_pixel = (image[i, j] + image[i, j + 1]) / 2
        inserted_image[i, :j + 1] = image[i, :j + 1]
        inserted_image[i, j + 1] = new_pixel
        inserted_image[i, j + 2:] = image[i, j + 1:]
    return inserted_image


def insert_red_to_rem(image, path_record):
    """
    insert red seams to image from path records
    Parameters
    ----------
    image: np.ndarray (dtype=np.float64)
        shape of (r, c, ch)
    path_record: list
        shape of (number of records, r)

    Returns
    -------
    image: np.ndarray (dtype=np.float64)
        shape of (r, c+n, ch)
    """
    r, c, ch = image.shape
    channel_b, channel_g, channel_r = np.moveaxis(image, 2, 0)
    backward_path_record = np.flip(np.array(path_record), axis=0)

    for path in backward_path_record:
        new_path = path + c * np.arange(r)
        channel_b = np.insert(channel_b.flatten(), new_path, 0).reshape(r, c + 1)
        channel_g = np.insert(channel_g.flatten(), new_path, 0).reshape(r, c + 1)
        channel_r = np.insert(channel_r.flatten(), new_path, 255).reshape(r, c + 1)
        c += 1
    image = np.dstack([channel_b, channel_g, channel_r])
    return image


def insert_red_to_original(image, path_record):
    """
    insert red seam line
    Parameters
    ----------
    image
    path_record

    Returns
    -------

    """
    r, c, ch = image.shape
    channel_b, channel_g, channel_r = np.moveaxis(image, 2, 0)
    path_record_copy = path_record.copy()

    for _ in range(len(path_record_copy)):
        path = path_record_copy.pop(0)
        new_path = path + c * np.arange(r)
        channel_b = np.insert(channel_b.flatten(), new_path, 0).reshape(r, c + 1)
        channel_g = np.insert(channel_g.flatten(), new_path, 0).reshape(r, c + 1)
        channel_r = np.insert(channel_r.flatten(), new_path, 255).reshape(r, c + 1)
        c += 1
        path_record_copy = update_path_record_2(path_record_copy, path)

    image = np.dstack([channel_b, channel_g, channel_r])
    return image


def update_path_record_2(path_record, current_path):
    """

    Parameters
    ----------
    path_record: list of np.ndarray (dtype=np.int)
        shape of (n, r)
    current_path: np.ndarray  (dtype=np.int)
        shape of (r, )

    Returns
    -------

    """
    new_path_record = []
    for path in path_record:
        path[np.where(path >= current_path)] += 2
        new_path_record.append(path)
    return new_path_record


def update_path_record(path_record, current_path):
    """

    Parameters
    ----------
    path_record: list of np.ndarray (dtype=np.int)
        shape of (n, r)
    current_path: np.ndarray (dtype=np.int)
        shape of (r, )

    Returns
    -------
    new_path_record: list of np.ndarray (dtype=np.int)
        shape of (n, r)
    """
    new_path_record = []
    for path in path_record:
        new_path = []
        for r in range(len(path)):
            if path[r] >= current_path[r]:
                new_path.append(path[r] + 2)
            else:
                new_path.append(path[r])
        new_path_record.append(np.array(new_path))
    return new_path_record


def seams_removal_back(image, num_pixel, red_seams=False):
    """

    Parameters
    ----------
    image: np.ndarray (dtype=np.uint8)
        shape of (r, c, ch)
    num_pixel: int
    red_seams: bool

    Returns
    -------
    removed_image: np.ndarray (dtype=np.float64)
        image after removal
    inserted_image_red: np.ndarray (dtype=np.float64)
        if red_seams=True, image after removal then insert with red
    """
    removed_image = np.copy(image.astype(np.float64))
    inserted_image_red = np.copy(image.astype(np.float64))
    path_record = []
    for i in range(num_pixel):
        removed_image, path = seam_remove_1_backward(removed_image)
        path_record.append(path)
        # print("remove column" + str(i))

    if red_seams:
        inserted_image_red = insert_red_to_rem(removed_image, path_record)
    return removed_image.astype(np.uint8), inserted_image_red.astype(np.uint8)


def seams_removal_forward(image, num_pixel, red_seams=False):
    """

    Parameters
    ----------
    image: np.ndarray (dtype=np.uint8)
        shape of (r, c, ch)
    num_pixel: int
    red_seams: bool

    Returns
    -------
    removed_image: np.ndarray (dtype=np.float64)
        image after removal
    inserted_image_red: np.ndarray (dtype=np.float64)
        if red_seams=True, image after removal then insert with red
    """
    removed_image = np.copy(image.astype(np.float64))
    inserted_image_red = np.copy(image.astype(np.float64))
    path_record = []
    for i in range(num_pixel):
        removed_image, path = seam_remove_1_forward(removed_image)
        path_record.append(path)
        # print("remove column" + str(i))

    if red_seams:
        inserted_image_red = insert_red_to_rem(removed_image, path_record)
    return removed_image.astype(np.uint8), inserted_image_red.astype(np.uint8)


def seams_insertion_back(image, num_pixel, red_seams=False):
    """
    We first perform seam removal for n seams on a duplicated input image and record all the coordinates in the same order when removing.
    Then, we insert new seams to original input image in the same order at the recorded coordinates location.
    The inserted artificial pixel values are derived from an average of left and right neighbors.
    Parameters
    ----------
    image
    num_pixel
    red_seams

    Returns
    -------

    """
    removed_image = np.copy(image.astype(np.float64))
    inserted_image = np.copy(image.astype(np.float64))
    inserted_image_red = np.copy(image.astype(np.float64))  # image with red line
    path_record = []
    for i in range(num_pixel):
        removed_image, path = seam_remove_1_backward(removed_image)
        path_record.append(path)

    for i in range(len(path_record)):
        path = path_record[i]
        inserted_image = insert_1(inserted_image, path)
        if red_seams:
            inserted_image_red = insert_1(inserted_image_red, path, red_seam=True)
        path_record = update_path_record(path_record, path)

    return inserted_image.astype(np.uint8), inserted_image_red.astype(np.uint8)


def seams_insertion_forward(image, num_pixel, red_seams=False):
    removed_image = np.copy(image.astype(np.float64))
    inserted_image = np.copy(image.astype(np.float64))
    inserted_image_red = np.copy(image.astype(np.float64))  # image with red line
    path_record = []
    for i in range(num_pixel):
        removed_image, path = seam_remove_1_forward(removed_image)
        path_record.append(path)

    for _ in range(len(path_record)):
        path = path_record.pop(0)
        inserted_image = insert_1(inserted_image, path)
        if red_seams:
            inserted_image_red = insert_1(inserted_image_red, path, red_seam=True)
        path_record = update_path_record(path_record, path)

    return inserted_image.astype(np.uint8), inserted_image_red.astype(np.uint8)


# -------------------------------------------------------------------
""" IMAGE GENERATION
    Parameters and Returns are as follows for all of the removal/insert 
    functions:

    Parameters
    ----------
    image : numpy.ndarray (dtype=uint8)
        Three-channel image of shape (r,c,ch)
    pctSeams : float
        Decimal value in range between(0. - 1.); percent of vertical seams to be
        inserted or removed.
    redSeams : boolean
        Boolean variable; True = this is a red seams image, False = no red seams
        
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        An image of shape (r, c_new, ch) where c_new = new number of columns.
        Make sure you deal with any needed normalization or clipping, so that 
        your image array is complete on return.
"""


def beach_backward_removal(image, pctSeams=0.50, redSeams=False):
    """ Use the backward method of seam carving from the 2007 paper to remove
    50% of the vertical seams in the provided image. Do NOT hard-code the
    percent of seams to be removed.
    """
    # WRITE YOUR CODE HERE.
    r, c, ch = image.shape
    num_pixel = int(c * pctSeams)
    new_image, inserted_image = seams_removal_back(image, num_pixel, redSeams)
    if redSeams:
        return inserted_image
    else:
        return new_image
    raise NotImplementedError


def dolphin_backward_insert(image, pctSeams=0.50, redSeams=False):
    """ Fig 8c, 8d from 2007 paper. Use the backward method of seam carving to insert
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    
    This function is called twice:  Fig 8c with red seams
                                    Fig 8d without red seams
    """
    # WRITE YOUR CODE HERE.
    r, c, ch = image.shape
    num_pixel = int(c * pctSeams)
    new_image, inserted_image = seams_insertion_back(image, num_pixel, redSeams)
    if redSeams:
        return inserted_image
    else:
        return new_image
    raise NotImplementedError


def dolphin_backward_5050(image, pctSeams=0.50, redSeams=False):
    """ Fig 8f from 2007 paper. Use the backward method of seam carving to insert
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    
    *****************************************************************
    IMPORTANT NOTE: this function is passed the image array from the 
    dolphin_backward_insert function in main.py
    *****************************************************************
    
    """
    # WRITE YOUR CODE HERE.
    r, c, ch = image.shape
    original_c = c / (1+pctSeams)
    num_pixel = int(original_c * pctSeams)
    new_image, inserted_image = seams_insertion_back(image, num_pixel, redSeams)

    if redSeams:
        return inserted_image
    else:
        return new_image

    raise NotImplementedError


def bench_backward_removal(image, pctSeams=0.50, redSeams=False):
    """ Fig 8 from 2008 paper. Use the backward method of seam carving to remove
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    
    This function is called twice:  Fig 8 backward with red seams
                                    Fig 8 backward without red seams
    """
    # WRITE YOUR CODE HERE.
    r, c, ch = image.shape
    num_pixel = int(c * pctSeams)
    new_image, inserted_image = seams_removal_back(image, num_pixel, redSeams)
    if redSeams:
        return inserted_image
    else:
        return new_image
    raise NotImplementedError


def bench_forward_removal(image, pctSeams=0.50, redSeams=False):
    """ Fig 8 from 2008 paper. Use the forward method of seam carving to remove
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    
    This function is called twice:  Fig 8 forward with red seams
                                    Fig 8 forward without red seams
  """
    # WRITE YOUR CODE HERE.
    r, c, ch = image.shape
    num_pixel = int(c * pctSeams)
    new_image, inserted_image = seams_removal_forward(image, num_pixel, redSeams)
    if redSeams:
        return inserted_image
    else:
        return new_image
    raise NotImplementedError


def car_backward_insert(image, pctSeams=0.50, redSeams=False):
    """ Fig 9 from 2008 paper. Use the backward method of seam carving to insert
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    """
    # WRITE YOUR CODE HERE.
    r, c, ch = image.shape
    num_pixel = int(c * pctSeams)
    new_image, inserted_image = seams_insertion_back(image, num_pixel, redSeams)
    if redSeams:
        return inserted_image
    else:
        return new_image
    raise NotImplementedError


def car_forward_insert(image, pctSeams=0.50, redSeams=False):
    """ Fig 9 from 2008 paper. Use the backward method of seam carving to insert
    vertical seams in the image. Do NOT hard-code the percent of seams to be removed.
    """
    # WRITE YOUR CODE HERE.
    r, c, ch = image.shape
    num_pixel = int(c * pctSeams)
    new_image, inserted_image = seams_insertion_forward(image, num_pixel, redSeams)
    if redSeams:
        return inserted_image
    else:
        return new_image
    raise NotImplementedError


# __________________________________________________________________
""" COMPARISON METRICS 
    There are two functions here, one for visual comparison support and one 
    for a quantitative metric. The """


def difference_image(result_image, comparison_image):
    """ Take two images and produce a difference image that best visually
    indicates where the two images differ in pixel values.
    
    Parameters
    ----------
    result_image, comparison_image : numpy.ndarray (dtype=uint8)
        two BGR images of the same shape (r,c,ch) to be compared
    
    Returns
    -------
    numpy.ndarray (dtype=uint8)
        An image of shape (r, c, ch) representing the difference between two
        images. Make sure you deal with any needed normalization or clipping,
        so that your image array is complete on return.
    """
    # WRITE YOUR CODE HERE.
    difference = np.absolute(result_image.astype(np.float64) - comparison_image.astype(np.float64))
    return difference.astype(np.uint8)
    raise NotImplementedError


def numerical_comparison(result_image, comparison_image):
    """ Take two images and produce one or two single-value metrics that
    numerically best indicate(s) how different or similar two images are.
    Only one metric is required, you may submit two, but no more.
    
    If your metric produces a result indicating a number of pixels,
    formulate it as a percentage of the total pixels in the image.

    ******************************************************************
    NOTE: You may not use functions that perform the whole function for you.
    Research methods, find an algorithm (equation) and implement it. You may
    use numpy array functions such as abs, sqrt, min, max, dot, .T and others
    that perform a single operation for you.
    ******************************************************************

    Parameters
    ----------
    result_image, comparison_image : numpy.ndarray (dtype=uint8)
        two BGR images of the same shape (r,c,ch) to be compared

    Returns
    -------
    value(s) : float   NOTE: you may present one or two metrics only.
        One or two single_value metric comparisons
        Return a tuple of values if you are using two metrics.
    """
    # WRITE YOUR CODE HERE.
    r, c, ch = result_image.shape
    difference = np.sum(np.absolute(result_image.astype(np.float64) - comparison_image.astype(np.float64)), axis=2)
    total_pixels = r * c
    same_pixel_count = 0
    for i in np.arange(r):
        for j in np.arange(c):
            pixel_diff = difference[i, j]
            if pixel_diff <= 20:
                same_pixel_count += 1
    result_image_energy = np.sum(calc_energy_map(result_image).flatten())
    comparison_image_energy = np.sum(calc_energy_map(comparison_image).flatten())
    return same_pixel_count/total_pixels, result_image_energy/comparison_image_energy
    raise NotImplementedError


if __name__ == "__main__":
    """ You may use this area for code that allows you to test your functions.
    This section will not be graded, and is optional. Comment out this section when you
    test on the autograder to avoid the chance of wasting time and attempts.
    """
    # WRITE YOUR CODE HERE
