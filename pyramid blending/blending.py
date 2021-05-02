""" Pyramid Blending

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

References
----------
See the following papers, available on the course website under Reading Materials:

(1) "The Laplacian Pyramid as a Compact Image Code"
        Burt and Adelson, 1983

(2) "A Multiresolution Spline with Application to Image Mosaics"
        Burt and Adelson, 1983

Notes
-----
    You may NOT use cv2.pyrUp or cv2.pyrDown anywhere in this assignment.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but these functions should NOT save the image to disk.

    2. DO NOT import any other libraries aside from those that we provide.
    You should be able to complete the assignment with the given libraries
    (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the course environment for this term.
    You are responsible for ensuring that your code executes properly in the
    course environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import scipy as sp
import scipy.signal  # one option for a 2D convolution library
import cv2


def generatingKernel(a):
    """Return a 5x5 filter based on a 5-element input vector and a generating parameter 
    (i.e., makes a square "5-tap" filter.) 

    Parameters 
    ----------
    a : float
        The kernel generating parameter in the range [0, 1] used to generate a
        5-tap filter kernel. The 5-tap kernel is 1D with 5 elements.

    Returns
    -------
    output : numpy.ndarray
        A 5x5 array generated by taking the outer product (kernel x kernel)
    """
    # DO NOT CHANGE THE CODE IN THIS FUNCTION

    # generate the 5-tap kernel
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    # use outer product of 2 vectors to generate the square filter
    return np.outer(kernel, kernel)


def convolutionManual(image, filter):
    """
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxW) and type np.uint8
        You image should be a 2D
    filter : numpy.ndarray
        A 2D numpy array of variable dinensions, with type np.float.
        The filter (also called kernel) will be square and symmetrical,
        with odd values, e.g. [3,3], [11,11], etc. Your code should be able to handle
        any filter that fits this description.
    Returns
    ----------
    image:  numpy.ndarray
        a convolved numpy array with the same dimensions as the input image
        with type np.float64.
    """
    img_H = image.shape[0]
    img_W = image.shape[1]
    filter_H = filter.shape[0]
    filter_W = filter.shape[1]

    pad = (filter_W - 1) // 2

    pad_img = np.zeros((img_H + pad * 2, img_W + pad * 2), dtype=np.float64)

    for row in range(img_H):
        row_array = image[row, :]
        new_row_array = np.pad(image[row, :], (pad, pad), 'reflect')
        pad_img[row + pad] = new_row_array

    for col in range(pad_img.shape[1]):
        new_col_array = np.pad(pad_img[pad: -pad, col], (pad, pad), 'reflect')
        pad_img[:, col] = new_col_array

    # convolution
    out_img = np.zeros((img_H, img_W), dtype=np.float64)
    for row in np.arange(img_H):
        for col in np.arange(img_W):
            f_matrix = pad_img[row: row + filter_W, col: col + filter_W]
            new_pixel = (filter * f_matrix[:, :]).sum()
            out_img[row, col] = new_pixel

    return out_img

def reduce_layer(image, kernel=generatingKernel(0.4)):
    """Convolve the input image with a generating kernel filter and then reduce its
    width and height each by a factor of two.

    For grading purposes, it is important that you use a reflected border
    (i.e., padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid
    region (i.e., the convolution operation should return an image of the same
    shape as the input) for the convolution. Subsampling must include the first
    row and column, skip the second, etc.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data type
        (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data type (e.g.,
        np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (ceil(r/2), ceil(c/2)). For instance, if the input is
        5x7, the output will be 3x4.
    """

    # WRITE YOUR CODE HERE.
    image = image.copy()
    image = image.astype(np.float64)
    img_H = image.shape[0]
    img_W = image.shape[1]

    out_img_H = int(np.ceil(img_H / 2))
    out_img_W = int(np.ceil(img_W / 2))

    convolved_image = convolutionManual(image, kernel)
    # convolved_image = convolutionCV2(image, kernel)

    out_img = np.zeros((out_img_H, out_img_W), dtype=np.float64)
    for y in np.arange(img_H):
        for x in np.arange(img_W):
            if y % 2 == 0 and x % 2 == 0:
                out_img[y // 2][x // 2] = convolved_image[y][x]
    return out_img

    raise NotImplementedError


def expand_layer(image, kernel=generatingKernel(0.4)):
    """Upsample the image to double the row and column dimensions, and then
    convolve it with a generating kernel.

    Upsampling the image means that every other row and every other column will
    have a value of zero. For grading purposes, it is important that you use a 
    reflected border (i.e., padding equivalent to cv2.BORDER_REFLECT101) and only 
    keep the valid region (i.e., the convolution operation should return an image 
    of the same shape as the input) for the convolution.

    Finally, multiply your output image by a factor of 4 in order to scale it
    back up. If you do not do this (and you should try it out without that)
    you will see that your images' brightness changes as you apply the convolution.
    You must explain why this happens in your report PDF.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (2*r, 2*c). For instance, if the input is 3x4, then
        the output will be 6x8.
    """

    # WRITE YOUR CODE HERE.
    img_H = image.shape[0]
    img_W = image.shape[1]

    out_img_H = img_H * 2
    out_img_W = img_W * 2

    expanded_img = np.zeros((out_img_H, out_img_W), dtype=np.float64)
    for y in np.arange(out_img_H):
        for x in np.arange(out_img_W):
            if y % 2 == 0 and x % 2 == 0:
                expanded_img[y][x] = image[y // 2][x // 2]
            else:
                expanded_img[y][x] = 0

    convolved_image = convolutionManual(expanded_img, kernel)
    # convolved_image = convolutionCV2(expanded_img, kernel)

    out_img = convolved_image * 4
    return out_img

    raise NotImplementedError


def gaussPyramid(image, levels):
    """Construct a pyramid from the image by reducing it by the number of
    levels specified by the input.

    Parameters
    ----------
    image : numpy.ndarray
        An image of dimension (r, c).

    levels : int
        A positive integer that specifies the number of reductions to perform.
        For example, levels=0 should return a list containing just the input
        image; levels = 1 should perform one reduction and return a list with
        two images. In general, len(output) = levels + 1.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list of arrays of dtype np.float. The first element of the list
        (output[0]) is layer 0 of the pyramid (the image itself). output[1] is
        layer 1 of the pyramid (image reduced once), etc.
    """

    # WRITE YOUR CODE HERE.
    temp = image.copy()
    result = [temp.astype(np.float64)]
    for i in range(levels):
        temp = reduce_layer(temp)
        result.append(temp)

    return result
    raise NotImplementedError


def laplPyramid(gaussPyr):
    """Construct a Laplacian pyramid from a Gaussian pyramid; the constructed
    pyramid will have the same number of levels as the input.

    Parameters
    ----------
    gaussPyr : list<numpy.ndarray(dtype=np.float)>
        A Gaussian Pyramid (as returned by your gaussPyramid function), which
        is a list of numpy.ndarray items.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of the same size as gaussPyr. This pyramid should
        be represented in the same way as guassPyr, as a list of arrays. Every
        element of the list now corresponds to a layer of the laplacian
        pyramid, containing the difference between two layers of the gaussian
        pyramid.

        NOTE: The last element of output should be identical to the last layer
              of the input pyramid since it cannot be subtracted anymore.

    Notes
    -----
        (1) Sometimes the size of the expanded image will be larger than the
        given layer. You should crop the expanded image to match in shape with
        the given layer. If you do not do this, you will get a 'ValueError:
        operands could not be broadcast together' because you can't subtract
        differently sized matrices.

        For example, if my layer is of size 5x7, reducing and expanding will
        result in an image of size 6x8. In this case, crop the expanded layer
        to 5x7.
    """

    # WRITE YOUR CODE HERE.
    result = [gaussPyr[-1]]
    for i in range(len(gaussPyr)-1, 0, -1):
        expanded = expand_layer(gaussPyr[i])
        if gaussPyr[i-1].shape[0] != expanded.shape[0]:
            expanded = expanded[:-1, :]
        if gaussPyr[i-1].shape[1] != expanded.shape[1]:
            expanded = expanded[:, :-1]
        L = gaussPyr[i - 1] - expanded
        result.insert(0, L)

    return result

    raise NotImplementedError


def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    """Blend two laplacian pyramids by weighting them with a gaussian mask.

    You should return a laplacian pyramid that is of the same dimensions as the
    input pyramids. Every layer should be an alpha blend of the corresponding
    layers of the input pyramids, weighted by the gaussian mask.

    Therefore, pixels where current_mask == 1 should be taken completely from
    the white image, and pixels where current_mask == 0 should be taken
    completely from the black image.

    (The variables `current_mask`, `white_image`, and `black_image` refer to
    the images from each layer of the pyramids. This computation must be
    performed for every layer of the pyramid.)

    Parameters
    ----------
    laplPyrWhite : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of an image constructed by your laplPyramid
        function.

    laplPyrBlack : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of another image constructed by your laplPyramid
        function.

    gaussPyrMask : list<numpy.ndarray(dtype=np.float)>
        A gaussian pyramid of the mask. Each value should be in the range
        [0, 1].

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list containing the blended layers of the two laplacian pyramids

    Notes
    -----
        (1) The input pyramids will always have the same number of levels.
        Furthermore, each layer is guaranteed to have the same shape as
        previous levels.
    """

    # WRITE YOUR CODE HERE.

    result = []
    for i in range(len(laplPyrWhite)):
        img_H = laplPyrWhite[i].shape[0]
        img_W = laplPyrWhite[i].shape[1]
        white = laplPyrWhite[i]
        black = laplPyrBlack[i]
        mask = gaussPyrMask[i]
        new_img = np.zeros((img_H, img_W), dtype=np.float64)
        for y in range(img_H):
            for x in range(img_W):
                k = mask[y][x]
                new_img[y][x] = white[y][x]*k + (1-k)*black[y][x]
        result.append(new_img)
    return result
    raise NotImplementedError


def collapse(pyramid):
    """Collapse an input pyramid.

    Approach this problem as follows: start at the smallest layer of the
    pyramid (at the end of the pyramid list). Expand the smallest layer and
    add it to the second to smallest layer. Then, expand the second to
    smallest layer, and continue the process until you are at the largest
    image. This is your result.

    Parameters
    ----------
    pyramid : list<numpy.ndarray(dtype=np.float)>
        A list of numpy.ndarray images. You can assume the input is taken
        from blend() or laplPyramid().

    Returns
    -------
    numpy.ndarray(dtype=np.float)
        An image of the same shape as the base layer of the pyramid.

    Notes
    -----
        (1) Sometimes expand will return an image that is larger than the next
        layer. In this case, you should crop the expanded image down to the
        size of the next layer. Look into numpy slicing to do this easily.

        For example, expanding a layer of size 3x4 will result in an image of
        size 6x8. If the next layer is of size 5x7, crop the expanded image
        to size 5x7.
    """

    # WRITE YOUR CODE HERE.
    new_pyramid = [pyramid[-1]]
    for i in reversed(range(len(pyramid)-1)):
        # expand previous
        expanded = expand_layer(new_pyramid[0])
        # constrain size
        if pyramid[i].shape[0] != expanded.shape[0]:
            expanded = expanded[:-1, :]
        if pyramid[i].shape[1] != expanded.shape[1]:
            expanded = expanded[:, :-1]
        new = pyramid[i] + expanded
        new_pyramid.insert(0, new)

    return new_pyramid[0]
    raise NotImplementedError
