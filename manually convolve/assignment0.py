# CS6475 - Spring 2021
import cv2
import numpy as np


""" Assignment 0 - Introduction

This file has a number of basic image handling functions that you need
to write python3 code for in order to complete the assignment. We will
be using these operations throughout the course, and this assignment helps
to familiarize yourself with the cv2 and numpy libraries. Please write
the appropriate code, following the instructions in the docstring for each
function. Make sure that you know which commands and libraries you may or may
not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, prints, or writes
    over the image that you are being passed in. Any code line that you may
    have in your code to complete these actions must be commented out when
    you turn in your code. These actions may cause the autograder to crash,
    which will count as one of your limited attempts.

    2. DO NOT import any other libraries aside from the three libraries that we
    provide. You may not import anything else, you should be able to complete
    the assignment with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the course environment.
    You are responsible for ensuring that your code executes properly in the
    course environment and in the Gradescope autograder. Any changes you make
    outside the areas annotated for student code must not impact your performance
    on the autograder system.
    Thank you.
"""


def imageDimensions(image):
    """ This function takes your input image and returns its array shape.
    You may use a numpy command to find the shape.

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of three dimensions (HxWxD) and type np.uint8
        Your code should work for both grayscale (single channel)
        and color (three channel RGB) images.
    Returns
    ----------
    tuple:  tuple of numpy integers of type np.int
        the tuple returns the shape of the image ordered as
        (rows, columns, channels)
    """
    # WRITE YOUR CODE HERE.
    return image.shape
    
    # End of code  # raise NotImplementedError


def convolutionManual(image, filter):
    """ This function takes your input color (BGR) image and a square, symmetrical
    filter with odd dimensions and convolves them. The returned image must be the same
    size and type as the input image. We may input different sizes of filters,
    do not hard-code a value.

    Your code must use loops (it will be slower) and move the filter past each pixel
    of your image to determine the new pixel values. We assign this exercise to help
    you understand what filtering does and exactly how it is applied to an image.
    Almost every assignment will involve filtering, this is essential understanding.

    **************** FORBIDDEN COMMANDS ******************
    NO CV2 LIBRARY COMMANDS MAY BE USED IN THIS FUNCTION
    NO NUMPY CONVOLVE OR STRIDES COMMANDS
    In general, no use of commands that do your homework for you.
    The use of forbidden commands may result in a zero score
    and an honor code violation review
    ******************************************************

    Follow these steps:
    (1) Copy the image into a new array, so that you do not alter the image.
        Change the type to float64, for calculations.
    (2) From the shape of the filter, determine how many rows of padding you need.
    (3) Pad the copied image with mirrored padding. You must use the correct
        amount of padding of the correct type. For example: if a 7x7 filter is
        used you will require three rows/columns of padding around the image.
        A padded row will look like:

            image row [abcdefgh] ====> padded image row [cba|abcdefgh|hgf]

        Note1: If you use np.pad for this, each channel must be filtered separately.

    (5) Convolve, passing the filter (kernel) through all of the padded image.
        Save new pixel values in the result array. Don't overwrite the padded image!
    (6) Use numpy to round your values.
    (7) Convert to the required output type.

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8
        You image should be a 3D
    filter : numpy.ndarray
        A 2D numpy array of variable dinensions, with type np.float.
        The filter (also called kernel) will be square and symmetrical,
        with odd values, e.g. [3,3], [11,11], etc. Your code should be able to handle
        any filter that fits this description.
    Returns
    ----------
    image:  numpy.ndarray
        a convolved numpy array with the same dimensions as the input image
        with type np.uint8.
    """
    # WRITE YOUR CODE HERE.
    img_H = image.shape[0]
    img_W = image.shape[1]
    filter_H = filter.shape[0]
    filter_W = filter.shape[1]

    pad = (filter_W - 1) // 2

    pad_img = np.zeros((img_H + pad*2, img_W + pad*2, 3), dtype=np.float64)
    
    for row in range(img_H):
        row_array = image[row, :, :]
        new_row_array = np.zeros((pad_img.shape[1], 3), dtype=np.float64)
        for i in np.arange(3):
            channel = np.pad(image[row, :, i], (pad, pad), 'symmetric')
            new_row_array[:, i] = channel
        pad_img[row + pad] = new_row_array

    for col in range(pad_img.shape[1]):
        new_col_array = np.zeros((pad_img.shape[0], 3), dtype=np.float64)
        for i in np.arange(3):
            channel = np.pad(pad_img[pad: -pad, col, i], (pad, pad), 'symmetric')
            new_col_array[:, i] = channel
        pad_img[:, col] = new_col_array

    # convolution
    out_img = np.zeros((img_H, img_W, 3), dtype=np.uint8)
    for row in np.arange(img_H):
        for col in np.arange(img_W):
            new_pixel = np.zeros((3,), dtype=np.float64)
            f_matrix = pad_img[row: row + filter_W, col: col + filter_W]
            # get 3 pixels
            new_pixel[0] = (filter * f_matrix[:, :, 0]).sum()
            new_pixel[1] = (filter * f_matrix[:, :, 1]).sum()
            new_pixel[2] = (filter * f_matrix[:, :, 2]).sum()
            out_img[row, col] = np.rint(new_pixel)

    return out_img
    # End of code
    raise NotImplementedError


def convolutionCV2(image, filter):
    """ This function performs convolution on your image using a square,
    symmetrical odd-dimension 2D filter. You may use cv2 commands to complete this
    function. See the opencv docs for the version of cv2 used in the class env.
    Opencv has good tutorials for image processing.

    *** You may use any cv2 or numpy commands for this function ***

    Follow these steps:
    (1) same as convolutionManual.
    (2) Pad the copied image in the same mirrored style as convolutionManual:
            
            image row [abcdefgh] ====> padded image row [cba|abcdefgh|hgf]
            
        With cv2 commands you may not need to code the padding,
        but you must specify the correct Border Type.
        Note: Numpy and cv2 use different names for this type of padding. Be careful.
    (3) Complete the convolution
    (4) Finish is same as convolutionManual.

    Parameters
    ----------
    image : numpy.ndarray
        A 3-channel color (BGR) image, represented as a numpy array of
        dimensions (HxWxD) and type np.uint8
    filter : numpy.ndarray
        A 2D numpy array of variable dimension values and type np.float.
        The filter will be square and symmetrical, with odd values, e.g. [3,
        3], [11,11], etc. Your code should be able to handle any filter that
        fits this description.
    Returns
    ----------
    image:  numpy.ndarray
        a convolved numpy array with the same dimensions as the input image and
        type np.uint8.
    """
    # WRITE YOUR CODE HERE.
    row = image.shape[0]
    col = image.shape[1]
    pad = (filter.shape[0]-1) // 2

    rPadded = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
    newImg = cv2.filter2D(rPadded, -1, filter)
    return newImg[pad:pad+row, pad:pad+col]
    # End of code
    raise NotImplementedError


# ----------------------------------------------------------
if __name__ == "__main__":
    """ YOU MAY USE THIS AREA FOR CODE THAT ALLOWS YOU TO TEST YOUR FUNCTIONS.
    This section will not be graded, you can change or delete the code below.
    When you are ready to test your code on the autograder, comment out any code
    below, along with print statements you may have in your functions.
    Code here may cause the autograder to crash, which will cost you a try!
    """
    # WRITE YOUR CODE HERE

    # create filter
    n = 5   # filter dimension
    filter = np.ones((n, n))/n**2  # blur filter

    gaussian_filter = np.array([[1, 4, 7, 4, 1],
                                [4, 16, 26, 16, 4],
                                [7, 26, 41, 26, 7],
                                [4, 16, 26, 16, 4],
                                [1, 4, 7, 4, 1]]) / 273

    # read in your image, change image format to match
    image = cv2.imread("image.png")
    # OR
    # Create a small random toy image for testing
    # image = np.random.randint(0, 255, (8, 7, 3), dtype=(np.uint8))

    # save the original image in .png for the report
    # cv2.imwrite("image.png", image)

    print("\nimageDimensions:")
    dims = imageDimensions(image)
    print(dims, type(dims), type(dims[0]), len(dims))

    print('\nconvolutionManual:')
    convolve = convolutionManual(image, gaussian_filter)
    print(np.shape(convolve), convolve.dtype)
    cv2.imwrite("convolveManual.png", convolve)     # save image

    print('\nconvolutionCV2:')
    cv2convolve = convolutionCV2(image, gaussian_filter)
    print(np.shape(cv2convolve), cv2convolve.dtype)
    cv2.imwrite("convolveCV2.png", cv2convolve)     # save image
    
    # End of code
    # DON'T FORGET TO COMMENT OUT YOUR CODE IN THIS SECTION!
    pass
