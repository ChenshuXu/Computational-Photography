""" Panoramas

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file.

    2. DO NOT import any other libraries aside from those that we provide.
    You may not import anything else, and you should be able to complete
    the assignment with the given libraries (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the provided virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import scipy as sp
import cv2
import scipy.ndimage as nd


def getImageCorners(image):
    """Return the x, y coordinates for the four corners bounding the input
    image and return them in the shape expected by the cv2.perspectiveTransform
    function. (The boundaries should completely encompass the image.)

    Parameters
    ----------
    image : numpy.ndarray
        Input can be a grayscale or color image

    Returns
    -------
    numpy.ndarray(dtype=np.float32)
        Array of shape (4, 1, 2).  The precision of the output is required
        for compatibility with the cv2.warpPerspective function.

    Notes
    -----
        (1) Review the documentation for cv2.perspectiveTransform (which will
        be used on the output of this function) to see the reason for the
        unintuitive shape of the output array.

        (2) When storing your corners, they must be in (X, Y) order -- keep
        this in mind and make SURE you get it right.
    """
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE
    img_H = image.shape[0]
    img_W = image.shape[1]
    top_left = (0, img_H)
    top_right = (img_W, img_H)
    button_left = (0, 0)
    button_right = (img_W, 0)
    corners[0, 0] = button_left
    corners[1, 0] = button_right
    corners[2, 0] = top_right
    corners[3, 0] = top_left
    return corners
    raise NotImplementedError


def findMatchesBetweenImages(image_1, image_2, num_matches):
    """Return the top list of matches between two input images.

    Parameters
    ----------
    image_1 : numpy.ndarray
        The first image (can be a grayscale or color image)

    image_2 : numpy.ndarray
        The second image (can be a grayscale or color image)

    num_matches : int
        The number of keypoint matches to find. If there are not enough,
        return as many matches as you can.

    Returns
    -------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_1

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_2

    matches : list<cv2.DMatch>
        A list of the top num_matches matches between the keypoint descriptor
        lists from image_1 and image_2

    Notes
    -----
        (1) You will not be graded for this function.
    """
    feat_detector = cv2.ORB_create(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),
                     key=lambda x: x.distance)[:num_matches]
    return image_1_kp, image_2_kp, matches


def findHomography(image_1_kp, image_2_kp, matches):
    """Returns the homography describing the transformation between the
    keypoints of image 1 and image 2.

        ************************************************************
          Before you start this function, read the documentation
                  for cv2.DMatch, and cv2.findHomography
        ************************************************************

    Follow these steps:

        1. Iterate through matches and store the coordinates for each
           matching keypoint in the corresponding array (e.g., the
           location of keypoints from image_1_kp should be stored in
           image_1_points).

            NOTE: Image 1 is your "query" image, and image 2 is your
                  "train" image. Therefore, you index into image_1_kp
                  using `match.queryIdx`, and index into image_2_kp
                  using `match.trainIdx`.

        2. Call cv2.findHomography() and pass in image_1_points and
           image_2_points, using method=cv2.RANSAC and
           ransacReprojThreshold=5.0.

        3. cv2.findHomography() returns two values: the homography and
           a mask. Ignore the mask and return the homography.

    Parameters
    ----------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the first image

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the second image

    matches : list<cv2.DMatch>
        A list of matches between the keypoint descriptor lists

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2
    """
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE.
    for i in range(len(matches)):
        mat = matches[i]
        image_1_points[i, 0] = image_1_kp[mat.queryIdx].pt
        image_2_points[i, 0] = image_2_kp[mat.trainIdx].pt
    M_hom, inliers = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=5.0)
    return M_hom
    raise NotImplementedError


def getBoundingCorners(corners_1, corners_2, homography):
    """Find the coordinates of the top left corner and bottom right corner of a
    rectangle bounding a canvas large enough to fit both the warped image_1 and
    image_2.

    Given the 8 corner points (the transformed corners of image 1 and the
    corners of image 2), we want to find the bounding rectangle that
    completely contains both images.

    Follow these steps:

        1. Use the homography to transform the perspective of the corners from
           image 1 (but NOT image 2) to get the location of the warped
           image corners.

        2. Get the boundaries in each dimension of the enclosing rectangle by
           finding the minimum x, maximum x, minimum y, and maximum y.

    Parameters
    ----------
    corners_1 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 1

    corners_2 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 2

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_min, y_min) -- the coordinates of the
        top left corner of the bounding rectangle of a canvas large enough to
        fit both images (leave them as floats)

    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_max, y_max) -- the coordinates of the
        bottom right corner of the bounding rectangle of a canvas large enough
        to fit both images (leave them as floats)

    Notes
    -----
        (1) The inputs may be either color or grayscale, but they will never
        be mixed; both images will either be color, or both will be grayscale.

        (2) Python functions can return multiple values by listing them
        separated by commas.

        Ex.
            def foo():
                return [], [], []
    """
    # WRITE YOUR CODE HERE
    # Use the homography to transform the perspective of the corners from image 1
    x_max = max(corners_2[:, :, 0])[0]
    x_min = min(corners_2[:, :, 0])[0]
    y_max = max(corners_2[:, :, 1])[0]
    y_min = min(corners_2[:, :, 1])[0]

    for corner in corners_1:
        # print("old corner {}".format(corner))
        formated_corner = np.array([[corner[0][0]], [corner[0][1]], [1]], dtype=np.float64)
        new_pt = np.matmul(homography, formated_corner)
        w = new_pt[2, 0]
        x = new_pt[0, 0] / w
        y = new_pt[1, 0] / w
        # print("new corner {}".format((x, y)))
        x_max = max(x, x_max)
        x_min = min(x, x_min)
        y_max = max(y, y_max)
        y_min = min(y, y_min)

    min_xy = np.array([x_min, y_min], dtype=np.float64)
    max_xy = np.array([x_max, y_max], dtype=np.float64)
    return min_xy, max_xy
    raise NotImplementedError


def warpCanvas(image, homography, min_xy, max_xy):
    """Warps the input image according to the homography transform and embeds
    the result into a canvas large enough to fit the next adjacent image
    prior to blending/stitching.

    Follow these steps:

        1. Create a translation matrix (numpy.ndarray) that will shift
           the image by x_min and y_min. This looks like this:

            [[1, 0, -x_min],
             [0, 1, -y_min],
             [0, 0, 1]]

        2. Compute the dot product of your translation matrix and the
           homography in order to obtain the homography matrix with a
           translation.

        NOTE: Matrix multiplication (dot product) is not the same thing
              as the * operator (which performs element-wise multiplication).
              See Numpy documentation for details.

        3. Call cv2.warpPerspective() and pass in image 1, the combined
           translation/homography transform matrix, and a vector describing
           the dimensions of a canvas that will fit both images.

        NOTE: cv2.warpPerspective() is touchy about the type of the output
              shape argument, which should be an integer.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale or color image (test cases only use uint8 channels)

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between two sequential
        images in a panorama sequence

    min_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the top left corner of a
        canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    max_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the bottom right corner of
        a canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    Returns
    -------
    numpy.ndarray(dtype=image.dtype)
        An array containing the warped input image embedded in a canvas
        large enough to join with the next image in the panorama; the output
        type should match the input type (following the convention of
        cv2.warpPerspective)

    Notes
    -----
        (1) You must explain the reason for multiplying x_min and y_min
        by negative 1 in your writeup.
        Because x_min and y_min are movements of the second picture to match the first picture in the calculation
        of homography. In this step, we are moving the first picture to side, and place the second picture in center.
        So, we are moving the first picture to match the second picture. So, the direction is opposite to the calculated
        homography transform direction.
    """
    # canvas_size properly encodes the size parameter for cv2.warpPerspective,
    # which requires a tuple of ints to specify size, or else it may throw
    # a warning/error, or fail silently
    canvas_size = tuple(np.round(max_xy - min_xy).astype(np.int))
    # WRITE YOUR CODE HERE
    x_min = min_xy[0]
    y_min = min_xy[1]
    translation_matrix = np.array([[1, 0, -x_min],
             [0, 1, -y_min],
             [0, 0, 1]], dtype=np.float64)
    new_homography = np.dot(translation_matrix, homography)
    img_pano = cv2.warpPerspective(image, new_homography, canvas_size)
    return img_pano
    raise NotImplementedError


def createImageMask(image):
    '''
    This method creates a mask representing all the "valid" pixels of an image
    and excludes the black border pixels that are introduced as part of 
    processing.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale or color image (test cases only use uint8 channels)
    
    Returns
    -------
    numpy.ndarray(dtype=dtype.bool)
        An 2d-array of bools with the same height and width as the input image.
        True values indicate any pixel that is part of the image and false values
        indicate empty border pixels.
    
    Notes
    -----
    There are a number of ways to find the mask.  It is recommended that you 
    read the documentation for cv2.findContours and cv2.drawContours. If you
    choose to use cv2.findContours, use mode=cv2.RETR_EXTERNAL,method = cv2.CHAIN_APPROX_SIMPLE.
    '''
    h, w, d = np.atleast_3d(image).shape
    mask = np.zeros((h, w), dtype=np.bool)
    # TODO: WRITE YOUR CODE HERE
    if d == 3:
        img_gray = np.sum(image, axis=2)
        img_gray = np.where(img_gray == 0, 0, 255).astype(np.uint8)
    else:
        img_gray = np.where(image == 0, 0, 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_img, contours, -1, 255, thickness=cv2.FILLED)
    # cv2.imwrite("contour.png", mask_img)
    mask = mask_img.astype(bool)
    return mask
    raise NotImplementedError
    

def createRegionMasks(left_mask, right_mask):
    '''
    This method will take two masks, one created from the warped image and one 
    created from the second, translated image.  It will generate three masks:
    one for a cutout of True values of the left mask that do not contain True
    values in the right mask, one that represents the overlap region between
    the two masks and a third mask that represents all of the True values of 
    the right mask that are not part of the overlap region.

    Parameters
    ----------
    left_mask : numpy.ndarray(dtype=dtype.bool)
        An array that contains a mask representing the post-warp image with True values
        indicating the valid image region of the *warped* image.

    right_mask : numpy.ndarray(dtype=dtype.bool)
        An array that contains a mask representing the post-translated image with True values
        indicating the valid image region of the *translated* image.
    
    Returns
    -------
    tuple(numpy.ndarray(dtype=dtype.bool),numpy.ndarray(dtype=dtype.bool),numpy.ndarray(dtype=dtype.bool))
    First argument to tuple:
        An 2d-array of bools with the same height and width as the input image.
        True values represent all pixels that are part of the left_mask AND NOT part
        of the right_mask.
    
    Second argument to tuple:
        An 2d-array of bools with the same height and width as the input image.
        True values represent all pixels that are part of the left_mask AND part
        of the right_mask.  This argument represents the overlap region.

    Third argument to tuple:
        An 2d-array of bools with the same height and width as the input image.
        True values represent all pixels that are part of the right_mask AND NOT part
        of the left_mask.
    
    Notes
    -----
    Read the documentation on numpy's np.bitwise_* methods.
    '''
    # TODO: WRITE YOUR CODE HERE
    overlap_mask = np.bitwise_and(left_mask, right_mask)
    left_only_mask = np.bitwise_xor(left_mask, overlap_mask)
    right_only_mask = np.bitwise_xor(right_mask, overlap_mask)
    return left_only_mask, overlap_mask, right_only_mask
    raise NotImplementedError

def findDistanceToMask(mask):
    '''
    This method will calculate the distance from each pixel marked as True
    by the mask to the CLOSEST pixel marked as False by the mask.
  
    Parameters
    ----------
    mask : numpy.ndarray(dtype=dtype.bool)
        An array that contains a mask.
    
    Returns
    -------
    nump.ndarray(dtype=np.float)
    
    An array that is the same shape as the mask.  Every element contains
    the distance from that pixel location to the nearest False value.

    Notes
    -----
    Refer to Notebook 1 on how to use the scipy.ndimage.distance_transform_edt method.
    Also note that you may have to flip the mask values for the distance transform 
    method to work properly.
    '''
    # TODO: WRITE YOUR CODE HERE
    fliped_mask = np.invert(mask).astype(np.float64)
    return nd.distance_transform_edt(fliped_mask)

    raise NotImplementedError

def generateAlphaWeights(left_distance, right_distance):
    '''
    This method takes two distance maps and generates a set of 
    alpha weights to be used to create a smooth gradient used
    to select colors from the left and right images according
    to the ratio of distances to either the left or right mask
    borders in the overlap region.
    
    Parameters
    ----------
    left_distance : numpy.ndarray(dtype=dtype.float)
        An array the same size as the resultant image containing
        distances from every pixel outside the left cutout mask
        to the nearest pixel within the left cutout mask.
    
    right_distance : numpy.ndarray(dtype=dtype.float)
        An array the same size as the resultant image containing
        distances from every pixel outside the right cutout mask
        to the nearest pixel within the right cutout mask.

    Returns
    -------
    numpy.ndarray(dtype=np.float)
        An array of ratios the same size as the resultant image
        representing the ratio of the distances of each pixel from
        the **right** distance mask to the sum of the right and 
        left distance masks.
    '''
    # TODO: WRITE YOUR CODE HERE
    sum_distance_mask = left_distance + right_distance
    ratio_mask = right_distance / sum_distance_mask
    return ratio_mask

    raise NotImplementedError



def blendImagePair(image_1, image_2, num_matches):
    """This function takes two images as input and fits them onto a single
    canvas by performing a homography warp on image_1 so that the keypoints
    in image_1 aligns with the matched keypoints in image_2.

    **************************************************************************

       The most common implementation is to use alpha blending to take the
       average between the images for the pixels that overlap, but you are
                    encouraged to use other approaches.

    **************************************************************************

    Parameters
    ----------
    image_1 : numpy.ndarray
        A grayscale or color image

    image_2 : numpy.ndarray
        A grayscale or color image

    num_matches : int
        The number of keypoint matches to find between the input images

    Returns:
    ----------
    numpy.ndarray
        An array containing both input images on a single canvas

    Notes
    -----
        (1) This function is not graded by the autograder. It will be scored
        manually by the TAs.

        (2) The inputs may be either color or grayscale, but they will never be
        mixed; both images will either be color, or both will be grayscale.

        You are free to create your blend however you like, but the methods 
        above will help you create a gradient for the overlap region of the two 
        images which you can use as an alpha blend.

    """
    kp1, kp2, matches = findMatchesBetweenImages(image_1, image_2, num_matches)
    homography = findHomography(kp1, kp2, matches)
    corners_1 = getImageCorners(image_1)
    corners_2 = getImageCorners(image_2)
    min_xy, max_xy = getBoundingCorners(corners_1, corners_2, homography)
    left_image = warpCanvas(image_1, homography, min_xy, max_xy)
    output_image = np.zeros_like(left_image)
    right_image = np.zeros_like(left_image)
    min_xy = min_xy.astype(np.int)
    right_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],
                 -min_xy[0]:-min_xy[0] + image_2.shape[1]] = image_2

    # TODO: WRITE YOUR CODE HERE.
    # cv2.imwrite("left_image.png", left_image)
    # cv2.imwrite("right_image.png", right_image)

    left_mask = createImageMask(left_image)
    right_mask = createImageMask(right_image)
    left_only_mask, overlap_mask, right_only_mask = createRegionMasks(left_mask, right_mask)

    # left_distance_mask = findDistanceToMask(left_mask)
    left_only_distance_mask = findDistanceToMask(left_only_mask)
    # overlap_distance_mask = findDistanceToMask(overlap_mask)
    # right_distance_mask = findDistanceToMask(right_mask)
    right_only_distance_mask = findDistanceToMask(right_only_mask)
    left_ratio_mask = generateAlphaWeights(left_only_distance_mask, right_only_distance_mask)
    right_ratio_mask = generateAlphaWeights(right_only_distance_mask, left_only_distance_mask)

    left_image_ch = []
    for ch in np.rollaxis(left_image, -1):
        left_image_ch.append(ch * left_ratio_mask)

    right_image_ch = []
    for ch in np.rollaxis(right_image, -1):
        right_image_ch.append(ch * right_ratio_mask)

    new_left_image = np.dstack(left_image_ch)
    new_right_image = np.dstack(right_image_ch)
    # cv2.imwrite("left_image_alpha.png", new_left_image)
    # cv2.imwrite("right_image_alpha.png", new_right_image)

    output_image = new_left_image + new_right_image

    return output_image.astype(np.uint8)
    # END OF FUNCTION
