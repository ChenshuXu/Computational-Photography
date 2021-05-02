"""
You can use this file to execute your code. You are NOT required
to use this file, and you are allowed to make ANY changes you want in
THIS file. This file will not be submitted with your assignment
or report, so if you write code improvements here, make sure
that you include important snippets in your writeup.

CODE ALONE IS NOT SUFFICIENT FOR CREDIT.
DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import errno
from os import path

import hdr as hdr


# Change the source folder and exposure times to match your own
# input images. Note that the response curve is calculated from
# a random sampling of the pixels in the image, so there may be
# variation in the output even for the example exposure stack

SRC_FOLDER = "images/source/sample"
EXPOSURE_TIMES = np.float64([1/8.0, 1/15.0, 1/30.0,
                             1/60.0, 1/125.0, 1/250.0, 1/500.0, 1/1000.0])
OUT_FOLDER = "images/output"
EXTENSIONS = set(["bmp", "jpg", "jpeg", "png", "tif", "tiff"])


def plotRadianceMap(rad_map, output_folder, channel):
    """ Enable at end of file, default=False
        This function produces one radiance_map for each channel.
        Plots are similar to Figure 8(d) in Debevek & Malik 1997.
    """
    filename = path.join(output_folder, 'radiance_map_'+str(channel)+'.png')
    plt.set_cmap('nipy_spectral')
    plt.imsave(filename, rad_map)
    plt.close()
    return
    

def plotHistogramCDF(histogram, cdf, output_folder, plot_filename):
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

    filename = path.join(output_folder, plot_filename+'.png')
    plt.savefig(filename)
    plt.close()
    return

def minmaxNorm(array, newmax=1.0):
    """ Takes a np.array and normalizes it between 0 and newmax. """
    arrmin = np.min(array)
    arrmax = np.max(array)
    return newmax*(array-arrmin)/(arrmax-arrmin)


def computeHDR(images, log_exposure_times, output_folder,
               smoothing_lambda=100., rad_plot=False):
    """Computational pipeline to produce the HDR images according to the
    process in the Debevec paper.

    The basic overview is to do the following for each channel:
        1. Sample pixel intensities from random locations through the
           image stack to determine the camera response curve
        2. Compute response curves for each color channel
        3. Build image radiance map from response curves

    Parameters
    ----------
    images : list<numpy.ndarray>
        A list containing an exposure stack of images

    log_exposure_times : numpy.ndarray
        The log exposure times for each image in the exposure stack

    smoothing_lambda : np.int (Optional)
        A constant value to correct for scale differences between
        data and smoothing terms in the constraint matrix -- source
        paper suggests a value of 100.

    Returns
    -------
    numpy.ndarray
        The resulting HDR with intensities scaled to fit uint8 range
    """
    images = [np.atleast_3d(i) for i in images]
    num_channels = images[0].shape[2]
    
    hdr_image = np.zeros(images[0].shape, dtype=np.float64)

    for channel in range(num_channels):
        # Collect the current layer of each input image from
        # the exposure stack
        layer_stack = [img[:, :, channel] for img in images]

        # Sample image intensities
        intensity_samples = hdr.sampleIntensities(layer_stack)

        # Compute Response Curve
        response_curve = hdr.computeResponseCurve(intensity_samples,
                                                  log_exposure_times,
                                                  smoothing_lambda,
                                                  hdr.linearWeight)

        # Build radiance map
        rad_map = hdr.computeRadianceMap(layer_stack,
                                             log_exposure_times,
                                             response_curve,
                                             hdr.linearWeight)
        
        if rad_plot:
            plotRadianceMap(rad_map, output_folder, channel)
            
        # rad_map output is logarithmic and must be normalized for images
        out = np.zeros(shape=rad_map.shape, dtype=rad_map.dtype)
        cv2.normalize(rad_map, out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        hdr_image[..., channel] = out
    
    return hdr_image


def computeHistogramEQ(image, output_folder, histo_plot=False):
    ''' Computes a histogram equalized image from the result image
        produced by computeHDR.
    NOTE: This function is NOT autograded as part of this assignment.
          You may modify it.
          
    Parameters
    ----------
    image : list<numpy.ndarray>
        three channel HDR image in BGR format
        
    Returns
    -------
    numpy.ndarray
        histogram equalized HDR with intensities scaled to uint8 range
    '''
    histogram = hdr.computeHistogram(image)
    cdf = hdr.computeCumulativeDensity(histogram)
    if histo_plot:
        plotHistogramCDF(histogram, cdf, output_folder, plot_filename="histCDF_basicHDR")
        
    histEQ = hdr.applyHistogramEqualization(image, cdf)
    if histo_plot:
        histogram = hdr.computeHistogram(histEQ)
        cdf = hdr.computeCumulativeDensity(histogram)
        plotHistogramCDF(histogram, cdf, output_folder, plot_filename="histCDF_histEQ")
    return histEQ


def main(image_files, output_folder, exposure_times,
         resize=False, align=False, rad_plot=False, histo_plot=False):
    """ Generate a basic HDR image and rad_map plot from the images
        in the source folder,
        - and -
        Generate histograms from basic and histogram-equalized (histEQ) HDR images.
    """
    
    # Print the information associated with each image -- use this
    # to verify that the correct exposure time is associated with each
    # image, or else you will get very poor results
    print("{:^30} {:>15}".format("Filename", "Exposure Time"))
    print("\n".join(["{:>30} {:^15.4f}".format(*v)
                     for v in zip(image_files, exposure_times)]))

    img_stack = [cv2.imread(name) for name in image_files
                 if path.splitext(name)[-1][1:].lower() in EXTENSIONS]

    if any([im is None for im in img_stack]):
        raise RuntimeError("One or more input files failed to load.")

    # RESIZE: Subsampling the images can reduce runtime for large files,
    # set resize to True in main call at end of this file.
    if resize:
        img_stack = [img[::4, ::4] for img in img_stack]
        print('images resized, new shape =', img_stack[0].shape)
    else: print('images full size, shape =', img_stack[0].shape)

    # ALIGN: Corrects small misalignments in images. If your images need more
    # intensive alignment, do additional processing on your own.
    # Set align to True in main call at end of this file.
    if align:
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(img_stack, img_stack)
        print('images aligned')
    else: print('image alignment disabled')

    # Compute HDR image and make rad_plot
    log_exposure_times = np.log(exposure_times)
    basicHDR = computeHDR(img_stack,
                          log_exposure_times,
                          output_folder,
                          rad_plot=rad_plot)
    cv2.imwrite(path.join(output_folder, "basicHDR.png"), basicHDR)
    print("Basic HDR image complete")

    # compute histogram equalized HDR image and make histogram plots
    histEQ = computeHistogramEQ(basicHDR.astype(np.uint8),
                                output_folder,
                                histo_plot=histo_plot)
    cv2.imwrite(path.join(output_folder, "histEQ.png"), histEQ)
    print("histogram EQ image complete")
    
    # compute best HDR
    bestHDR_image = hdr.bestHDR(basicHDR)
    cv2.imwrite(path.join(output_folder, "bestHDR.png"), bestHDR_image)
    
    histogram = hdr.computeHistogram(bestHDR_image.astype(np.uint8))
    cdf = hdr.computeCumulativeDensity(histogram)
    plotHistogramCDF(histogram, cdf, output_folder,
                     plot_filename="histCDF_bestHDR")
    print("best HDR image complete")
    
    return


if __name__ == "__main__":
    """ Generate HDR images and rad_map plot from the images
        in the SRC_FOLDER directory
    """

    # np.random.seed()  # set a fixed seed if you want repeatable results

    src_contents = os.walk(SRC_FOLDER)
    dirpath, _, fnames = next(src_contents)

    image_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join(OUT_FOLDER, image_dir)

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print("Processing '" + image_dir + "' folder...")

    image_files = sorted([os.path.join(dirpath, name) for name in fnames
                          if not name.startswith(".")])

    main(image_files,
         output_dir,
         EXPOSURE_TIMES,
         resize=False,
         align=False,
         rad_plot=True,
         histo_plot=True)
    

