# Computational-Photography

## Set up the Virtual Environment

- This course uses Python 3.6, so download and install the latest Python3 supported version of Anaconda for your OS [here](https://www.anaconda.com/download) (scroll to the bottom to see the list of installers). Do NOT download a Python2 version. Downloading the latest Python3 version of Anaconda and creating an environment with the `cs6475.yml` (which specifies Python 3.6), should work. 

**Resources:** Anaconda Documentation for installation on [Windows](https://docs.anaconda.com/anaconda/install/windows/), [macOS](https://docs.anaconda.com/anaconda/install/mac-os/), and [Linux](https://docs.anaconda.com/anaconda/install/linux/)

- From the local course repository directory you created when you cloned the remote, create the virtual environment by running 
```
conda env create -f environment.yml
```

You can then activate the virtual environment from a terminal by executing the following command:

```
conda activate computational-photography
```

**NOTE 1:** `conda activate` and `conda deactivate` only works on conda 4.6 and later. For conda versions prior to 4.6, Linux and macOS users should use `conda source activate` and `conda source deactivate`.

**NOTE 2:** To remove the environment by running 
```
conda env remove --name computational-photography
``` 

**Resource:** Anaconda Documentation - [Creating an environment from an environment yml file](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

Once the virtual environment has been activated, you can execute code from the same terminal. It's worth mentioning that popular Python IDEs, such as PyCharm, VSCode or Atom, facilitate using an Anaconda environment as the interpreter for your project or repository. This course setup will not go over those steps, as they will vary from tool to tool.


### Validate Environment Setup

You can test that your environment is setup correctly by opening a terminal, activating the virtual environment, and running the `test.py` file in the root folder of this repository. The test file performs basic checks to confirm that the required versions of all packages are correctly installed.

```
~$ conda activate computational-photography
(computational-photography) ~$python test.py     # Note: its possible for some that *python3 test.py* may work if this doesn't 
.....
----------------------------------------------------------------------
Ran 5 tests in 0.272s

OK
```

## Manually convolve

In this introductory assignment, I did some Python programming **using an
 image that you took with your camera or smartphone**. I used NumPy and
OpenCV libraries to work with the image array and write my own convolution
   function.

## Pyramid Blending

In this assignment, I put together a pyramid blending pipeline that will combine separate images into a seamlessly blended image.

## Panoramas

In this assignment, I implemented a pipeline to align and stitch together a series of images into a panorama. I used my code on my own pictures to make a panorama.

## High Dynamic Range

This assignment focuses on the core algorithms behind computing HDR images based on the paper ["Recovering High
 Dynamic Range Radiance Maps from Photographs” by Debevec & Malik](https://www.pauldebevec.com/Research/HDR/). I implemented the HDR algorithms from the paper and used [contrast limited adaptive histogram equalization (CLAHE)](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization) to improve the basic HDR image

## Seam carving

The goal of this project is to replicate the results of published 
Computational Photography papers by following the methods described in the 
papers:

- Shai Avidan, Ariel Shamir. [Seam Carving for Content-Aware Image Resizing](http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html). (2007)
- Micheal Rubinstein, Ariel Shamir, Shai Avidan. [Improved Seam Carving for 
  Video Retargeting](http://www.faculty.idc.ac.il/arik/SCWeb/vidret/index.html). (2008)   **You are using methods from this paper for static images, 
  not video.**

Specifically, I implemented seam removal and insertion with both backward and forward energy methods.

## Video stablization

For this project I replicated results of the video stabilization method in paper:

- Matthias Grundman, Vivek Kwatra, Irfan Essa. [Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths](https://www.cc.gatech.edu/cpl/projects/videostabilization/). (2011)

