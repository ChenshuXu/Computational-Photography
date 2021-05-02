# Panoramas

## Synopsis

In this assignment, I implemented a pipeline to align and stitch together a series of images into a panorama. I used my code on my own pictures to make a panorama.

The pipeline looks like this:
![img.png](img.png)

## Results

### Blend result of sample images

Input images:

image 1 | image 2 | image 3
|---|---|---|
![1](images/source/sample/1.jpg) | ![2](images/source/sample/2.jpg) | ![3](images/source/sample/3.jpg)

Result:
![](images/output/sample/output.jpg)

### Blend result of my own images

Input images:

image 1 | image 2 | image 3
|---|---|---|
![1](images/source/sample2/1.jpg) | ![2](images/source/sample2/2.jpg) | ![3](images/source/sample2/3.jpg)

Interim result:

![left_image_alpha](left_image_alpha.png)
After getting the homography, I apply the transformation to the image. Also, create a new canvas that is large enough to include two input images.

![right_image_alpha](right_image_alpha.png)
I use distance map to generate a mask and use this mask to blend two images together.
In the overlap area, pixels closer to the left image will have more weight from the left image. Pixels closer to the right image will have more weight from the right image.


Result:
![](images/output/sample2/output.jpg)

The blending result of my input images looks good. My input images are very easy to blend. They have clear features between images to be detected by algorithm. They also have same exposure values and that makes the overlapping area almost invisible. 
