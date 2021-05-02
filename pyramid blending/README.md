# Pyramid Blending

## Synopsis
In this assignment, I put together a pyramid blending pipeline that will combine separate images into a seamlessly blended image.

Reference papers:
  - [“The Laplacian Pyramid as a Compact Image Code"](http://persci.mit.edu/pub_pdfs/pyramid83.pdf) (Burt and Adelson; 1983)
  - [“A Multiresolution Spline With Application to Image Mosaics”](http://persci.mit.edu/pub_pdfs/spline83.pdf) (Burt and Adelson; 1983)

## Results

### Blend input of image set 1:

black image | mask | white image
|----|----|----|
![](images/source/sample/black.jpg) | ![](images/source/sample/mask.jpg) | ![](images/source/sample/white.jpg)

Interim result:

Gauss pyramid of black image | Gauss pyramid of mask image | Gauss pyramid of white image
|----|----|----|
![gauss_pyr_black](images/output/sample/gauss_pyr_black.png) | ![gauss_pyr_mask](images/output/sample/gauss_pyr_mask.png) | ![gauss_pyr_white](images/output/sample/gauss_pyr_white.png)

Laplacian pyramid of black image | Laplacian pyramid of white image | Output pyramid
|----|----|----|
![lapl_pyr_black](images/output/sample/lapl_pyr_black.png) | ![lapl_pyr_white](images/output/sample/lapl_pyr_white.png) | ![outpyr](images/output/sample/outpyr.png)

Blend result:
![](images/output/sample/result.png)

### Blend input of image set 2:

black image | mask | white image
|----|----|----|
![](images/source/my_sample_0/black.jpg) | ![](images/source/my_sample_0/mask.jpg) | ![](images/source/my_sample_0/white.jpg)

Interim result:

Gauss pyramid of black image | Gauss pyramid of mask image | Gauss pyramid of white image
|----|----|----|
![gauss_pyr_black](images/output/my_sample_0/gauss_pyr_black.png) | ![gauss_pyr_mask](images/output/my_sample_0/gauss_pyr_mask.png) | ![gauss_pyr_white](images/output/my_sample_0/gauss_pyr_white.png)

Laplacian pyramid of black image | Laplacian pyramid of white image | Output pyramid
|----|----|----|
![lapl_pyr_black](images/output/my_sample_0/lapl_pyr_black.png) | ![lapl_pyr_white](images/output/my_sample_0/lapl_pyr_white.png) | ![outpyr](images/output/my_sample_0/outpyr.png)

Blend result:
![](images/output/my_sample_0/result.png)

The  blend  allowed  me to smoothly combine two images even with an unprecise mask that has a sharp edge on the blending position. The algorithm will take information from both sides and blend them well.
