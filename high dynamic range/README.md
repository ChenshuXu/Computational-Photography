# High Dynamic Range Imaging

## Synopsis

This assignment focuses on the core algorithms behind computing HDR images based on the paper ["Recovering High
 Dynamic Range Radiance Maps from Photographs” by Debevec & Malik](https://www.pauldebevec.com/Research/HDR/). I implemented the HDR algorithms from the paper and used [contrast limited adaptive histogram equalization (CLAHE)](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization) to improve the basic HDR image

## Results

### Result of sample image set

Input stack of images:

1 | 2 | 3
|:---:|:---:|:---:|
![00](submission/sample-00.png) | ![00](submission/sample-01.png) | ![00](submission/sample-02.png)
 4 | 5 | 6
![00](submission/sample-03.png) | ![00](submission/sample-04.png) | ![00](submission/sample-05.png)

Radiance map of BGR channels:

Blue channel | Green channel | Red channel
|---|---|---|
![B](submission/radiance_map_0.png) | ![G](submission/radiance_map_1.png) | ![R](submission/radiance_map_2.png)

- The red and bright area represent the high light intensity 
- The blue and deep area represents the low light intensity

Basic HDR image:

basic HDR | histogram
|---|---|
![basic HDR](submission/basicHDR.png) | ![hist of basic HDR](submission/histCDF_basicHDR.png)

- Produced by min-max normalization of radiance values in each color channel
- The blue bar represents the number of pixels with that brightness value
- The red line represents the cumulative density of pixels

Histogram equalized picture | histogram
|---|---|
![histEQ](submission/histEQ.png) | ![](submission/histCDF_histEQ.png)

- Produced by applying histogram equalization
- The contrast in increased after histogram equalization. However, it introduced too much contrast in the whole image.

After improvement with CLAHE, the best HDR picture | histogram
|---|---|
![basic HDR](submission/bestHDR.png) | ![](submission/histCDF_bestHDR.png)

- I used [contrast limited adaptive histogram equalization (CLAHE)](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization) to improve the basic HDR image.
- Each distinct section of the image get improved by histogram equalization

### Result of another image set

Input stack of images from [HDR Pics to play with: High Five](https://farbspiel-photo.com/learn/hdr-pics-to-play-with/high-five-ppw):

Parameters:

Img | Exposure | Aperture | ISO
|---|---|---|---|
01 | 1/8 | f5.6 | 400
02 | 1/15 | f5.6 | 400 
03 | 1/30 | f5.6 | 400
04 | 1/60 | f5.6 | 400
05 | 1/125 | f5.6 | 400
06 | 1/250 | f5.6 | 400
07 | 1/500 | f5.6 | 400

1 | 2 | 3
|:---:|:---:|:---:|
![00](submission2/input_01.png) | ![00](submission2/input_02.png) | ![00](submission2/input_03.png)
 4 | 5 | 6
![00](submission2/input_04.png) | ![00](submission2/input_05.png) | ![00](submission2/input_06.png)
 7 | 8
![00](submission2/input_07.png) | ![00](submission2/input_08.png)

Radiance map of BGR channels:

Blue channel | Green channel | Red channel
|---|---|---|
![B](submission2/radiance_map_0.png) | ![G](submission2/radiance_map_1.png) | ![R](submission2/radiance_map_2.png)

Basic HDR image:

basic HDR | histogram
|---|---|
![basic HDR](submission2/basicHDR.png) | ![hist of basic HDR](submission2/histCDF_basicHDR.png)

Histogram equalized picture | histogram
|---|---|
![histEQ](submission2/histEQ.png) | ![](submission2/histCDF_histEQ.png)

After improvement with CLAHE, the best HDR picture | histogram
|---|---|
![basic HDR](submission2/bestHDR.png) | ![](submission2/histCDF_bestHDR.png)

