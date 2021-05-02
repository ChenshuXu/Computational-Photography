# Manually convolve

In this introductory assignment, I did some Python programming **using an
 image that you took with your camera or smartphone**. I used NumPy and
OpenCV libraries to work with the image array and write my own convolution
   function.

The kernel I use:
```
[[1, 4, 7, 4, 1],
 [4, 16, 26, 16, 4],
 [7, 26, 41, 26, 7],
 [4, 16, 26, 16, 4],
 [1, 4, 7, 4, 1]]
```

My input image:
![my input image](image.png)

My output image after manual convolution:
![convolveManual.png](convolveManual.png)

Output image using cv2.filter2D:
![convolveCV2.png](convolveCV2.png)