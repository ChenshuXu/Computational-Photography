# Seam Carving

## Synopsis

The goal of this project is to replicate the results of published 
Computational Photography papers by following the methods described in the 
papers:

- Shai Avidan, Ariel Shamir. [Seam Carving for Content-Aware Image Resizing](http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html). (2007)
- Micheal Rubinstein, Ariel Shamir, Shai Avidan. [Improved Seam Carving for 
  Video Retargeting](http://www.faculty.idc.ac.il/arik/SCWeb/vidret/index.html). (2008)   **You are using methods from this paper for static images, 
  not video.**

Specifically, I implemented seam removal and insertion with both backward and forward energy methods.

## Results

### Results of beach image

#### Seam removal of 50% of seams with backward energy

Input image | my result | result in paper
|---|---|---|
![input image](images/base/beach.png) | ![50% removal](images/results/res_beach_back_rem.jpg) | ![comparison](images/comparison/comp_beach_back_rem.png)

### Results of bench image

#### Seam removal of 50% of seams with backward energy

Input image | my result | result in paper
|---|---|---|
![input image](images/base/bench.png) | ![50% removal](images/results/res_bench_back_rem.jpg) | ![comparison](images/comparison/comp_bench_back_rem.png)
| showing the seams removed on base image: | ![](images/results/res_bench_back_rem_red.jpg) | ![](images/comparison/comp_bench_back_rem_red.png)

#### Seam removal of 50% of seams with forward energy

Input image | my result | result in paper
|---|---|---|
![input image](images/base/bench.png) | ![50% removal](images/results/res_bench_forward_rem.jpg) | ![comparison](images/comparison/comp_bench_forward_rem.png)
| showing the seams removed on base image: | ![](images/results/res_bench_forward_rem_red.jpg) | ![](images/comparison/comp_bench_forward_rem_red.png)

- The algorithm will remove a connected seam from top to bottom that this removal will not affect much on the content in the image

### Result of dolphin image

#### Backwards seam insertion of 50% of seams

Input image | my result | result in paper
|---|---|---|
![input image](images/base/dolphin.png) | ![50% removal](images/results/res_dolphin_back_ins.jpg) | ![comparison](images/comparison/comp_dolphin_back_ins.png)
| showing the seams removed on base image: | ![](images/results/res_dolphin_back_ins_red.jpg) | ![](images/comparison/comp_dolphin_back_ins_red.png)

#### Backwards seam insertion of additional 50% of seams

Input image | my result | result in paper
|---|---|---|
![input image](images/base/dolphin.png) | ![](images/results/res_dolphin_back_5050.jpg) | ![](images/comparison/comp_dolphin_back_5050.png)

- The algorithm will insert a connected seam from top to bottom that this insertion will not affect much on the content in the image
