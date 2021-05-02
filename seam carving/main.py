"""
You can use this file to execute your code. You are NOT required
to use this file, and ARE ALLOWED to make ANY changes you want in
THIS file. This file will not be submitted with your assignment
or report.

DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""
import cv2

import os
import errno

import time


from seam_carving import (beach_backward_removal,
                          dolphin_backward_insert,
                          dolphin_backward_5050,
                          bench_backward_removal,
                          bench_forward_removal,  
                          car_backward_insert,
                          car_forward_insert,
                          difference_image,
                          numerical_comparison)

SOURCE_FOLDER = "images"
OUT_FOLDER = "images/results"


def generate_base_arrays():
    prefix = "images/base/"
    try:
        beach = cv2.imread(prefix + "beach.png")
        dolphin = cv2.imread(prefix + "dolphin.png")
        bench = cv2.imread(prefix + "bench.png")
        car = cv2.imread(prefix + "car.png")
        print("base image arrays generated")
        return beach, dolphin, bench, car
    except:
        print("Error generating base image arrays")
    


def generate_result_images(beach, dolphin, bench, car):
    """ Call seam_carving to generate each of the ten result images
    """
    try:
        result = beach_backward_removal(beach,pctSeams=0.50)
        cv2.imwrite("images/results/res_beach_back_rem.jpg", result)
        print("done res_beach_back_rem")
    except: print('Error generating res_beach_back_rem ')

    try:
        result = dolphin_backward_insert(dolphin, pctSeams=0.50,redSeams=True)
        cv2.imwrite("images/results/res_dolphin_back_ins_red.jpg", result)
        print("done res_dolphin_back_ins_red")
    except: print('Error generating res_dolphin_back_ins_red')

    try:
        result = dolphin_backward_insert(dolphin, pctSeams=0.50, redSeams=False)
        cv2.imwrite("images/results/res_dolphin_back_ins.jpg", result)
        print("done res_dolphin_back_ins")
    except: print('Error generating res_dolphin_back_ins')

    try:
        # use the first dolphin expansion as an input
        dolphin_array = cv2.imread("images/results/res_dolphin_back_ins.jpg")
        array_5050 = dolphin_backward_5050(dolphin_array, pctSeams=0.50)
        cv2.imwrite("images/results/res_dolphin_back_5050.jpg", array_5050)
        print("done res_dolphin_back_5050")
    except: print('Error generating res_dolphin_back_5050')

    try:
        cv2.imwrite("images/results/res_bench_back_rem.jpg",
                    bench_backward_removal(bench, pctSeams=0.50))
        print("done res_bench_back_rem")
    except: print('Error generating res_bench_back_rem')

    try:
        cv2.imwrite("images/results/res_bench_back_rem_red.jpg",
                    bench_backward_removal(bench,pctSeams=0.50, redSeams=True))
        print("done res_bench_back_rem_red")
    except: print('Error generating res_bench_back_rem_red')

    try:
        cv2.imwrite("images/results/res_bench_forward_rem.jpg",
                    bench_forward_removal(bench, pctSeams=0.50))
        print("done res_bench_forward_rem")
    except: print('Error generating res_bench_forward_rem')

    try:
        cv2.imwrite("images/results/res_bench_forward_rem_red.jpg",
                    bench_forward_removal(bench, pctSeams=0.50, redSeams=True))
        print("done res_bench_forward_rem_red")
    except: print('Error generating res_bench_forward_rem_red')

    try:
        cv2.imwrite("images/results/res_car_back_ins.jpg",
                    car_backward_insert(car, pctSeams=0.50))
        print("done res_car_back_ins")
    except: print('Error generating res_car_back_ins')

    try:
        cv2.imwrite("images/results/res_car_forward_ins.jpg",
                    car_forward_insert(car, pctSeams=0.50))
        print("done res_car_forward_ins")
    except: print('Error generating res_car_forward_ins')

    print('result images generated')
    return
    
def generate_difference_images():
    results = ["res_beach_back_rem.jpg",
               "res_dolphin_back_ins.jpg",
               "res_dolphin_back_5050.jpg",
               "res_bench_back_rem.jpg",
               "res_bench_forward_rem.jpg",
               "res_car_back_ins.jpg",
               "res_car_forward_ins.jpg"]
    
    comps  =  ["comp_beach_back_rem.png",
               "comp_dolphin_back_ins.png",
               "comp_dolphin_back_5050.png",
               "comp_bench_back_rem.png",
               "comp_bench_forward_rem.png",
               "comp_car_back_ins.png",
               "comp_car_forward_ins.png"]
    
    diffs  =  ["diff_beach_back_rem.jpg",
               "diff_dolphin_back_ins.jpg",
               "diff_dolphin_back_5050.jpg",
               "diff_bench_back_rem.jpg",
               "diff_bench_forward_rem.jpg",
               "diff_car_back_ins.jpg",
               "diff_car_forward_ins.jpg"]


    for i in range(len(results)):
        result = cv2.imread("images/results/" + results[i])
        comp = cv2.imread("images/comparison/" + comps[i])
        diff = difference_image(result, comp)
        cv2.imwrite("images/results/" + diffs[i], diff)
        similar_percent, energy_percent = numerical_comparison(result, comp)
        print("{} similar_percent {} energy_percent {}".format(results[i], similar_percent, energy_percent))
            
    print('difference images completed')
    return
    

if __name__ == "__main__":
    """ Generate the 10 results and 7 diff images
    """
    # make the images/results folder
    output_dir = os.path.join(OUT_FOLDER)

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print("Processing files...")

    start_time = time.time()
    temp_time = time.time()
    beach, dolphin, bench, car = generate_base_arrays()
    print("---generate_base_arrays %s seconds ---" % (time.time() - temp_time))
    temp_time = time.time()

    generate_result_images(beach, dolphin, bench, car)
    print("---generate_result_images %s seconds ---" % (time.time() - temp_time))
    temp_time = time.time()

    generate_difference_images()
    print("---generate_result_images %s seconds ---" % (time.time() - temp_time))
    print("---total %s seconds ---" % (time.time() - start_time))

    # numerical_comparison is not tested here.
    
    

        


