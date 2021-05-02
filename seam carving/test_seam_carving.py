import cv2
import numpy as np
import scipy as sp
import unittest

import seam_carving

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


class MyTestCase(unittest.TestCase):
    def setUp(self):
        print("Processing files...")
        self.beach, self.dolphin, self.bench, self.car = generate_base_arrays()

    """
    test energy map
    """

    def test_calc_energy_map_1(self):
        image = np.array([[1, 1, 1], [2, 2, 2],
                          [3, 3, 3], [4, 4, 4]]).reshape((2, 2, 3)).astype(np.uint8)
        energy_map_1 = seam_carving.calc_energy_map(image)
        energy_map_2 = seam_carving.calc_energy_map_2(image)
        energy_map_3 = seam_carving.calc_energy_map_3(image)
        energy_map_4 = seam_carving.calc_energy_map_4(image)
        print(energy_map_1)
        print(energy_map_2)
        print(energy_map_3)
        print(energy_map_4)

    def test_calc_energy_map_2(self):
        image = np.array([(255, 101, 51), (255, 101, 153), (255, 101, 255),
                          (255, 153, 51), (255, 153, 153), (255, 153, 255),
                          (255, 203, 51), (255, 204, 153), (255, 205, 255),
                          (255, 255, 51), (255, 255, 153), (255, 255, 255)]).reshape((4, 3, 3)).astype(np.uint8)
        energy_map_1 = seam_carving.calc_energy_map(image)
        energy_map_2 = seam_carving.calc_energy_map_2(image)
        energy_map_3 = seam_carving.calc_energy_map_3(image)
        energy_map_4 = seam_carving.calc_energy_map_4(image)
        print(energy_map_1)
        print(energy_map_2)
        print(energy_map_3)
        print(energy_map_4)

    """
    test find path
    """

    def test_backtrack(self):
        ENERGIES = np.array([
            [9, 9, 0, 9, 9],
            [9, 1, 9, 8, 9],
            [9, 9, 9, 9, 0],
            [9, 9, 9, 0, 9],
        ])
        cumulative_map, backtrack = seam_carving.cumulative_map_backward(ENERGIES)
        path = seam_carving.get_path(cumulative_map, backtrack)
        print(path)
        print(cumulative_map)

    def test_get_path_mask(self):
        ENERGIES = np.array([
            [9, 9, 0, 9, 9],
            [9, 1, 9, 8, 9],
            [9, 9, 9, 9, 0],
            [9, 9, 9, 0, 9],
        ])
        cumulative_map, backtrack = seam_carving.cumulative_map_backward(ENERGIES)
        path = seam_carving.get_path(cumulative_map, backtrack)
        mask = seam_carving.get_path_mask(ENERGIES.shape, path)
        print(mask)

    def test_get_path_mask_with_image(self):
        energy_map = seam_carving.calc_energy_map(self.bench)
        cumulative_map, backtrack = seam_carving.cumulative_map_backward(energy_map)
        path = seam_carving.get_path(cumulative_map, backtrack)
        mask = seam_carving.get_path_mask(energy_map.shape, path)
        cv2.imwrite("path_mask.png", mask.astype(np.uint8) * 255)

    def test_seam_remove_1(self):
        new_image, path = seam_carving.seam_remove_1_backward(self.bench)
        cv2.imwrite("seam_remove_1.png", new_image)

    def test_seam_remove1_red(self):
        new_image, inserted_image = seam_carving.seams_removal_back(self.bench, 1, red_seams=True)
        cv2.imwrite("seam_remove_1_red.png", inserted_image)

    def test_insert_red(self):
        image = np.array([(255, 101, 51), (255, 101, 153), (255, 101, 255),
                          (255, 153, 51), (255, 153, 153), (255, 153, 255),
                          (255, 203, 51), (255, 204, 153), (255, 205, 255),
                          (255, 255, 51), (255, 255, 153), (255, 255, 255)]).reshape((4, 3, 3)).astype(np.uint8)
        new_image, inserted_image = seam_carving.seams_removal_back(image, 1, red_seams=True)
        print(inserted_image)

    """
    test with bench and beach
    """

    def test_bench_backward_removal(self):
        r, c, ch = self.bench.shape
        num_pixel = int(c * 0.5)
        removed_image, inserted_image_red = seam_carving.seams_removal_back(self.bench, num_pixel, True)
        cv2.imwrite("images/results/res_bench_back_rem.png", removed_image)
        cv2.imwrite("images/results/res_bench_back_rem_red.png", inserted_image_red)

    def test_bench_forward_removal(self):
        r, c, ch = self.bench.shape
        num_pixel = int(c * 0.5)
        removed_image, inserted_image_red = seam_carving.seams_removal_forward(self.bench, num_pixel, True)
        cv2.imwrite("images/results/res_bench_forward_rem.jpg", removed_image)
        cv2.imwrite("images/results/res_bench_forward_rem_red.jpg", inserted_image_red)

    def display_energy_maps(self, image, name):
        energy_map = seam_carving.calc_energy_map(image)
        energy_map_image = np.zeros_like(energy_map)
        cv2.normalize(energy_map, energy_map_image, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(name + "_back_energy_map.png", energy_map_image)

        cumulative_map_back, backtrack = seam_carving.cumulative_map_backward(energy_map)
        cumulative_map_back_image = np.zeros_like(cumulative_map_back)
        cv2.normalize(cumulative_map_back, cumulative_map_back_image, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(name + "_back_cumulative_map.png", cumulative_map_back_image)

        kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)
        matrix_x = seam_carving.calc_neighbor_grads(image, kernel_x)
        matrix_y_left = seam_carving.calc_neighbor_grads(image, kernel_y_left)
        matrix_y_right = seam_carving.calc_neighbor_grads(image, kernel_y_right)

        cumulative_map_forward, backtrack = seam_carving.cumulative_map_forward(energy_map, matrix_x, matrix_y_left,
                                                                                matrix_y_right)
        cumulative_map_forward_image = np.zeros_like(cumulative_map_forward)
        cv2.normalize(cumulative_map_forward, cumulative_map_forward_image, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(name + "_forward_cumulative_map.png", cumulative_map_forward_image)

    def test_bench_energy_map(self):
        self.display_energy_maps(self.bench, "bench")

    def test_beach_backward_removal(self):
        r, c, ch = self.beach.shape
        num_pixel = int(c * 0.5)
        new_image, inserted_image = seam_carving.seams_removal_back(self.beach, num_pixel, True)
        cv2.imwrite("images/results/res_beach_back_rem.png", new_image)
        cv2.imwrite("images/results/res_beach_back_rem_red.png", inserted_image)

    def test_beach_energy_map(self):
        self.display_energy_maps(self.beach, "beach")

    def test_dolphin_energy_map(self):
        self.display_energy_maps(self.dolphin, "dolphin")

    def test_car_energy_map(self):
        self.display_energy_maps(self.car, "car")

    def test_removal(self):
        # Some matrix
        a = np.arange(24).reshape(6, 4)
        print('Orignial matrix: \n', a)

        # Some random columns to remove, one per row.
        remove = np.random.randint(4, size=6)
        print('Remove: \n', remove[..., None])
        remove += 4 * np.arange(6)  # <=== Hey look!

        b = np.delete(a.flatten(), remove).reshape(6, 3)
        print('One element missing: \n', b)

    def test_insert(self):
        # Some matrix
        a = np.arange(12).reshape(3, 4)
        print('Orignial matrix: \n', a)

        path = np.array([1, 2, 0])
        print('insert: \n', path[..., None])
        path += 4 * np.arange(3)
        print('insert after: \n', path[..., None])

        b = np.insert(a.flatten(), path, 255).reshape(3, 5)
        print('One element added: \n', b)

    """
    insert part
    """

    def test_update_path_record(self):
        record = [np.array([1, 2, 3, 4, 5]),
                  np.array([5, 4, 3, 2, 1]),
                  np.array([2, 2, 2, 2, 2]),
                  np.array([3, 3, 4, 3, 3])]
        current = np.array([3, 3, 3, 3, 3])

        record_1 = seam_carving.update_path_record_2(record, current)
        print(record_1)
        print(record)

        record = [np.array([1, 2, 3, 4, 5]),
                  np.array([5, 4, 3, 2, 1]),
                  np.array([2, 2, 2, 2, 2]),
                  np.array([3, 3, 4, 3, 3])]
        current = np.array([3, 3, 3, 3, 3])
        record_2 = seam_carving.update_path_record(record, current)
        print(record_2)
        print(record)

    def test_dolphin_backward_insert(self):
        r, c, ch = self.dolphin.shape
        num_pixel = int(c * 0.5)
        inserted_image, inserted_image_red = seam_carving.seams_insertion_back(self.dolphin, num_pixel, True)
        cv2.imwrite("images/results/res_dolphin_back_ins.png", inserted_image)
        cv2.imwrite("images/results/res_dolphin_back_ins_red.png", inserted_image_red)

    def test_dolphin_backward_5050(self):
        dolphin_array = cv2.imread("images/results/res_dolphin_back_ins.jpg")
        array_5050 = seam_carving.dolphin_backward_5050(dolphin_array, pctSeams=0.50)
        cv2.imwrite("images/results/res_dolphin_back_5050.jpg", array_5050)

    def test_car_back(self):
        r, c, ch = self.car.shape
        num_pixel = int(c * 0.5)
        inserted_image, inserted_image_red = seam_carving.seams_insertion_back(self.car, num_pixel, True)
        cv2.imwrite("images/results/res_car_back_ins.jpg", inserted_image)
        cv2.imwrite("images/results/res_car_back_ins_red.jpg", inserted_image_red)

    def test_car_forward(self):
        r, c, ch = self.car.shape
        num_pixel = int(c * 0.5)
        inserted_image, inserted_image_red = seam_carving.seams_insertion_forward(self.car, num_pixel, True)
        cv2.imwrite("images/results/res_car_forward_ins.jpg", inserted_image)
        cv2.imwrite("images/results/res_car_forward_ins_red.jpg", inserted_image_red)

    def test_diff(self):
        result = cv2.imread("images/results/res_dolphin_back_5050.jpg")
        comp = cv2.imread("images/comparison/comp_dolphin_back_5050.png")
        diff = seam_carving.difference_image(result, comp)
        cv2.imwrite("images/results/diff_dolphin_back_5050.jpg", diff)



if __name__ == '__main__':
    unittest.main()
