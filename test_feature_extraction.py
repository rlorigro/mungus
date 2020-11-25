from modules.IterativeHistogram import IterativeHistogram
from matplotlib.patches import ConnectionPatch
from modules.align import BriefAligner
from matplotlib import pyplot
from random import randint
from scipy import signal
import os.path
import numpy
import math
import sys
import cv2


def main():
    project_directory = os.path.dirname(__file__)
    data_directory = os.path.join(project_directory, "data")
    test_data_paths = os.listdir(data_directory)

    feature_mask_path = os.path.join(project_directory, "feature_mask_90.npy")
    raw_feature_mask_path = os.path.join(project_directory, "feature_mask_raw.npy")

    raw_feature_mask = numpy.load(raw_feature_mask_path)
    raw_feature_mask_inverse = numpy.invert(raw_feature_mask)

    aligner = BriefAligner(
        feature_mask_path=feature_mask_path,
        smoothing_radius=20,
        kernel_radius=90,
        n_samples_per_kernel=160,
        n_samples_per_image=1000)

    for _ in range(100):
        i = randint(0,len(test_data_paths)-2)

        path_a = test_data_paths[i]
        path_b = test_data_paths[i + 1]

        absolute_path_a = os.path.join(data_directory, path_a)
        absolute_path_b = os.path.join(data_directory, path_b)

        print(path_a, path_b)

        image_a = cv2.imread(absolute_path_a)
        image_b = cv2.imread(absolute_path_b)

        # fig = pyplot.figure()
        # gs = fig.add_gridspec(ncols=4, nrows=2)
        # axes0 = fig.add_subplot(gs[0, 0])
        # axes1 = fig.add_subplot(gs[1, 0])
        # axes2 = fig.add_subplot(gs[0, 1])
        # axes3 = fig.add_subplot(gs[1, 1])
        # axes4 = fig.add_subplot(gs[:, 2:])
        #
        # axes2.set_title("x_shift histogram")
        # axes3.set_title("y_shift histogram")

        # axes0.imshow(image_a)
        # axes1.imshow(image_b)

        x_shift, y_shift = aligner.align(
            image_a=image_a,
            image_b=image_b)
            # axes_a=axes0,
            # axes_b=axes1,
            # axes_x_shift=axes2,
            # axes_y_shift=axes3)

        print(x_shift)
        print(y_shift)

        x_size = image_a.shape[1]
        y_size = image_a.shape[0]

        x_a_start = max(0, x_shift)
        y_a_start = max(0, y_shift)

        x_b_start = max(0, -x_shift)
        y_b_start = max(0, -y_shift)

        x_a_stop = x_a_start + x_size
        y_a_stop = y_a_start + y_size

        x_b_stop = x_b_start + x_size
        y_b_stop = y_b_start + y_size

        # print(image_a.shape)
        # print(x_a_start)
        # print(y_a_start)
        # print(x_b_start)
        # print(y_b_start)
        # print(x_a_stop)
        # print(y_a_stop)
        # print(x_b_stop)
        # print(y_b_stop)

        stitched_shape = list(image_a.shape)

        stitched_shape[0] += abs(y_shift)
        stitched_shape[1] += abs(x_shift)

        stitched_image_a = numpy.zeros(stitched_shape, dtype=image_a.dtype)
        stitched_image_b = numpy.zeros(stitched_shape, dtype=image_a.dtype)

        stitched_image_a[y_a_start:y_a_stop, x_a_start:x_a_stop] = image_a
        stitched_image_b[y_b_start:y_b_stop, x_b_start:x_b_stop] = image_b

        stitched_image = cv2.addWeighted(stitched_image_a,0.5,stitched_image_b,0.5,0)

        # axes4.imshow(stitched_image)
        #
        # pyplot.show()
        # pyplot.close()


if __name__ == "__main__":
    main()
