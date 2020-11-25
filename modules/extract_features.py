from scipy import signal
import numpy
import cv2
import os


SCHARR_KERNEL = numpy.array([[ -3-3j, 0-10j,  +3 -3j],
                             [-10+0j, 0+ 0j, +10 +0j],
                             [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy


def extract_features(grayscale_image, kernel):
    convolved = signal.convolve2d(grayscale_image, kernel, mode="valid")

    cv2.imshow("B", numpy.absolute(convolved))
    cv2.waitKey()


if __name__ == "__main__":
    project_directory = os.path.dirname(__file__)
    data_directory = os.path.join(project_directory, "data")
    test_data_paths = os.listdir(data_directory)

    for i in range(0,len(test_data_paths)):
        path_a = test_data_paths[i]

        absolute_path_a = os.path.join(data_directory, path_a)

        image_a = cv2.imread(absolute_path_a)
        grayscale_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)

        extract_features(grayscale_a, SCHARR_KERNEL)


