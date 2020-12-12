from modules.IterativeHistogram import IterativeHistogram
from matplotlib.patches import ConnectionPatch
from modules.align import align_cv2
from matplotlib import pyplot
from random import randint
from scipy import signal
import os.path
import numpy
import math
import sys
import cv2


def get_sortable_value_from_path(path):
    value = int(path.split(".")[0].split("_")[-1])

    return value


def expand_mask(mask, radius):
    expanded_mask = numpy.ones(mask.shape, dtype=mask.dtype)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == False:
                i_min = max(0,i-radius)
                i_max = min(mask.shape[0]-1,i+radius)

                j_min = max(0,j-radius)
                j_max = min(mask.shape[1]-1,j+radius)

                expanded_mask[i_min:i_max,j_min:j_max] = False

    return expanded_mask


def main():
    project_directory = os.path.dirname(__file__)
    data_directory = os.path.join(project_directory, "data")
    test_data_paths = sorted(os.listdir(data_directory), key=lambda x: get_sortable_value_from_path(x))

    raw_feature_mask_path = os.path.join(project_directory, "feature_mask_raw.npy")

    raw_feature_mask = numpy.invert(numpy.load(raw_feature_mask_path)).astype(numpy.uint8)
    feature_mask = expand_mask(raw_feature_mask, 20)

    pyplot.imshow(feature_mask)
    pyplot.show()
    pyplot.close()

    detector = cv2.FastFeatureDetector_create()

    print("threshold: ", detector.getThreshold())
    print("type: ", detector.getType())

    detector.setType(1)
    detector.setThreshold(10)

    # axes = pyplot.axes()

    # brief = cv2.xfeatures2d.brief

    for _ in range(100):
        i = randint(0,len(test_data_paths)-2)
        # i=0

        path_a = test_data_paths[i]
        path_b = test_data_paths[i + 1]

        absolute_path_a = os.path.join(data_directory, path_a)
        absolute_path_b = os.path.join(data_directory, path_b)

        print(path_a, path_b)

        image_a = cv2.imread(absolute_path_a)
        image_b = cv2.imread(absolute_path_b)

        grayscale_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        grayscale_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

        features_a = detector.detect(grayscale_a, feature_mask)
        features_b = detector.detect(grayscale_b, feature_mask)



        print(len(features_a))

        # axes.imshow(image_a)

        # for feature in features_a:
            # print(type(feature))
            # print(feature.pt)
            # print(feature.response)

            # if feature.response < 60:
            #     axes.plot(feature.pt, marker='.', mfc="red")
            #     axes.text(feature.pt[0], feature.pt[1], str(feature.response), color="red")

        output = numpy.empty((image_a.shape[0], image_a.shape[1], 3), dtype=numpy.uint8)
        cv2.drawKeypoints(grayscale_a, keypoints=features_a, outImage=output)
        cv2.imshow("features", output)
        cv2.waitKey()

        # pyplot.show()
        # pyplot.close()


if __name__ == "__main__":
    main()
