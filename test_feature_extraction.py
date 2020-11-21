from matplotlib.patches import ConnectionPatch
from modules.BRIEF import extract_features
from modules.BRIEF import generate_kernel
from matplotlib import pyplot
from random import randint
from scipy import signal
import os.path
import numpy
import sys
import cv2


# from:
# https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_image_blur.html
def generate_gaussian_kernel(kernel_size):
    # First a 1-D  Gaussian
    t = numpy.linspace(-10, 10, kernel_size)
    bump = numpy.exp(-0.1 * t ** 2)
    bump /= numpy.trapz(bump)  # normalize the integral to 1

    # make a 2-D kernel out of it
    kernel = bump[:, numpy.newaxis] * bump[numpy.newaxis, :]

    return kernel


def main():
    project_directory = os.path.dirname(__file__)
    data_directory = os.path.join(project_directory, "data")
    test_data_paths = os.listdir(data_directory)

    feature_mask_path = os.path.join(project_directory, "feature_mask_90.npy")
    feature_mask = numpy.load(feature_mask_path)

    radius = 90
    kernel_size = radius*2 + 1
    n_samples_per_kernel = 160

    kernel_a = generate_kernel(size=kernel_size, n_samples=n_samples_per_kernel)
    kernel_b = generate_kernel(size=kernel_size, n_samples=n_samples_per_kernel)

    # Make sure there's no leftover valid spots near the borders of the image
    x_size = feature_mask.shape[0]
    y_size = feature_mask.shape[1]
    feature_mask[0:radius+1,:] = 1
    feature_mask[x_size-radius-1:x_size:] = 1
    feature_mask[:,0:radius+1] = 1
    feature_mask[:,y_size-radius-1:y_size] = 1

    # fig, axes = pyplot.subplots(ncols=1,nrows=2)
    # axes[0].imshow(kernel_a)
    # axes[1].imshow(kernel_b)
    # pyplot.show()
    # pyplot.close()

    feature_mask_coordinates = numpy.nonzero(1 - feature_mask)

    n_samples_per_image = 1200

    smoothing_kernel = generate_gaussian_kernel(kernel_size=20)

    for _ in range(100):
        i = randint(0,len(test_data_paths))

        path_a = test_data_paths[i]
        path_b = test_data_paths[i + 1]

        absolute_path_a = os.path.join(data_directory, path_a)
        absolute_path_b = os.path.join(data_directory, path_b)

        image_a = cv2.imread(absolute_path_a)
        image_b = cv2.imread(absolute_path_b)

        grayscale_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        grayscale_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

        grayscale_a_smoothed = signal.oaconvolve(grayscale_a, smoothing_kernel, mode="same")
        grayscale_b_smoothed = signal.oaconvolve(grayscale_b, smoothing_kernel, mode="same")

        # fig, axes = pyplot.subplots(nrows=2, sharex=True, sharey=True)
        # axes[0].imshow(grayscale_a)
        # axes[1].imshow(grayscale_a_smoothed)
        # pyplot.show()
        # pyplot.close()

        coord_indexes_a, features_a = extract_features(
            grayscale_image=grayscale_a_smoothed,
            feature_mask_coordinates=feature_mask_coordinates,
            kernel_a=kernel_a,
            kernel_b=kernel_b,
            n_samples_per_image=n_samples_per_image)

        coord_indexes_b, features_b = extract_features(
            grayscale_image=grayscale_b_smoothed,
            feature_mask_coordinates=feature_mask_coordinates,
            kernel_a=kernel_a,
            kernel_b=kernel_b,
            n_samples_per_image=n_samples_per_image)

        ambiguity = numpy.zeros(n_samples_per_image, dtype=numpy.int)
        pairs = numpy.zeros(n_samples_per_image, dtype=numpy.int)

        n_tests = 50

        for i_a in range(n_tests):
            min_a = sys.maxsize

            n_nonzero = numpy.count_nonzero(features_a[i_a])
            # print(n_nonzero)
            # print(features_a[i_a].astype(numpy.int))

            if (n_nonzero == 0):
                pairs[i_a] = -1
                continue

            for i_b in range(len(features_b)):
                hamming_distance = numpy.count_nonzero(features_a[i_a] != features_b[i_b])

                if hamming_distance < min_a:
                    pairs[i_a] = i_b
                    min_a = hamming_distance

                elif hamming_distance == min_a:
                    ambiguity[i_a] += 1

        fig, axes = pyplot.subplots(ncols=1, nrows=2)
        # image_a[feature_mask] = 0
        # image_b[feature_mask] = 0
        axes[0].imshow(image_a)
        axes[1].imshow(image_b)

        x_shift = None
        y_shift = None

        for i_a in range(n_tests):
            i_b = pairs[i_a]

            if i_b < 0:
                print("skipping")
                continue

            # print(features_a[i_a])
            # print(features_b[i_b])

            y_a = feature_mask_coordinates[0][coord_indexes_a[i_a]]
            x_a = feature_mask_coordinates[1][coord_indexes_a[i_a]]

            y_b = feature_mask_coordinates[0][coord_indexes_b[i_b]]
            x_b = feature_mask_coordinates[1][coord_indexes_b[i_b]]

            x_shift = x_b - x_a
            y_shift = y_b - y_a

            linestyle = "solid"

            if ambiguity[i_a] > 1:
                linestyle = (0, (1,10))

            con = ConnectionPatch(xyA=(x_a, y_a), xyB=(x_b, y_b),
                                  coordsA="data", coordsB="data",
                                  axesA=axes[0], axesB=axes[1], color="red", linestyle=linestyle)

            axes[1].text(x_b, y_b, str(ambiguity[i_a]), bbox=dict(facecolor='white', edgecolor='none', pad=0))
            axes[1].add_artist(con)

        x_shift /= n_tests
        y_shift /= n_tests

        pyplot.show()
        pyplot.close()


if __name__ == "__main__":
    main()
