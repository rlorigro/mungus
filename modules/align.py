from modules.IterativeHistogram import IterativeHistogram
from matplotlib.patches import ConnectionPatch
from collections import defaultdict
from matplotlib import pyplot
from random import randint
from random import shuffle
from scipy import signal
import os.path
import numpy
import math
import sys
import cv2


# BRIEF: Binary Robust Independent Elementary Features
# Michael Calonder, Vincent Lepetit, Christoph Strecha, and Pascal Fua
class BriefAligner:
    def __init__(self, feature_mask_path, smoothing_radius, kernel_radius, n_samples_per_kernel, n_samples_per_image):
        self.radius = kernel_radius
        self.kernel_size = 2*kernel_radius + 1
        self.n_samples_per_kernel = n_samples_per_kernel
        self.n_samples_per_image = n_samples_per_image

        self.smoothing_radius = smoothing_radius
        self.smoothing_kernel = None

        if smoothing_radius is not None:
            self.generate_gaussian_kernel(kernel_size=20)

        self.keypoint_detector = cv2.FastFeatureDetector_create()
        print("threshold: ", self.keypoint_detector.getThreshold())
        print("type: ", self.keypoint_detector.getType())
        self.keypoint_detector.setType(1)
        self.keypoint_detector.setThreshold(10)

        self.feature_mask = None
        self.load_feature_mask(path=feature_mask_path, radius=kernel_radius)

        self.kernel_a = self.generate_boolean_kernel(kernel_size=self.kernel_size, n_samples=n_samples_per_kernel)
        self.kernel_b = self.generate_boolean_kernel(kernel_size=self.kernel_size, n_samples=n_samples_per_kernel)

    def load_feature_mask(self, path, radius):
        mask = numpy.invert(numpy.load(path)).astype(numpy.uint8)

        self.feature_mask = numpy.ones(mask.shape, dtype=mask.dtype)

        x_size, y_size = mask.shape

        mask[0:1, :] = 0
        mask[(x_size - 1):x_size, :] = 0
        mask[:, 0:1] = 0
        mask[:, (y_size - 1):y_size] = 0

        # Expand the mask by at least the radius of the feature so that no feature encounters an edge or empty value
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == False:
                    i_min = max(0, i - radius)
                    i_max = min(mask.shape[0], i + radius)

                    j_min = max(0, j - radius)
                    j_max = min(mask.shape[1], j + radius)

                    self.feature_mask[i_min:i_max, j_min:j_max] = False

        return self.feature_mask

    # from:
    # https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_image_blur.html
    def generate_gaussian_kernel(self, kernel_size):
        # First a 1-D  Gaussian
        t = numpy.linspace(-10, 10, kernel_size)
        bump = numpy.exp(-0.1 * t ** 2)
        bump /= numpy.trapz(bump)  # normalize the integral to 1

        # make a 2-D kernel out of it
        self.smoothing_kernel = bump[:, numpy.newaxis] * bump[numpy.newaxis, :]

        return

    @staticmethod
    def generate_boolean_kernel(kernel_size, n_samples):
        mask = numpy.zeros(shape=[kernel_size, kernel_size], dtype=numpy.bool)

        sites = defaultdict(set)

        for i in range(n_samples):
            while True:
                x = randint(0, kernel_size - 1)
                y = randint(0, kernel_size - 1)

                if y not in sites[x]:
                    sites[x].add(y)
                    break

            mask[x, y] = True

        return mask

    def extract_features(self, image):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.smoothing_radius is not None:
            grayscale_image = signal.oaconvolve(grayscale_image, self.smoothing_kernel, mode="same")

        print(grayscale_image.shape, self.feature_mask.shape)
        print(grayscale_image.dtype, self.feature_mask.dtype)

        keypoints = self.keypoint_detector.detect(grayscale_image, self.feature_mask)

        # Take a subset of the key points in random order
        shuffle(keypoints)
        keypoints = keypoints[0:min(len(keypoints), self.n_samples_per_image)]

        x_length = self.kernel_a.shape[0]
        y_length = self.kernel_a.shape[1]

        if x_length % 2 != 1:
            exit("ERROR: even kernel dimension cannot be centered")

        if y_length % 2 != 1:
            exit("ERROR: even kernel dimension cannot be centered")

        x_radius = int((x_length - 1) / 2)
        y_radius = int((y_length - 1) / 2)

        features = numpy.zeros([self.n_samples_per_image, self.n_samples_per_kernel])

        i = 0
        for n in range(self.n_samples_per_image):
            x = int(keypoints[i].pt[0])
            y = int(keypoints[i].pt[1])

            i += 1

            if i == len(keypoints):
                break

            a = grayscale_image[y - y_radius:y + y_radius + 1, x - x_radius:x + x_radius + 1]
            b = grayscale_image[y - y_radius:y + y_radius + 1, x - x_radius:x + x_radius + 1]

            # fig, axes = pyplot.subplots(ncols=2)
            # axes[0].imshow(grayscale_image)
            # r = pyplot.Rectangle(xy=(x-x_radius,y-y_radius), width=x_radius*2+1, height=y_radius*2+1, linewidth=1,edgecolor='r',facecolor='none')
            # axes[0].add_patch(r)
            # axes[1].imshow(a)
            # pyplot.show()
            # pyplot.close()

            # print(a.shape)
            # print(b.shape)
            # print(self.kernel_a.shape)
            # print(self.kernel_b.shape)

            a = a[self.kernel_a]
            b = b[self.kernel_b]

            features[n] = (a > b)

        return keypoints, features

    def compute_shift(self, image_shape, features_a, features_b, keypoints_a, keypoints_b,
                      axes_a=None, axes_b=None, axes_x_shift=None, axes_y_shift=None):

        ambiguity = numpy.zeros(self.n_samples_per_image, dtype=numpy.int)
        pairs = numpy.zeros(self.n_samples_per_image, dtype=numpy.int)

        n_tests = 200

        print(len(keypoints_a), len(keypoints_b))

        i_a = 0
        n_attempted = 0
        while i_a < n_tests:

            # There might not be enough good features to satisfy the desired number of tests per image
            n_attempted += 1
            if n_attempted == len(features_a):
                break

            min_a = sys.maxsize

            # # Attempt to skip uniform patches
            # n_nonzero = numpy.count_nonzero(features_a[i_a])
            #
            # if n_nonzero == 0:
            #     pairs[i_a] = -1
            #     continue

            for i_b in range(len(features_b)):
                hamming_distance = numpy.count_nonzero(features_a[i_a] != features_b[i_b])

                if hamming_distance < min_a:
                    pairs[i_a] = i_b
                    min_a = hamming_distance

                elif hamming_distance == min_a:
                    ambiguity[i_a] += 1

            i_a += 1

        x_size = image_shape[1]
        y_size = image_shape[0]

        x_shifts = numpy.zeros(2*x_size + 1)
        y_shifts = numpy.zeros(2*y_size + 1)

        x_shift_modal_frequency = 0
        y_shift_modal_frequency = 0
        x_shift_mode = 0
        y_shift_mode = 0

        for i_a in range(n_tests):
            i_b = pairs[i_a]

            if i_b < 0:
                print("skipping")
                continue

            if ambiguity[i_a] > 1:
                continue

            y_a = int(keypoints_a[i_a].pt[1])
            x_a = int(keypoints_a[i_a].pt[0])

            y_b = int(keypoints_b[i_b].pt[1])
            x_b = int(keypoints_b[i_b].pt[0])

            # print(y_a, x_a, y_b, x_b)

            x_shift_i = x_b - x_a
            y_shift_i = y_b - y_a

            x_shifts[x_shift_i+x_size] += 1
            y_shifts[y_shift_i+y_size] += 1

            if x_shifts[x_shift_i+x_size] > x_shift_modal_frequency:
                x_shift_mode = x_shift_i
                x_shift_modal_frequency = x_shifts[x_shift_i+x_size]

            if y_shifts[y_shift_i+y_size] > y_shift_modal_frequency:
                y_shift_mode = y_shift_i
                y_shift_modal_frequency = y_shifts[y_shift_i+y_size]

            if axes_a is not None and axes_b is not None:
                linestyle = "solid"
                con = ConnectionPatch(xyA=(x_a, y_a), xyB=(x_b, y_b),
                                      coordsA="data", coordsB="data",
                                      axesA=axes_a, axesB=axes_b, color="red", linestyle=linestyle)

                axes_b.text(x_b, y_b, str(ambiguity[i_a]), bbox=dict(facecolor='white', edgecolor='none', pad=0))
                axes_b.add_artist(con)

        x_shift = x_shift_mode
        y_shift = y_shift_mode

        if axes_x_shift is not None:
            x_range = numpy.arange(-x_size,x_size+1,1)
            axes_x_shift.plot(x_range, x_shifts)
            axes_x_shift.set_xlim([-x_size, x_size])

        if axes_y_shift is not None:
            y_range = numpy.arange(-y_size,y_size+1,1)
            axes_y_shift.plot(y_range, y_shifts)
            axes_y_shift.set_xlim([-y_size, y_size])

        return x_shift, y_shift

    def align(self, image_a, image_b, axes_a=None, axes_b=None, axes_x_shift=None, axes_y_shift=None):

        # fig, axes = pyplot.subplots(nrows=2, sharex=True, sharey=True)
        # axes[0].imshow(grayscale_a)
        # axes[1].imshow(grayscale_a_smoothed)
        # pyplot.show()
        # pyplot.close()

        keypoints_a, features_a = self.extract_features(image=image_a)
        keypoints_b, features_b = self.extract_features(image=image_b)

        x_shift, y_shift = self.compute_shift(
            image_shape=image_a.shape,
            features_a=features_a,
            features_b=features_b,
            keypoints_a=keypoints_a,
            keypoints_b=keypoints_b,
            axes_a=axes_a,
            axes_b=axes_b,
            axes_x_shift=axes_x_shift,
            axes_y_shift=axes_y_shift)

        return x_shift, y_shift


# NOT working for Among us
def align_cv2(grayscale_a, grayscale_b, mask):

    max_iterations = 500
    terminal_eps = 1e-4

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, terminal_eps)

    warp_matrix = numpy.eye(2, 3, dtype=numpy.float32)
    (c, warp_matrix) = cv2.findTransformECC(grayscale_a, grayscale_b, warp_matrix, cv2.MOTION_TRANSLATION, criteria, mask, 1)

    # print(warp_matrix.get())

    # aligned_b = cv2.warpAffine(
    #     cv_image_b,
    #     warp_matrix,
    #     (game_window.width, game_window.height),
    #     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # pyplot.imshow(aligned_b.get())

    return warp_matrix