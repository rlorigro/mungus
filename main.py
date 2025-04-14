from modules.align import BriefAligner

from matplotlib import pyplot
import matplotlib

from datetime import datetime
from scipy import signal
from time import sleep
import pywinctl as pwc
import os.path
from PIL.ImageGrab import grab
import numpy
import cv2


def run_screencapture_loop(window_name_query: str):
    candidate_windows = pwc.getAllWindows()
    target_window = None

    for window in candidate_windows:
        if window_name_query in window.title:
            target_window = window

    if target_window is None:
        exit("ERROR: no window with %s title found" % window_name_query)

    # axis = pyplot.axes()
    # pyplot.ion()
    # pyplot.show(block=False)

    i=0
    while (True):
        if not target_window.isActive:
            # target_window.activate()
            print("Waiting for window to become active...")
            sleep(1)
            continue

        window_region = (target_window.left + 10, target_window.top + 96, target_window.right - 20, target_window.bottom - 42)
        print(window_region)

        image = grab(window_region)
        image.save("screenshot_" + str(i) + ".png")

        i = i + 1

        print(image.size)

        # axis.imshow(image)
        # pyplot.draw()
        sleep(0.05)


def run_alignment_loop(window_name_query, aligner, stitch_mask_path=None):
    stitch_mask = None

    if stitch_mask_path is not None:
        stitch_mask = numpy.invert(numpy.load(stitch_mask_path))

    candidate_windows = pwc.getAllWindows()
    target_window = None

    for window in candidate_windows:
        if window_name_query in window.title:
            target_window = window

    if target_window is None:
        exit("ERROR: no window with %s title found" % window_name_query)

    fig = pyplot.figure(figsize=[14,9])
    axes = pyplot.axes()

    gs = fig.add_gridspec(ncols=4, nrows=2)
    axes_0 = fig.add_subplot(gs[0, 0])
    axes_1 = fig.add_subplot(gs[1, 0])
    axes_2 = fig.add_subplot(gs[0, 1])
    axes_3 = fig.add_subplot(gs[1, 1])
    axes_4 = fig.add_subplot(gs[:, 2:])
    axes_2.set_title("x_shift histogram")
    axes_3.set_title("y_shift histogram")

    # An empty array big enough to accommodate the entire map
    collage = numpy.zeros([6_000,12_000,3],dtype=numpy.uint8)

    # Pretend we are starting on the Skeld (top middle)
    x_prev = 6000 + target_window.width
    y_prev = 1000 + target_window.height

    prev_image = None
    prev_features = None
    prev_keypoints = None
    i = 0
    while (True):
        start = datetime.now()

        if not target_window.isActive:
            print("Waiting for window to become active... use keyboard interrupt (CTRL-C) to save stitched image and exit")
            prev_image = None

            try:
                sleep(1)
            except KeyboardInterrupt:
                cv2.imwrite("test.png", numpy.flip(collage, axis=2))
                exit()

            continue

        # axes.clear()
        # axes_0.clear()
        # axes_1.clear()
        # axes_2.clear()
        # axes_3.clear()
        # axes_4.clear()

        window_region = (target_window.left + 10, target_window.top + 96, target_window.right - 20, target_window.bottom - 42)
        print(window_region)

        image = numpy.uint8(grab(window_region))

        keypoints, features = aligner.extract_features(image=image)

        features = list(features)
        keypoints = list(keypoints)

        if prev_image is not None:
            axes_0.imshow(prev_image/255)
            axes_1.imshow(image/255)

            x_shift, y_shift = aligner.compute_shift(
                image_shape=image.shape,
                features_a=prev_features,
                features_b=features,
                keypoints_a=prev_keypoints,
                keypoints_b=keypoints)
                # axes_a=axes_0,
                # axes_b=axes_1,
                # axes_x_shift=axes_2,
                # axes_y_shift=axes_3)

            x_size = image.shape[1]
            y_size = image.shape[0]

            x_start = x_prev - x_shift
            y_start = y_prev - y_shift

            print("shift: ", x_shift, y_shift)
            print("size: ", x_size, y_size)

            collage_subregion = collage[y_start:y_start+y_size, x_start:x_start+x_size, :]

            print(collage_subregion.shape)
            print(image.shape)

            stitched_image = cv2.addWeighted(collage_subregion, 0.5, image, 0.5, 0)
            collage[y_start:y_start+y_size, x_start:x_start+x_size, :][stitch_mask] = stitched_image[stitch_mask]

            x_prev = x_start
            y_prev = y_start

            # cv2.imwrite("test.png", numpy.flip(collage,axis=2))

            # x_size = image.shape[1]
            # y_size = image.shape[0]
            #
            # x_a_start = max(0, x_shift)
            # y_a_start = max(0, y_shift)
            #
            # x_b_start = max(0, -x_shift)
            # y_b_start = max(0, -y_shift)
            #
            # x_a_stop = x_a_start + x_size
            # y_a_stop = y_a_start + y_size
            #
            # x_b_stop = x_b_start + x_size
            # y_b_stop = y_b_start + y_size
            #
            # stitched_shape = list(image.shape)
            #
            # stitched_shape[0] += abs(y_shift)
            # stitched_shape[1] += abs(x_shift)
            #
            # stitched_image_a = numpy.zeros(stitched_shape, dtype=image.dtype)
            # stitched_image_b = numpy.zeros(stitched_shape, dtype=image.dtype)
            #
            # stitched_image_a[y_a_start:y_a_stop, x_a_start:x_a_stop] = prev_image
            # stitched_image_b[y_b_start:y_b_stop, x_b_start:x_b_stop] = image
            #
            # stitched_image = cv2.addWeighted(stitched_image_a, 0.5, stitched_image_b, 0.5, 0)
            #
            # cv2.imwrite("stitched_"+str(i)+".png", stitched_image)
            # cv2.imwrite("image_"+str(i)+".png", image)

            # pyplot.savefig("test_features_%d.png" % i)


        prev_image = image
        prev_features = features
        prev_keypoints = keypoints

        stop = datetime.now()

        elapsed = stop - start

        print(elapsed.total_seconds())

        i += 1


def main():

    project_directory = os.path.dirname(__file__)

    raw_feature_mask_path = os.path.join(project_directory, "feature_mask_raw.npy")
    stitch_mask_path = os.path.join(project_directory, "feature_mask_90.npy")

    pyplot.imshow(numpy.load(raw_feature_mask_path))
    pyplot.show()
    pyplot.close()
    # pyplot.imshow(numpy.load(stitch_mask_path))
    # pyplot.show()
    # pyplot.close()

    matplotlib.use('Agg')

    aligner = BriefAligner(
        feature_mask_path=raw_feature_mask_path,
        smoothing_radius=None,
        kernel_radius=100,
        n_samples_per_kernel=300,
        n_samples_per_image=800)

    # window_name = "Among Us"
    window_name = "Google Maps"

    run_alignment_loop(window_name, aligner, stitch_mask_path)
    # run_screencapture_loop(window_name)


if __name__ == "__main__":
    main()
