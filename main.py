from modules.align import BriefAligner

from matplotlib import pyplot
import matplotlib

from datetime import datetime
from scipy import signal
from time import sleep
import pyautogui
import os.path
import numpy
import cv2


def run_screencapture_loop():
    candidate_windows = pyautogui.getWindowsWithTitle("Among Us")

    if len(candidate_windows) == 0:
        exit("ERROR: no window with 'Among Us' title found")

    game_window = candidate_windows[0]
    # axis = pyplot.axes()
    # pyplot.ion()
    # pyplot.show(block=False)

    i=0
    while (True):
        if not game_window.isActive:
            game_window.activate()
            pyplot.pause(0.1)

        window_region = (game_window.left + 10, game_window.top + 32, game_window.width - 20, game_window.height - 42)
        print(window_region)

        image = pyautogui.screenshot(region=window_region, imageFilename="screenshot_"+str(i)+".png")

        i = i + 1

        print(image.size)

        # axis.imshow(image)
        # pyplot.draw()
        pyplot.pause(0.05)




def run_alignment_loop(window_name, aligner):
    candidate_windows = pyautogui.getWindowsWithTitle(window_name)

    if len(candidate_windows) == 0:
        exit("ERROR: no window with '" + window_name + "' title found")

    game_window = candidate_windows[0]

    fig = pyplot.figure(figsize=[14,9])
    gs = fig.add_gridspec(ncols=4, nrows=2)
    axes_0 = fig.add_subplot(gs[0, 0])
    axes_1 = fig.add_subplot(gs[1, 0])
    axes_2 = fig.add_subplot(gs[0, 1])
    axes_3 = fig.add_subplot(gs[1, 1])
    axes_4 = fig.add_subplot(gs[:, 2:])

    axes_2.set_title("x_shift histogram")
    axes_3.set_title("y_shift histogram")

    prev_image = None
    prev_features = None
    prev_coord_indexes = None
    i=0
    while (True):
        start = datetime.now()

        axes_0.clear()
        axes_1.clear()
        axes_2.clear()
        axes_3.clear()
        axes_4.clear()

        if not game_window.isActive:
            game_window.activate()
            sleep(0.2)

        window_region = (game_window.left + 10, game_window.top + 32, game_window.width - 20, game_window.height - 42)

        image = numpy.float32(pyautogui.screenshot(region=window_region))

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        grayscale_smoothed = signal.oaconvolve(grayscale, aligner.smoothing_kernel, mode="same")

        coord_indexes, features = aligner.extract_features(grayscale_image=grayscale_smoothed)

        if prev_image is not None:
            axes_0.imshow(prev_image/255)
            axes_1.imshow(image/255)

            x_shift, y_shift = aligner.compute_shift(
                image_shape=image.shape,
                features_a=prev_features,
                features_b=features,
                coord_indexes_a=prev_coord_indexes,
                coord_indexes_b=coord_indexes,
                axes_a=axes_0,
                axes_b=axes_1,
                axes_x_shift=axes_2,
                axes_y_shift=axes_3)

            x_size = image.shape[1]
            y_size = image.shape[0]

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

            stitched_shape = list(image.shape)

            stitched_shape[0] += abs(y_shift)
            stitched_shape[1] += abs(x_shift)

            stitched_image_a = numpy.zeros(stitched_shape, dtype=image.dtype)
            stitched_image_b = numpy.zeros(stitched_shape, dtype=image.dtype)

            stitched_image_a[y_a_start:y_a_stop, x_a_start:x_a_stop] = prev_image
            stitched_image_b[y_b_start:y_b_stop, x_b_start:x_b_stop] = image

            stitched_image = cv2.addWeighted(stitched_image_a,0.5,stitched_image_b,0.5,0)

            axes_4.imshow(stitched_image/255)

            pyplot.savefig("stitched_image_" + str(i) + ".png")

        prev_image = image
        prev_features = features
        prev_coord_indexes = coord_indexes

        stop = datetime.now()

        elapsed = stop - start
        elapsed_seconds = elapsed.total_seconds()

        print(elapsed.total_seconds())

        if elapsed_seconds < 1:
            sleep(1-elapsed_seconds)

        i += 1


def main():
    matplotlib.use('Agg')

    project_directory = os.path.dirname(__file__)

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

    window_name = "Among Us"

    run_alignment_loop(window_name, aligner)


if __name__ == "__main__":
    main()
