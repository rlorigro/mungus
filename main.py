from matplotlib import pyplot
from modules.align import align
import matplotlib
import pygetwindow
import pyautogui
import numpy
import PIL
import cv2



def run_screencapture_loop():
    candidate_windows = pyautogui.getWindowsWithTitle("Among Us")

    if len(candidate_windows) == 0:
        exit("ERROR: no window with 'Among Us' title found")

    game_window = candidate_windows[0]
    # axis = pyplot.axes()
    # pyplot.ion()
    # pyplot.show(block=False)

    prev_image = None
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

        # if prev_image is not None:
        #     align(image, prev_image, game_window)

        prev_image = image

        # axis.imshow(image)
        # pyplot.draw()
        pyplot.pause(0.05)


def main():
    run_screencapture_loop()


if __name__ == "__main__":
    main()
