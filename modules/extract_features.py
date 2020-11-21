from scipy import signal
import numpy
import cv2


SCHARR_KERNEL = numpy.array([[ -3-3j, 0-10j,  +3 -3j],
                             [-10+0j, 0+ 0j, +10 +0j],
                             [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy


def extract_features(grayscale_image, kernel):
    convolved = signal.convolve2d(grayscale_image, kernel, mode="valid")

    cv2.imshow("B", numpy.absolute(convolved))
    cv2.waitKey()

