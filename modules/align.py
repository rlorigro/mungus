import numpy
import cv2


def align(image_a, image_b):
    # cv_image_a = cv2.UMat(numpy.array(image_a))
    # cv_image_b = cv2.UMat(numpy.array(image_b))

    grayscale_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    grayscale_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    max_iterations = 5000
    terminal_eps = 1e-2

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, terminal_eps)

    warp_matrix = numpy.eye(2, 3, dtype=numpy.float32)
    (c, warp_matrix) = cv2.findTransformECC(grayscale_a, grayscale_b, warp_matrix, cv2.MOTION_TRANSLATION, criteria, None, 1)

    # print(warp_matrix.get())

    # aligned_b = cv2.warpAffine(
    #     cv_image_b,
    #     warp_matrix,
    #     (game_window.width, game_window.height),
    #     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # pyplot.imshow(aligned_b.get())

    return warp_matrix