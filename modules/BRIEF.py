from matplotlib import pyplot
from random import randint
import numpy
import cv2


# BRIEF: Binary Robust Independent Elementary Features
# Michael Calonder, Vincent Lepetit, Christoph Strecha, and Pascal Fua


def generate_kernel(size, n_samples):
    mask = numpy.zeros(shape=[size,size], dtype=numpy.bool)

    for i in range(n_samples):
        x = randint(0,size-1)
        y = randint(0,size-1)

        mask[x,y] = True

    return mask


def extract_features(grayscale_image, feature_mask_coordinates, kernel_a, kernel_b, n_samples_per_image):
    x_length = kernel_a.shape[0]
    y_length = kernel_a.shape[1]

    if x_length % 2 != 1:
        exit("ERROR: even kernel dimension cannot be centered")

    if y_length % 2 != 1:
        exit("ERROR: even kernel dimension cannot be centered")

    x_radius = int((x_length - 1) / 2)
    y_radius = int((y_length - 1) / 2)

    kernel_a_coords = numpy.nonzero(kernel_a)
    kernel_b_coords = numpy.nonzero(kernel_b)

    coord_indexes = list()
    features = list()

    for n in range(n_samples_per_image):
        i = randint(0,len(feature_mask_coordinates[0])-1)
        coord_indexes.append(i)

        x = feature_mask_coordinates[1][i]
        y = feature_mask_coordinates[0][i]

        # print(feature_mask_coordinates[0][0:100])
        # print(feature_mask_coordinates[1][0:100])
        # print("i")
        # print(i)
        # print("x, y")
        # print(x, y)
        # print("x_radius, y_radius")
        # print(x_radius, y_radius)

        a = grayscale_image[y-y_radius:y+y_radius+1,x-x_radius:x+x_radius+1]
        b = grayscale_image[y-y_radius:y+y_radius+1,x-x_radius:x+x_radius+1]

        # fig, axes = pyplot.subplots(ncols=2)
        # axes[0].imshow(grayscale_image)
        # r = pyplot.Rectangle(xy=(x-x_radius,y-y_radius), width=x_radius*2+1, height=y_radius*2+1, linewidth=1,edgecolor='r',facecolor='none')
        # axes[0].add_patch(r)
        # axes[1].imshow(a)
        # pyplot.show()
        # pyplot.close()

        # print("a.shape, b.shape")
        # print(a.shape, b.shape)
        # print("kernel_a.shape, kernel_b.shape")
        # print(kernel_a.shape, kernel_b.shape)

        # fig, axes = pyplot.subplots(ncols=2, nrows=2)
        # axes[0][0].imshow(kernel_a)
        # axes[0][1].imshow(a)
        # axes[1][0].imshow(kernel_b)
        # axes[1][1].imshow(b)
        # pyplot.show()
        # pyplot.close()

        a = a[kernel_a]
        b = b[kernel_b]

        feature = (a > b)

        # print("a.shape, b.shape")
        # print(a.shape, b.shape)
        # print(a)
        # print(b)
        # print(feature)
        # input()

        features.append(feature)

    return coord_indexes, features




