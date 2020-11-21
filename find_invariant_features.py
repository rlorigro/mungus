from modules.extract_features import SCHARR_KERNEL
from modules.extract_features import extract_features
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
import os.path
import numpy
import cv2


def main():
    project_directory = os.path.dirname(__file__)
    data_directory = os.path.join(project_directory, "data")
    test_data_paths = os.listdir(data_directory)

    image_stack = numpy.array([])

    fig1 = pyplot.figure()
    axes1 = pyplot.axes()

    for i in range(len(test_data_paths)):
        path = test_data_paths[i]
        absolute_path = os.path.join(data_directory, path)

        image = cv2.imread(absolute_path)

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if i == 0:
            shape = list(grayscale.shape) + [len(test_data_paths)]

            image_stack.resize(shape)

            print(shape)

        image_stack[:,:,i] = grayscale

    variance = numpy.var(image_stack, axis=2)

    min_mask = (variance < 450)
    max_mask = (variance > 2500)

    mask = numpy.logical_or(min_mask, max_mask)

    print("variance.shape")
    print(variance.shape)

    radius = 90

    expanded_mask = numpy.zeros(mask.shape, dtype=mask.dtype)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == True:
                i_min = max(0,i-radius)
                i_max = min(mask.shape[0]-1,i+radius)

                j_min = max(0,j-radius)
                j_max = min(mask.shape[1]-1,j+radius)

                expanded_mask[i_min:i_max,j_min:j_max] = True

    print("expanded_mask")
    print(expanded_mask.dtype)
    print(expanded_mask.shape)

    im = axes1.imshow(variance)
    divider = make_axes_locatable(axes1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pyplot.colorbar(im, cax=cax)

    fig2 = pyplot.figure()
    axes2 = pyplot.axes()

    variance[expanded_mask] = 0

    im2 = axes2.imshow(variance)
    divider2 = make_axes_locatable(axes2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    pyplot.colorbar(im2, cax=cax2)

    fig3 = pyplot.figure()
    axes3 = pyplot.axes()
    axes3.set_ylabel("Frequency")
    axes3.set_xlabel("Variance")

    variances = variance.flatten()

    axes3.hist(variances, bins=2000)

    pyplot.show()
    pyplot.close()

    numpy.save("feature_mask_" + str(radius) + ".npy", expanded_mask, allow_pickle=False)









if __name__ == "__main__":
    main()
