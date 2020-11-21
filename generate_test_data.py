from matplotlib import pyplot
from matplotlib import patches
import matplotlib
import os
import numpy
import cv2


def main():
    project_directory = os.path.dirname(__file__)
    source_path = os.path.join(project_directory, "data2/b.png")

    image = cv2.imread(source_path)

    print(image.shape)

    width = image.shape[0]
    height = image.shape[1]

    step_size = 100
    window_size = 200
    n_steps_x = int((width-200)/step_size)
    n_steps_y = int((height-200)/step_size)

    print(n_steps_x,n_steps_y)

    x_offsets = [i*step_size for i in range(n_steps_x)]
    y_offsets = [i*step_size for i in range(n_steps_y)]

    x_offsets_cycle = x_offsets + [x_offsets[-1]]*len(y_offsets) + list(reversed(x_offsets)) + [x_offsets[0]]*len(y_offsets)
    y_offsets_cycle = [y_offsets[-1]]*len(x_offsets) + list(reversed(y_offsets)) + [y_offsets[0]]*len(x_offsets) + y_offsets

    axes = pyplot.axes()
    pyplot.imshow(image)
    for i in range(len(x_offsets_cycle)):
        x_start = x_offsets_cycle[i]
        x_stop = x_start + window_size

        y_start = y_offsets_cycle[i]
        y_stop = y_start + window_size

        r = pyplot.Rectangle(xy=(x_start,y_start), width=200, height=200, linewidth=1,edgecolor='r',facecolor='none')
        axes.add_patch(r)
        pyplot.show(block=False)
        pyplot.pause(0.1)

        cv2.imwrite("test_"+str(i)+".png",image[x_start:x_stop,y_start:y_stop])


if __name__ == "__main__":
    main()

