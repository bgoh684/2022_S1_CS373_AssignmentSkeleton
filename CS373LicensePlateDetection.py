import math
import statistics
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# Queue from Week 12
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

# RGB to greyscale
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    # STUDENT CODE HERE
    for h in range(image_height):
        for w in range(image_width):
            greyscale_pixel_array[h][w] = round(
                0.299 * pixel_array_r[h][w] + 0.587 * pixel_array_g[h][w] + 0.114 * pixel_array_b[h][w])

    return greyscale_pixel_array

# Contrast Stretching
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    fMinMax = computeMinAndMaxValues(pixel_array, image_width, image_height)

    output = createInitializedGreyscalePixelArray(image_width, image_height)
    if fMinMax[0] == fMinMax[1]:
        return output

    for y in range(image_height):
        for x in range(image_width):
            output[y][x] = round((pixel_array[y][x] - fMinMax[0]) * (255 / (fMinMax[1] - fMinMax[0])))
    return output

# Compute min and max pixel values
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    highest = 0
    lowest = 255
    for h in range(image_height):
        for w in range(image_width):
            if pixel_array[h][w] <= lowest:
                lowest = pixel_array[h][w]
            if pixel_array[h][w] >= highest:
                highest = pixel_array[h][w]

    tuple = [lowest, highest]
    return tuple

# Standard Deviation 5x5
def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    array = createInitializedGreyscalePixelArray(image_width, image_height)
    for height in range(image_height):
        for width in range(image_width):
            if height == 0 or width == 0 or height == image_height - 1 or width == image_width - 1:
                continue
            if height == 1 or width == 1 or height == image_height - 2 or width == image_width - 2:
                continue
            else:
                twotopleft2 = pixel_array[height - 2][width - 2]
                twotopleft = pixel_array[height - 2][width - 1]
                twotop = pixel_array[height - 2][width]
                twotopright = pixel_array[height - 2][width + 1]
                twotopright2 = pixel_array[height - 2][width + 2]

                topleft2 = pixel_array[height - 1][width - 2]
                topleft = pixel_array[height - 1][width - 1]
                top = pixel_array[height - 1][width]
                topright = pixel_array[height - 1][width + 1]
                topright2 = pixel_array[height - 1][width + 2]

                left2 = pixel_array[height][width - 2]
                left = pixel_array[height][width - 1]
                center = pixel_array[height][width]
                right = pixel_array[height][width + 1]
                right2 = pixel_array[height][width + 2]

                bottomleft2 = pixel_array[height + 1][width - 2]
                bottomleft = pixel_array[height + 1][width - 1]
                bottom = pixel_array[height + 1][width]
                bottomright = pixel_array[height + 1][width + 1]
                bottomright2 = pixel_array[height + 1][width + 2]

                twobottomleft2 = pixel_array[height + 2][width - 2]
                twobottomleft = pixel_array[height + 2][width - 1]
                twobottom = pixel_array[height + 2][width]
                twobottomright = pixel_array[height + 2][width + 1]
                twobottomright2 = pixel_array[height + 2][width + 2]

                array[height][width] = statistics.pstdev(
                    [twotopleft2,twotopleft,twotop, twotopright, twotopright2, topleft2, topleft, top, topright,
                     topright2, left2, left, center, right, right2, bottomleft2, bottomleft, bottom, bottomright,
                     bottomright2, twobottomleft2, twobottomleft, twobottom, twobottomright, twobottomright2])
    return array

# Thresholding
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    array = createInitializedGreyscalePixelArray(image_width, image_height)
    arrayRow = 0
    for current_array in pixel_array:
        for x in range(image_width):
            if current_array[x] >= threshold_value:
                array[arrayRow][x] = 255
        arrayRow = arrayRow+1
    return array

# Erosion 3x3
def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    array = createInitializedGreyscalePixelArray(image_width, image_height)
    zerosborder = createInitializedGreyscalePixelArray(image_width + 2, image_height + 2)

    for h in range(image_height):
        for w in range(image_width):
            zerosborder[h + 1][w + 1] = pixel_array[h][w]

    for height in range(1, image_height + 1):
        for width in range(1, image_width + 1):
            topleft = zerosborder[height - 1][width - 1]
            top = zerosborder[height - 1][width]
            topright = zerosborder[height - 1][width + 1]

            left = zerosborder[height][width - 1]
            center = zerosborder[height][width]
            right = zerosborder[height][width + 1]

            bottomleft = zerosborder[height + 1][width - 1]
            bottom = zerosborder[height + 1][width]
            bottomright = zerosborder[height + 1][width + 1]

            if topleft == 0 or top == 0 or topright == 0 or left == 0 or center == 0 or right == 0 or bottomleft == 0 or bottom == 0 or bottomright == 0:
                array[height - 1][width - 1] = 0
            else:
                array[height - 1][width - 1] = 1
    return array

# Dilation 3x3
def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    array = createInitializedGreyscalePixelArray(image_width, image_height)
    zerosborder = createInitializedGreyscalePixelArray(image_width + 2, image_height + 2)

    for h in range(image_height):
        for w in range(image_width):
            zerosborder[h + 1][w + 1] = pixel_array[h][w]

    for height in range(1, image_height + 1):
        for width in range(1, image_width + 1):
            topleft = zerosborder[height - 1][width - 1]
            top = zerosborder[height - 1][width]
            topright = zerosborder[height - 1][width + 1]

            left = zerosborder[height][width - 1]
            center = zerosborder[height][width]
            right = zerosborder[height][width + 1]

            bottomleft = zerosborder[height + 1][width - 1]
            bottom = zerosborder[height + 1][width]
            bottomright = zerosborder[height + 1][width + 1]

            if topleft != 0 or top != 0 or topright != 0 or left != 0 or center != 0 or right != 0 or bottomleft != 0 or bottom != 0 or bottomright != 0:
                array[height - 1][width - 1] = 1
            else:
                array[height - 1][width - 1] = 0
    return array

# Connected Components
def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    label = 1
    visited = createInitializedGreyscalePixelArray(image_width, image_height)
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    labels = {}
    q = Queue()

    for h in range(image_height):
        for w in range(image_width):
            if visited[h][w] == 1:
                continue

            if pixel_array[h][w] != 0:
                visited[h][w] = 1
                labels[label] = 0
                Queue.enqueue(q, [h, w])

                while Queue.isEmpty(q) != 1:
                    position = Queue.dequeue(q)
                    height = position[0]
                    width = position[1]
                    output[height][width] = label
                    visited[height][width] = 1

                    if height - 1 >= 0:
                        if pixel_array[height - 1][width] != 0 and visited[height - 1][width] == 0:
                            Queue.enqueue(q, [height - 1, width])
                            output[height - 1][width] = label
                            visited[height - 1][width] = 1

                    if height + 1 <= image_height - 1:
                        if pixel_array[height + 1][width] != 0 and visited[height + 1][width] == 0:
                            Queue.enqueue(q, [height + 1, width])
                            output[height + 1][width] = label
                            visited[height + 1][width] = 1

                    if width - 1 >= 0:
                        if pixel_array[height][width - 1] != 0 and visited[height][width - 1] == 0:
                            Queue.enqueue(q, [height, width - 1])
                            output[height][width - 1] = label
                            visited[height][width - 1] = 1

                    if width + 1 <= image_width - 1:
                        if pixel_array[height][width + 1] != 0 and visited[height][width + 1] == 0:
                            Queue.enqueue(q, [height, width + 1])
                            output[height][width + 1] = label
                            visited[height][width + 1] = 1

                label = label + 1

    for y in range(image_height):
        for x in range(image_width):
            if (output[y][x] == 0):
                continue
            labels[output[y][x]] = labels.get(output[y][x]) + 1

    return (output, labels)

def calculateBoxPosition(connected , labels, image_width, image_height, disregard):
    numPixels = 0
    component = 0
    for nr in labels.keys():
        if nr == disregard:
            continue

        if labels[nr] > numPixels:
            numPixels = labels[nr]
            component = nr

    bbox_min_x = image_width
    bbox_max_x = 0
    bbox_min_y = image_height
    bbox_max_y = 0

    for h in range(image_height):
        for w in range(image_width):
            if connected[h][w] != component:
                continue
            else:
                if h < bbox_min_y:
                    bbox_min_y = h

                if h > bbox_max_y:
                    bbox_max_y = h

                if w < bbox_min_x:
                    bbox_min_x = w

                if w > bbox_max_x:
                    bbox_max_x = w
    if ((bbox_max_x - bbox_min_x)/(bbox_max_y-bbox_min_y) > 5 or (bbox_max_x - bbox_min_x)/(bbox_max_y-bbox_min_y) < 1.5):
        disregard = component
        calculateBoxPosition(connected, labels, image_width, image_height, disregard)

    return (bbox_min_x,bbox_max_x,bbox_min_y,bbox_max_y)

# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate4.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here

    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)
    px_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)
    #Stretch again
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)
    px_array = computeThresholdGE(px_array, 140, image_width, image_height)
    #Dilate 3 times
    for a in range(3):
        px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)

    #Erode 3 times
    for a in range(3):
        px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)

    #Connect components
    (connected, labels) = computeConnectedComponentLabeling(px_array, image_width, image_height)

    #Get box position
    (bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y) = calculateBoxPosition(connected, labels, image_width, image_height, 0)

    #Display original image
    px_array = px_array_r

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)



    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()