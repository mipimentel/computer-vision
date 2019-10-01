import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


# Basics
def load_image(image):
    image = cv.imread(image)
    # OpenCV default color format is BGR
    # convert it to RGB format for matplotlib
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def show(image):
    # Figure size in inches
    plt.figure(figsize=(12, 8))

    # check if image has color
    if len(image.shape) == 3:
        # Show image, with nearest neighbour interpolation
        plt.imshow(image, interpolation="nearest")
    else:
        plt.imshow(image, interpolation="nearest", cmap="gray")
    plt.tight_layout()
    plt.show()


# Colorspace visualization
def show_rgb(image):
    # Show Red/Green/Blue
    # we set other channels to zero for seeing it better
    images = []
    for i in [0, 1, 2]:
        colour = image.copy()
        if i != 0:
            colour[:, :, 0] = 0
        if i != 1:
            colour[:, :, 1] = 0
        if i != 2:
            colour[:, :, 2] = 0
        images.append(colour)

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.axis("off")
    plt.title("R")
    plt.imshow(images[0])
    plt.subplot(132)
    plt.axis("off")
    plt.imshow(images[1])
    plt.title("G")
    plt.subplot(133)
    plt.axis("off")
    plt.title("B")
    plt.imshow(images[2])
    plt.tight_layout()
    plt.show()


def show_lab(image):
    # Convert from RGB to Lab
    lab = cv.cvtColor(image, cv.COLOR_RGB2LAB)

    images = []
    for i in [0, 1, 2]:
        colour = lab.copy()
        if i != 0:
            # we set Luminance to a medium value
            colour[:, :, 0] = 127
        if i != 1:
            # the a and b channels range from -127 to +127
            # so we set it to zero in a 0 to 255 which will be 127
            colour[:, :, 1] = 127
        if i != 2:
            colour[:, :, 2] = 127
        images.append(cv.cvtColor(colour, cv.COLOR_LAB2RGB))

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.axis("off")
    plt.title("L")
    plt.imshow(images[0])
    plt.subplot(132)
    plt.axis("off")
    plt.imshow(images[1])
    plt.title("a")
    plt.subplot(133)
    plt.axis("off")
    plt.title("b")
    plt.imshow(images[2])
    plt.tight_layout()
    plt.show()


def show_hsv(image):
    # Convert from RGB to HSV
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    images = []
    for i in [0, 1, 2]:
        colour = hsv.copy()
        if i != 0:
            # for the other dimensions we set Hue to 0
            # which will result in a red color, but any color should be fine
            colour[:, :, 0] = 0
        if i != 1:
            # max out Saturation
            colour[:, :, 1] = 255
        if i != 2:
            # max out Value
            colour[:, :, 2] = 255
        images.append((cv.cvtColor(colour, cv.COLOR_HSV2RGB)))

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.axis("off")
    plt.title("Hue")
    plt.imshow(images[0])
    plt.subplot(132)
    plt.axis("off")
    plt.imshow(images[1])
    plt.title("Saturation")
    plt.subplot(133)
    plt.axis("off")
    plt.title("Value")
    plt.imshow(images[2])
    plt.tight_layout()
    plt.show()


def show_sobel(image):
    # Transform from RGB to gray and calculate dx and dy
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    dx = cv.Sobel(gray, cv.CV_32F, 1, 0)
    dy = cv.Sobel(gray, cv.CV_32F, 0, 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.axis("off")
    plt.title("image")
    plt.imshow(gray, cmap="gray")
    plt.subplot(132)
    plt.axis("off")
    plt.imshow(dx, cmap="gray")
    plt.title(r"$\frac{dI}{dx}$")
    plt.subplot(133)
    plt.axis("off")
    plt.title(r"$\frac{dI}{dy}$")
    plt.imshow(dy, cmap="gray")
    plt.tight_layout()
    plt.show()


# 3D Plots
def plot_3dRGB(image):
    # split RGB channels
    r, g, b = cv.split(image)
    fig = plt.figure()  # figsize=(15, 15)
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    # Reshape it
    pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    # Normalize for matplotlib
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(
        r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker="."
    )
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.tight_layout()
    plt.show()


def plot_3dHSV(image):
    # split RGB channels
    r, g, b = cv.split(image)
    fig = plt.figure()  # figsize=(15, 15)
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    # Reshape it
    pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    # Normalize
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(image)
    axis.scatter(
        h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker="."
    )
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.tight_layout()
    plt.show()


def plot_3dLAB(image):
    # split RGB channels
    r, g, b = cv.split(image)
    fig = plt.figure()  # figsize=(15, 15)
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    # Reshape it
    pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    # Normalize
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    image = cv.cvtColor(image, cv.COLOR_RGB2LAB)
    l, a, b = cv.split(image)
    axis.scatter(
        l.flatten(), a.flatten(), b.flatten(), facecolors=pixel_colors, marker="."
    )
    axis.set_xlabel("Lightness")
    axis.set_ylabel("a")
    axis.set_zlabel("b")
    plt.tight_layout()
    plt.show()


def show_rgb_hist(image):
    colours = ("r", "g", "b")
    pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    # print(pixel_colors)
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    pixel_colors = norm(pixel_colors).tolist()

    for i, c in enumerate(colours):
        plt.figure(figsize=(12, 4))
        histr = cv.calcHist([image], [i], None, [256], [0, 256])
        # print(histr)
        if c == "r":
            colours = [(i / 256, 0, 0) for i in range(0, 256)]
        # print(colours)
        if c == "g":
            colours = [((0, i / 256, 0)) for i in range(0, 256)]
        if c == "b":
            colours = [((0, 0, i / 256)) for i in range(0, 256)]
        norm.autoscale(colours)
        plt.bar(
            range(0, 256),
            histr.ravel(),
            color=norm(colours),
            edgecolor=norm(colours),
            width=2,
        )
        plt.tight_layout()

    plt.show()


def show_hsv_hist(image):
    # Hue
    image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    # print(pixel_colors)
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    pixel_colors = norm(pixel_colors).tolist()

    plt.figure(figsize=(12, 4))
    histr = cv.calcHist([image], [0], None, [180], [0, 180])
    plt.xlim([0, 180])
    colours = [colors.hsv_to_rgb((i / 180, 1, 1)) for i in range(0, 180)]
    plt.bar(
        range(0, 180),
        histr.ravel(),
        color=norm(colours),
        edgecolor=norm(colours),
        width=1,
    )
    plt.title("Hue")
    plt.tight_layout()

    # Saturation
    plt.figure(figsize=(12, 4))
    histr = cv.calcHist([image], [1], None, [256], [0, 256])
    plt.xlim([0, 256])

    colours = [colors.hsv_to_rgb((0, i / 256, 1)) for i in range(0, 256)]
    plt.bar(
        range(0, 256),
        histr.ravel(),
        color=norm(colours),
        edgecolor=norm(colours),
        width=1,
    )
    plt.title("Saturation")
    plt.tight_layout()

    # Value
    plt.figure(figsize=(12, 4))
    histr = cv.calcHist([image], [2], None, [256], [0, 256])
    plt.xlim([0, 256])

    colours = [colors.hsv_to_rgb((0, 1, i / 256)) for i in range(0, 256)]
    plt.bar(
        range(0, 256),
        histr.ravel(),
        color=norm(colours),
        edgecolor=norm(colours),
        width=1,
    )
    plt.title("Value")
    plt.tight_layout()
    plt.show()


# Threshold
def plot_thresh_comparrison(image):
    ret, th1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(
        image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2
    )
    th3 = cv.adaptiveThreshold(
        image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )
    titles = [
        "Original Image",
        "Global Thresholding (v = 127)",
        "Adaptive Mean Thresholding",
        "Adaptive Gaussian Thresholding",
    ]
    images = [image, th1, th2, th3]

    plt.figure(figsize=(12, 12))
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
    plt.show()


def segment_otsu(image):
    image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return image[1]


def custom_gray(image, rgb_weights=np.array([1.3, 0.3, -0.8])):
    grayimg = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # create some kind of "plane"
            graylevel = rgb_weights[0] * image[i][j][0]
            graylevel += rgb_weights[1] * image[i][j][1]
            graylevel += rgb_weights[2] * image[i][j][2]
            graylevel = int(graylevel)
            # clip to avoid overflow
            grayimg[i][j] = np.clip(graylevel, 0, 255)
    return grayimg.astype(np.uint8)


def overlay_mask(image, mask, alpha=0.5, beta=0.5):
    rgb_mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    image = cv.addWeighted(rgb_mask, alpha, image, beta, 0)
    return image
