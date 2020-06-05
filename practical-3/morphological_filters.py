''' Morphological Filters '''
import cv2
import numpy as np


def main():
    image = cv2.imread('image.png', 0)
    if image is None:
        print("Error Opening Image")
        return -1

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    dilation = cv2.dilate(image, kernel, iterations=1)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    cv2.imwrite("erosion.jpg", erosion)
    cv2.imwrite("dilation.jpg", dilation)
    cv2.imwrite("opening.jpg", opening)
    cv2.imwrite("closing.jpg", closing)
    cv2.imwrite("gradient.jpg", gradient)
    cv2.imwrite("tophat.jpg", tophat)
    cv2.imwrite("blackhat.jpg", blackhat)


if __name__ == "__main__":
    main()