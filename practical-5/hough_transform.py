''' Hough Transform'''
import cv2
import numpy as np


def main():
    image = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
    if image is None:
        print("Error Opening Image")
        return -1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray =  cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2.imwrite('houghlines.jpg', image)


if __name__ == "__main__":
    main()
