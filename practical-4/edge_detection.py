'''Edge Detection '''
import cv2
import numpy as np


def main():
    image = cv2.imread('../practical-1/demo.jpg')
    if image is None:
        print("Error Opening Image")
        return -1

    canny = cv2.Canny(image, 100, 200)

    blurred = cv2.blur(image, (9, 9))
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    robert_cross_x = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    robert_cross_y = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
    grad_x = cv2.filter2D(image,-1,robert_cross_x)
    grad_y = cv2.filter2D(image,-1,robert_cross_y)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    robert = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    grad_x = cv2.filter2D(image, -1, prewitt_x)
    grad_y = cv2.filter2D(image, -1, prewitt_y)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    prewitt = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv2.imwrite("canny.jpg", canny)
    cv2.imwrite("sobel.jpg", sobel)
    cv2.imwrite("Robert.jpg", robert)
    cv2.imwrite("prewitt.jpg", prewitt)


if __name__ == "__main__":
    main()