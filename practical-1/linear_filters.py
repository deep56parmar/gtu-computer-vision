''' Linear Filters ''' 
import cv2 
import numpy as np
import time

def main():
    image = cv2.imread('demo.jpg', cv2.IMREAD_COLOR)
    if image is None:
        print("Error Opening Image")
        return -1
    filter1  = np.array([
        [1/9,0,1/9],
        [0,1,0],
        [1/9,0,1/9]
    ])
    filter2  = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])
    
    filter3  = np.array([
        [1/2,0,0],
        [-1,0,0],
        [0,0,1/2]
    ])

    
    filter4  = np.array([
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]
    ])

    
    filter5  = np.array([
        [-1/9,1/9,1/9],
        [1/9,-1/9,1/9],
        [1/9,1/9,-1/9]
    ])

    filtered_image1 = cv2.filter2D(image,-1,filter1)
    filtered_image2 = cv2.filter2D(image,-1,filter2)
    filtered_image3 = cv2.filter2D(image,-1,filter3)
    filtered_image4 = cv2.filter2D(image,-1,filter4)
    filtered_image5 = cv2.filter2D(image,-1,filter5)

    cv2.imwrite("Filter1.jpg", filtered_image1)
    cv2.imwrite("Filter2.jpg", filtered_image2)
    cv2.imwrite("Filter3.jpg", filtered_image3)
    cv2.imwrite("Filter4.jpg", filtered_image4)
    cv2.imwrite("Filter5.jpg", filtered_image5)

    



if __name__ == "__main__":
    main()