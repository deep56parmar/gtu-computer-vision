''' Mean Filter '''
import cv2



def main():
    image = cv2.imread('../practical-1/demo.jpg', cv2.IMREAD_COLOR)
    if image is None:
        print("Error Opening Image")
        return -1

    filtered_image1 = cv2.blur(image,(9,9))

    cv2.imwrite("Filter1.jpg", filtered_image1)


if __name__ == "__main__":
    main()