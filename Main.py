import cv2
import time as t
import serial
from matplotlib import pyplot as plt




# sets up cv2 and serial object, also creates a list for the images to be stored
def setup():
    cap = cv2.VideoCapture(1)
    images = []
    ser = serial.Serial('COM3', 9600)
    t.sleep(2)
    return cap, images, ser

# takes a picture, converts it to grey scale, and then adds it to the images list
def capture_imgs():
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img)

# applys edge detection and displays the images one at a time from the image list
def show_imgs():
    for i in range(1):
        '''
        mean = cv2.mean(images[i])
        min = .66 * mean[0]
        max = 1.33 * mean[0]
        edges = cv2.Canny(images[i], min, max)
        '''
        edges = cv2.Canny(images[i],100,100)
        plt.subplot(121), plt.imshow(images[i], cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
def deleteFirstImg():
    del images[0]


#main code that sends the signal to the arduino to start the motor, tkaes the pictures, then turns off the motor. Finally the images are displayed
if __name__ == "__main__":
    cap, images, ser = setup()
    ser.write(b'1')
    for i in range(2):
        capture_imgs()
        t.sleep(2)
    ser.write(bytes(b'0'))
    deleteFirstImg()
    show_imgs()