# check opencv version
import cv2
from cv2 import CascadeClassifier
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import rectangle

# print the version number
print(cv2.__version__)

# load the photograph
pixels = imread('test2.jpg')

# load the pre-trained model
# 
# see github page. https://github.com/opencv/opencv/tree/master/data/haarcascades
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# perform face detection
bboxes = classifier.detectMultiScale(pixels, 1.05, 7)
# print boinding box for each detected face 
for box in bboxes:
    print(box)
    # extract coordinates
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels
    rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)

# show the image
imshow('face detection', pixels)

# keep the window open until we press a key
waitKey(0)

# close the window
destroyAllWindows()

