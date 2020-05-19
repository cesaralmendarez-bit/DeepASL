import numpy as np
import cv2
import os
from os.path import isfile, join
from os import listdir, makedirs

class DataGetter:

    def image_taker():
        capture = cv2.VideoCapture(0)

        currentImages = input('On what image do you want to start on?')

        img_taken_counter = int(currentImages)

        directory = input('what directory do you want to write to...')

        while True:
            _, frame = capture.read()

            cv2.rectangle(frame, (300, 300), (600, 600), (0, 0, 0), 2)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_skin = np.array([0, 58, 50], dtype="uint8")
            upper_skin = np.array([30, 255, 255], dtype="uint8")

            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            maskROI = mask[300 : 600, 300 : 600]

            result = cv2.bitwise_and(frame, frame, mask = mask)

            cv2.imshow("Original Frame", frame)

            cv2.imshow("Hand Segmentation", maskROI)

            key = cv2.waitKey(1)

            if key == 27:
                break

            elif key%256 == 32:
                image_destination = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_{}_2/opencv_frame_{}.png'.format(directory, img_taken_counter)

                cv2.imwrite(image_destination, maskROI)

                print('{} was written!'.format(image_destination))

                img_taken_counter += 1

        capture.release()

        cv2.destroyAllWindows()

    def createNewGestureRepo(newLetter):
        newpath = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_{}_2'.format(newLetter)

        if not os.path.exists(newpath):
            os.makedirs(newpath)
