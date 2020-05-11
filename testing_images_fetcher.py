import cv2
import numpy as np

class test_fetcher:

    def test_image_getter():
        print('Take Image of Gesture...')

        capture = cv2.VideoCapture(0)

        img_counter = 0

        while True:
            _, frame = capture.read()

            cv2.rectangle(frame, (300, 300), (600, 600), (0, 0, 0), 2)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_skin = np.array([0, 58, 50], dtype = "uint8")
            upper_skin = np.array([30, 255, 255], dtype = "uint8")

            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            maskROI = mask[300 : 600, 300 : 600]

            result = cv2.bitwise_and(frame, frame, mask = mask)

            cv2.imshow('Original Frame', frame)

            cv2.imshow('Final', maskROI)

            key = cv2.waitKey(1)

            if key == 27:
                break
            elif key%256 == 32:
                img_dest = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/specific_testing_image/opencv_frame{}.png'.format(img_counter)

                cv2.imwrite(img_dest, maskROI)

                print('{} was written and ready to be analyzed!'.format(img_dest))

                img_counter += 1

                break

                capture.release()
                cv2.destroyAllWindows()
