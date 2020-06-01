import numpy as np
import cv2
import sys
import os
from os.path import isfile, join
from os import listdir, makedirs
from PIL import Image
from datetime import datetime
from PIL import Image, ImageOps
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from TrainedSoftmax import Softmax2
from TrainedConv import Conv3x3Trained
import pickle
from training_images_fetcher import DataGetter
from send_sms import trainNotification
from tts import sayLetter
import threading
from threading import Timer
from writingToTxtFile import writeToFile
import imutils
import math
import time
import itertools
import random
from collections import deque
from numpy import asarray
from numpy import savetxt
np.set_printoptions(threshold=sys.maxsize)

conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13 * 13 * 8, 7)
dataGetterClass = DataGetter
trainingNotify = trainNotification
tts = sayLetter
toFile = writeToFile
softmax2 = Softmax2(13 * 13 * 8, 7)
conv2 = Conv3x3Trained(8)

#Train, Test, Or Clear A Model
toTrainOrNot = input('Train New Model(Y), Test Model (N), Clear Model(C)')

#Conditional-Train New Model
if toTrainOrNot == 'Y':

    #Training Image Dataset Directories
    train_images_dir_a = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_a_2'
    train_images_dir_l = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_l_2'
    train_images_dir_b = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_b_2'
    train_images_dir_c = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_c_2'
    train_images_dir_h = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_h_2'
    train_images_dir_e = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_e_2'
    halfset_train_images_dir_a = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/halfSetA'
    halfset_train_images_dir_l = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/halfSetL'
    halfset_train_images_dir_b = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/halfSetB'
    halfset_train_images_dir_c = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/halfSetC'
    halfset_train_images_dir_h = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/halfSetH'
    min_a_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minA'
    min_l_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minL'
    min_b_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minB'
    min_c_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minC'
    min_h_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minH'
    min_e_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minE'
    min_d_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minD'
    #Empty List For Appendable Images
    train_images_list = []

    #Image Training Bounds
    countOfImgToTrain = input('How Many Images Do You Want To Train On: ')

    #Parse Image Training Bounds Into Integer
    intCountOfImgToTrain = int(countOfImgToTrain)

    #Epoch Training Bounds
    countOfEpochs = input('How Many Epochs: ')

    #Parse Epoch Training Bounds Into Integer
    intCountOfEpochs = int(countOfEpochs)

    #Rep Keeps Track On Which Image The Loop Is On In Order To Stop At Required One
    rep = 1

    #Resizing and Grayscaling of Training Images and Appending Into List
    #A
    for filename in os.listdir(min_a_dir):
        #Do Not Read Metadata
        if filename != '.DS_Store' and rep <= intCountOfImgToTrain:
            readimg = cv2.imread(min_a_dir + '/' + filename)

            resized = cv2.resize(readimg, dsize=(28, 28), interpolation = cv2.INTER_AREA)

            grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            train_images_list.append(grayscale)

            rep += 1

    rep = 1
    #L
    for filename2 in os.listdir(min_l_dir):

        if filename2 != '.DS_Store' and rep <= intCountOfImgToTrain:
            readimg2 = cv2.imread(min_l_dir + '/' + filename2)

            resized2 = cv2.resize(readimg2, dsize=(28, 28), interpolation = cv2.INTER_AREA)

            grayscale2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)

            train_images_list.append(grayscale2)

            rep += 1

    rep = 1
    #B
    for filename3 in os.listdir(min_b_dir):

        if filename3 != '.DS_Store' and rep <= intCountOfImgToTrain:
            readimg3 = cv2.imread(min_b_dir + '/' + filename3)

            resized3 = cv2.resize(readimg3, dsize=(28, 28), interpolation = cv2.INTER_AREA)

            grayscale3 = cv2.cvtColor(resized3, cv2.COLOR_BGR2GRAY)

            train_images_list.append(grayscale3)

            rep += 1

    rep = 1
    #C
    for filename4 in os.listdir(min_c_dir):

        if filename4 != '.DS_Store' and rep <= intCountOfImgToTrain:
            readimg4 = cv2.imread(min_c_dir + '/' + filename4)

            resized4 = cv2.resize(readimg4, dsize=(28, 28), interpolation = cv2.INTER_AREA)

            grayscale4 = cv2.cvtColor(resized4, cv2.COLOR_BGR2GRAY)

            train_images_list.append(grayscale4)

            rep += 1

    rep = 1
    #H
    for filename5 in os.listdir(min_h_dir):
        #Do Not Read Metadata
        if filename5 != '.DS_Store' and rep <= intCountOfImgToTrain:
            readimg5 = cv2.imread(min_h_dir + '/' + filename5)

            resized5 = cv2.resize(readimg5, dsize=(28, 28), interpolation = cv2.INTER_AREA)

            grayscale5 = cv2.cvtColor(resized5, cv2.COLOR_BGR2GRAY)

            train_images_list.append(grayscale5)

            rep += 1

    rep = 1
    #E
    for filename6 in os.listdir(min_e_dir):
        #Do Not Read Metadata
        if filename6 != '.DS_Store' and rep <= intCountOfImgToTrain:
            readimg6 = cv2.imread(min_e_dir + '/' + filename6)

            resized6 = cv2.resize(readimg6, dsize=(28, 28), interpolation = cv2.INTER_AREA)

            grayscale6 = cv2.cvtColor(resized6, cv2.COLOR_BGR2GRAY)

            train_images_list.append(grayscale6)

            rep += 1

    rep = 1
    #D
    for filename7 in os.listdir(min_d_dir):
        #Do Not Read Metadata
        if filename7 != '.DS_Store' and rep <= intCountOfImgToTrain:
            readimg7 = cv2.imread(min_d_dir + '/' + filename6)

            resized7 = cv2.resize(readimg7, dsize=(28, 28), interpolation = cv2.INTER_AREA)

            grayscale7 = cv2.cvtColor(resized7, cv2.COLOR_BGR2GRAY)

            train_images_list.append(grayscale7)

            rep += 1

    rep = 1
    #Convert List To Numpy Array
    train_images = np.array(train_images_list)

    #Initializing Of Training Labels, e.g([0xintCountOfImgToTrain, 1xintCountOfImgToTrain, 2xintCountOfImgToTrain])
    zeros = np.full((1, intCountOfImgToTrain), 0)
    ones = np.full((1, intCountOfImgToTrain), 1)
    twos = np.full((1, intCountOfImgToTrain), 2)
    threes = np.full((1, intCountOfImgToTrain), 3)
    fours = np.full((1, intCountOfImgToTrain), 4)
    fives = np.full((1, intCountOfImgToTrain), 5)
    sixes = np.full((1, intCountOfImgToTrain), 6)
    finalZeros = zeros.ravel()
    finalOnes = ones.ravel()
    finalTwos = twos.ravel()
    finalThrees = threes.ravel()
    finalFours = fours.ravel()
    finalFives = fives.ravel()
    finalSixes = sixes.ravel()

    #Concatenating Of All Labels Into Singulary Array Variable
    train_labels = np.concatenate([finalZeros, finalOnes, finalTwos, finalThrees, finalFours, finalFives, finalSixes])


    def forward(image, label):
        out = conv.forward((image / 255) - 0.5)

        out = pool.forward(out)

        out = softmax.forward(out)

        ('Confidence Percentages For Each Letter')
        print(out)

        loss = -np.log(out[label])

        acc = 1 if np.argmax(out) == label else 0

        rightOrWrongEval = 'Correct Prediction' if np.argmax(out) == label else 'Incorrect Prediction'

        print("Network Prediction: ")
        if np.argmax(out) == 0:
            print('I think that sign language letter is: A')
        elif np.argmax(out) == 1:
            print('I think that sign language letter is: L')
        elif np.argmax(out) == 2:
            print('I think that sign language letter is: B')
        elif np.argmax(out) == 3:
            print('I think that sign language letter is: C')
        elif np.argmax(out) == 4:
            print('I think that sign language letter is: H')
        elif np.argmax(out) == 5:
            print('I think that sign language letter is: E')
        elif np.argmax(out) == 6:
            print('I think that sign language letter is: D')

        print('Evaluation')
        print(rightOrWrongEval)

        return out, loss, acc

    def forwardForTest(image):
        out = conv.forward((image / 255) - 0.5)

        out = pool.forward(out)

        out = softmax.forward(out)

        print("Network Prediction: ")
        if np.argmax(out) == 0:
            print('I think that sign language letter is: A')
        elif np.argmax(out) == 1:
            print('I think that sign language letter is: L')
        elif np.argmax(out) == 2:
            print('I think that sign language letter is: B')
        elif np.argmax(out) == 3:
            print('I think that sign language letter is: C')
        elif np.argmax(out) == 4:
            print('I think that sign language letter is: H')
        elif np.argmax(out) == 5:
            print('I think that sign language letter is: E')
        elif np.argmax(out) == 6:
            print('I think that sign language letter is: D')

        toFile.toFile(np.argmax(out))

        return out

    #Training Method(array of training images, array of training labels, learning rate)
    def train(im, label, lr = .005):
        out, loss, acc = forward(im, label)

        gradient = np.zeros(7)
        gradient[label] = -1 / out[label]
        gradient = softmax.backprop(gradient, lr)
        gradient = pool.backprop(gradient)
        gradient = conv.backprop(gradient, lr)

        return loss, acc

    print('DeepASL CNN Initialized')

    #Train CNN for (args) epochs
    for epoch in range(intCountOfEpochs):
        print('--Epoch %d ---' % (epoch + 1))

        #Initialize Training Data Into Random Order
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

    loss = 0
    num_correct = 0

    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print('[Step %d] Past 100 Steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct))
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

    #Input To Save Model
    saveTrainingModelOp = input("Do You Want To Save This Model: ")

    #Conditional To Save The Weights, Biases, and Filters
    if saveTrainingModelOp == 'Y':
        with open('NominalWeights.txt', 'wb') as f:
            pickle.dump(softmax.weights, f)

        with open('NominalBiases.txt', 'wb') as d:
            pickle.dump(softmax.biases, d)

        with open('NominalFilters.txt', 'wb') as e:
            pickle.dump(conv.filters, e)


    else:
        print("Did Not Pickle")

    #Tesing Of The CNN Model
    def test():
        print('\n--- Testing the CNN ---')
        loss = 0
        num_correct = 0

        #Directories of Bulk Image Testing
        testing_images_dir_a = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/test_images_a_resized'
        testing_images_dir_l = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/test_images_l_resized'
        live_image_testing = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/specific_testing_image'
        test_images_list = []

        #Iteration Through Testing Image Directory, Resizing, Grayscaling and Appending to List
        for filename in os.listdir(live_image_testing):
            #Skip Metadata
            if filename != '.DS_Store':
                #Read Image
                readimg = cv2.imread(live_image_testing + '/' + filename)
                #Resize Image To 28x28
                resize = cv2.resize(readimg, dsize = (28, 28), interpolation = cv2.INTER_AREA)
                #Grayscale Image
                grayscale = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
                #Append Image to List
                test_images_list.append(grayscale)

        #Convert Test Images List Into Numpy Array
        test_images = np.array(test_images_list)

        #Perform Forward Pass With Set Weights, Biases, And Filters From Training
        for im in test_images:
            #Takes In A Single For Testing
            _ = forwardForTest(im)

        #Once Each Test Is Complete Erase Testing Image Form Directory
        print('Deleting Previous Test Image')

        for toBeRemoved in os.listdir(live_image_testing):
            os.remove(live_image_testing + '/' + toBeRemoved)

    capture = cv2.VideoCapture(0)
    capture.set(3, 1920)
    img_counter = 0

    #Skin Color For Hand Segmentation
    skinLower = (0, 58, 50)
    skinUpper = (30, 255, 255)

    pts = deque(maxlen=2)

    rows, cols = (2, 4)
    rows2, cols2 = (2, 2)
    rows3, cols3 = (2, 2)

    arr = [[0]*cols]*rows
    arr2 = [[0]*cols]*rows

    radius = []
    timeDown = 1

    #Calculation Of Distances Between All Four Corners And The Center Of The Hand Using Distance Formula
    def calculate_centroid_distances(center):

        upper_right_distance = math.sqrt((math.pow(center[0] - 600, 2)) + (math.pow(center[1] - 0, 2)))

        upper_left_distance = math.sqrt((math.pow(center[0] - 0, 2)) + (math.pow(center[1] - 0, 2)))

        lower_left_distance = math.sqrt((math.pow(center[0] - 0, 2)) + (math.pow(center[1] - 600, 2)))

        lower_right_distance = math.sqrt((math.pow(center[0] - 600, 2)) + (math.pow(center[1] - 600, 2)))

        distances = [upper_right_distance, upper_left_distance, lower_left_distance, lower_right_distance]

        return distances

    #Calculation Of Centroid Velocity
    def centroid_velocity(point1, point2):
        #Input Is An Old Point and A New Point
        #Velocity Is Calculated Using The Formula, V = D / T
        #D Being The Distance Between The New Position Of The Centroid and The Previous Position
        #T Being The Amount Of Time Betwen The Position Change Defualt Set To 1
        distance = math.sqrt((math.pow(point2[0] - point1[0], 2)) + (math.pow(point2[1] - point1[1] , 2)))

        timeVeloc = float(1)

        velocity = distance / timeVeloc

        return velocity


    #While Loop
    while True:

        if timeDown == 1:
            iter = 0
        elif timeDown == 2:
            iter = 1
            timeDown = 0

        ret, frame = capture.read()

        cv2.rectangle(frame, (300, 300), (600, 600), (0, 0, 0), 2)

        og = frame[300 : 600, 300 : 600]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 58, 50], dtype = "uint8")
        upper_skin = np.array([30, 255, 255], dtype = "uint8")

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        maskROI = mask[300 : 600, 300 : 600]

        result = cv2.bitwise_and(frame, frame, mask = mask)

        frame2 = imutils.resize(og, width=600)

        blurred = cv2.GaussianBlur(frame2, (11, 11), 0)

        hsv2 = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask2 = cv2.inRange(hsv2, skinLower, skinUpper)

        mask2 = cv2.erode(mask2, None, iterations=2)

        mask2 = cv2.dilate(mask2, None, iterations=2)

        cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        center = None

        distance = None

        if len(cnts) > 0:

            c = max(cnts, key=cv2.contourArea)

            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                distance = calculate_centroid_distances(center)
                arr[iter] = distance
                arr2[iter] = center
                velocityText = centroid_velocity(arr2[0], arr2[1])

                cv2.circle(frame2, (int(x), int(y)), int(radius), (255, 255, 255), 2)
                cv2.circle(frame2, center, 5, (0, 0, 0), -1)
                cv2.line(frame2, center, (0, 0), (255, 255, 255), 3)
                cv2.line(frame2, center, (600, 0), (255, 255, 255), 3)
                cv2.line(frame2, center, (0, 600), (255, 255, 255), 3)
                cv2.line(frame2, center, (600, 600), (255, 255, 255), 3)

                cv2.putText(frame2, "TRC:CENTROID", (480, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame2, str(int(distance[0])), (525, 45), cv2.FONT_HERSHEY_PLAIN , 1, (255, 255, 255), 2)

                cv2.putText(frame2, "TLC:CENTROID", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame2, str(int(distance[1])), (30, 45), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                cv2.putText(frame2, "BLC:CENTROID", (10, 505), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame2, str(int(distance[2])), (30, 525), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                cv2.putText(frame2, "BRC:CENTROID", (460, 505), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame2, str(int(distance[3])), (525, 525), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                cv2.putText(frame2, "Centroid Velocity", (250, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame2, str(int(velocityText)), (300, 45), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                timeDown += 1

        if timeDown == 2:
            difference = [a_i - b_i for a_i, b_i in zip(arr[1], arr[0])]
            velocity = centroid_velocity(arr2[0], arr2[1])

            onTest = 0


            for dif in difference:
                if abs(dif) <= 0.02 and velocity <= 0.2:
                    if onTest == 0:
                        img_name = "/Users/cesaralmendarez/Desktop/DeepASL/test_images/specific_testing_image/opencv_frame_{}.png".format(img_counter)
                        cv2.imwrite(img_name, maskROI)
                        ("{} writter!".format(img_counter))
                        test()
                        time.sleep(1.0)
                        onTest += 1
                    elif onTest != 0:
                        break

                img_counter += 1


        cv2.imshow('Orginal Frame', frame2)
        cv2.imshow('Final', maskROI)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#############################################################################################################################
#############################################################################################################################
elif toTrainOrNot == 'N':
    def forwardForTest(image):

        out = conv2.forward((image / 255) - 0.5)

        out = pool.forward(out)

        out = softmax2.forward(out)

        print("Network Prediction: ")
        if np.argmax(out) == 0:
            ('I think that sign language letter is: A')
        elif np.argmax(out) == 1:
            ('I think that sign language letter is: L')
        elif np.argmax(out) == 2:
            ('I think that sign language letter is: B')
        elif np.argmax(out) == 3:
            ('I think that sign language letter is: C')
        elif np.argmax(out) == 4:
            print('I think that sign language letter is: H')
        elif np.argmax(out) == 5:
            print('I think that sign language letter is: E')
        elif np.argmax(out) == 6:
            print('I think that sign language letter is: D')

        toFile.toFile(np.argmax(out))

        return out

    def test():
        #Test CNN
        print('\n--- Testing the CNN ---')
        loss = 0
        num_correct = 0

        #Directories of Bulk Image Testing
        testing_images_dir_a = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/test_images_a_resized'
        testing_images_dir_l = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/test_images_l_resized'
        live_image_testing = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/specific_testing_image'
        test_images_list = []

        #Iteration Through Testing Image Directory, Resizing, Grayscaling and Appending to List
        for filename in os.listdir(live_image_testing):
            #Skip Metadata
            if filename != '.DS_Store':
                #Read Image
                readimg = cv2.imread(live_image_testing + '/' + filename)
                #Resize Image To 28x28
                resize = cv2.resize(readimg, dsize = (28, 28), interpolation = cv2.INTER_AREA)
                #Grayscale Image
                grayscale = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
                #Append Image to List
                test_images_list.append(grayscale)

        #Convert List into Numpy Array
        test_images = np.array(test_images_list)

        for im in test_images:
            _ = forwardForTest(im)

        print('Deleting Previous Test Image')

        for toBeRemoved in os.listdir(live_image_testing):
            os.remove(live_image_testing + '/' + toBeRemoved)

    capture = cv2.VideoCapture(0)
    capture.set(3, 1920)
    img_counter = 0
    skinLower = (0, 58, 50)
    skinUpper = (30, 255, 255)
    pts = deque(maxlen=2)
    rows, cols = (2, 4)
    rows2, cols2 = (2, 2)
    rows3, cols3 = (2, 2)
    arr = [[0]*cols]*rows
    arr2 = [[0]*cols]*rows
    radius = []
    timeDown = 1

    def calculate_centroid_distances(center):

        upper_right_distance = math.sqrt((math.pow(center[0] - 600, 2)) + (math.pow(center[1] - 0, 2)))

        upper_left_distance = math.sqrt((math.pow(center[0] - 0, 2)) + (math.pow(center[1] - 0, 2)))

        lower_left_distance = math.sqrt((math.pow(center[0] - 0, 2)) + (math.pow(center[1] - 600, 2)))

        lower_right_distance = math.sqrt((math.pow(center[0] - 600, 2)) + (math.pow(center[1] - 600, 2)))

        distances = [upper_right_distance, upper_left_distance, lower_left_distance, lower_right_distance]

        return distances

    def centroid_velocity(point1, point2):
        #Input Is An Old Point and A New Point
        #Velocity Is Calculated Using The Formula, V = D / T
        #D Being The Distance Between The New Position Of The Centroid and The Previous Position
        #T Being The Amount Of Time Betwen The Position Change Defualt Set To 1
        distance = math.sqrt((math.pow(point2[0] - point1[0], 2)) + (math.pow(point2[1] - point1[1] , 2)))

        timeVeloc = float(1)

        velocity = distance / timeVeloc

        return velocity



    while True:

        if timeDown == 1:
            iter = 0
        elif timeDown == 2:
            iter = 1
            timeDown = 0

        ret, frame = capture.read()

        cv2.rectangle(frame, (300, 300), (600, 600), (0, 0, 0), 2)

        og = frame[300 : 600, 300 : 600]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 58, 50], dtype = "uint8")
        upper_skin = np.array([30, 255, 255], dtype = "uint8")

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        maskROI = mask[300 : 600, 300 : 600]

        result = cv2.bitwise_and(frame, frame, mask = mask)

        frame2 = imutils.resize(og, width=600)

        blurred = cv2.GaussianBlur(frame2, (11, 11), 0)

        hsv2 = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask2 = cv2.inRange(hsv2, skinLower, skinUpper)

        mask2 = cv2.erode(mask2, None, iterations=2)

        mask2 = cv2.dilate(mask2, None, iterations=2)

        cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        center = None

        distance = None

        if len(cnts) > 0:

            c = max(cnts, key=cv2.contourArea)

            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                distance = calculate_centroid_distances(center)
                arr[iter] = distance
                arr2[iter] = center
                velocityText = centroid_velocity(arr2[0], arr2[1])

                #cv2.circle(frame2, (int(x), int(y)), int(radius), (255, 255, 255), 2)
                cv2.circle(frame2, center, 5, (0, 0, 0), -1)
                cv2.line(frame2, center, (0, 0), (255, 255, 255), 3)
                cv2.line(frame2, center, (600, 0), (255, 255, 255), 3)
                cv2.line(frame2, center, (0, 600), (255, 255, 255), 3)
                cv2.line(frame2, center, (600, 600), (255, 255, 255), 3)
                cv2.circle(frame2, (int(x), int(y)), int(radius), (255, 255, 255), 2)

                cv2.putText(frame2, "TRC:CENTROID", (480, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame2, str(int(distance[0])), (525, 45), cv2.FONT_HERSHEY_PLAIN , 1, (255, 255, 255), 2)

                cv2.putText(frame2, "TLC:CENTROID", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame2, str(int(distance[1])), (30, 45), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                cv2.putText(frame2, "BLC:CENTROID", (10, 505), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame2, str(int(distance[2])), (30, 525), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                cv2.putText(frame2, "BRC:CENTROID", (460, 505), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame2, str(int(distance[3])), (525, 525), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                cv2.putText(frame2, "Centroid Velocity", (250, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame2, str(int(velocityText)), (300, 45), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                timeDown += 1

        if timeDown == 2:
            difference = [a_i - b_i for a_i, b_i in zip(arr[1], arr[0])]
            velocity = centroid_velocity(arr2[0], arr2[1])

            onTest = 0


            for dif in difference:
                if abs(dif) <= 0.02 and velocity <= 0.2:
                    if onTest == 0:
                        img_name = "/Users/cesaralmendarez/Desktop/DeepASL/test_images/specific_testing_image/opencv_frame_{}.png".format(img_counter)
                        cv2.imwrite(img_name, maskROI)
                        print("{} writter!".format(img_counter))
                        test()
                        time.sleep(1.0)
                        onTest += 1
                    elif onTest != 0:
                        break



                img_counter += 1


        cv2.imshow('Orginal Frame', frame2)
        cv2.imshow('Final', maskROI)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
elif toTrainOrNot == 'C':
    f = open('NominalBiases.txt', 'r+')
    f.truncate(0)
    t = open('NominalWeights.txt', 'r+')
    t.truncate(0)
    e = open('NominalFilters.txt', 'r+')
    e.truncate(0)
