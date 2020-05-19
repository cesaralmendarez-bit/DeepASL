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
from training_images_fetcher import DataGetter
from testing_images_fetcher import test_fetcher
from send_sms import trainNotification
from tts import sayLetter
import threading
import time
import threading
from threading import Timer
from writingToTxtFile import writeToFile
np.set_printoptions(threshold=sys.maxsize)


conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13 * 13 * 8, 5)
dataGetterClass = DataGetter
testimgGetter = test_fetcher
trainingNotify = trainNotification
tts = sayLetter
toFile = writeToFile

#Training Image Dataset Directories
train_images_dir_a = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_a_2'
train_images_dir_l = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_l_2'
train_images_dir_b = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_b_2'
train_images_dir_c = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_c_2'
train_images_dir_h = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_h_2'
halfset_train_images_dir_a = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/halfSetA'
halfset_train_images_dir_l = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/halfSetL'
halfset_train_images_dir_b = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/halfSetB'
halfset_train_images_dir_c = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/halfSetC'
min_a_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minA'
min_l_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minL'
min_b_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minB'
min_c_dir = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/minC'
train_images_list = []

#Resizing and Grayscaling of Training Images and Appending Into List
#A
for filename in os.listdir(train_images_dir_a):
    #Do Not Read Metadata
    if filename != '.DS_Store':
        readimg = cv2.imread(train_images_dir_a + '/' + filename)

        resized = cv2.resize(readimg, dsize=(28, 28), interpolation = cv2.INTER_AREA)

        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        train_images_list.append(grayscale)
#L
for filename2 in os.listdir(train_images_dir_l):

    if filename2 != '.DS_Store':
        readimg2 = cv2.imread(train_images_dir_l + '/' + filename2)

        resized2 = cv2.resize(readimg2, dsize=(28, 28), interpolation = cv2.INTER_AREA)

        grayscale2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)

        train_images_list.append(grayscale2)
#B
for filename3 in os.listdir(train_images_dir_b):

    if filename3 != '.DS_Store':
        readimg3 = cv2.imread(train_images_dir_b + '/' + filename3)

        resized3 = cv2.resize(readimg3, dsize=(28, 28), interpolation = cv2.INTER_AREA)

        grayscale3 = cv2.cvtColor(resized3, cv2.COLOR_BGR2GRAY)

        train_images_list.append(grayscale3)
#C
for filename4 in os.listdir(train_images_dir_c):

    if filename4 != '.DS_Store':
        readimg4 = cv2.imread(train_images_dir_c + '/' + filename4)

        resized4 = cv2.resize(readimg4, dsize=(28, 28), interpolation = cv2.INTER_AREA)

        grayscale4 = cv2.cvtColor(resized4, cv2.COLOR_BGR2GRAY)

        train_images_list.append(grayscale4)

#H
for filename5 in os.listdir(train_images_dir_h):
    #Do Not Read Metadata
    if filename5 != '.DS_Store':
        readimg5 = cv2.imread(train_images_dir_h + '/' + filename5)

         resized5 = cv2.resize(readimg5, dsize=(28, 28), interpolation = cv2.INTER_AREA)

        grayscale5 = cv2.cvtColor(resized5, cv2.COLOR_BGR2GRAY)

        train_images_list.append(grayscale5)

#Convert List To Numpy Array
train_images = np.array(train_images_list)
print(len(train_images))

'''
Create Training Labels
1D Array, Labels Each Have A Set Number Of Repeats
0 = A, 1 = L
'''
zeros = np.full((1, 11000), 0)
ones = np.full((1, 11000), 1)
twos = np.full((1, 11000), 2)
threes = np.full((1, 11000), 3)
fours = np.full((1, 11000), 4)
finalZeros = zeros.ravel()
finalOnes = ones.ravel()
finalTwos = twos.ravel()
finalThrees = threes.ravel()
finalFours = fours.ravel()

#[0x5000, 1x5000, 2x5000]
train_labels = np.concatenate([finalZeros, finalOnes, finalTwos, finalThrees, finalFours])

print(train_labels)

#def forward(image, label):
def forward(image, label):
    out = conv.forward((image / 255) - 0.5)

    out = pool.forward(out)

    out = softmax.forward(out)

    print('Percents for letters')
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

    print('Eval')
    print(rightOrWrongEval)

    return out, loss, acc

def forwardForTest(image):
    out = conv.forward((image / 255) - 0.5)

    out = pool.forward(out)

    out = softmax.forward(out)

    print('Percents for letters')
    print(out)

    tts.saySomething(np.argmax(out))

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

    toFile.toFile(np.argmax(out))

    return out

def train(im, label, lr = .005):
    out, loss, acc = forward(im, label)

    gradient = np.zeros(5)
    gradient[label] = -1 / out[label]
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc

print('DeepASL CNN Initialized')

#Train CNN for (args) epochs
for epoch in range(1000):
    print('--Epoch %d ---' % (epoch + 1))

    #Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

loss = 0
num_correct = 0

for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i > 0 and i % 100 == 99:
        print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct))
        loss = 0
        num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc

trainingNotify.sendsms()

def test():
    #Image Capture
    #testimgGetter.test_image_getter()

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
img_counter = 0
start_time = time.time()

while True:
    ret, frame = capture.read()

    cv2.rectangle(frame, (300, 300), (600, 600), (0, 0, 0), 2)

    og = frame[300 : 600, 300 : 600]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 58, 50], dtype = "uint8")
    upper_skin = np.array([30, 255, 255], dtype = "uint8")

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    maskROI = mask[300 : 600, 300 : 600]

    result = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow('Original Frame', og)

    cv2.imshow('Final', maskROI)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time.time() - start_time >= 10: #<---- Check if 5 sec passed
        img_name = "/Users/cesaralmendarez/Desktop/DeepASL/test_images/specific_testing_image/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, maskROI)
        print("{} written!".format(img_counter))
        test()
        start_time = time.time()

    img_counter += 1
