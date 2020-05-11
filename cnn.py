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
np.set_printoptions(threshold=sys.maxsize)

conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13 * 13 * 8, 2)
dataGetterClass = DataGetter
testimgGetter = test_fetcher

#Training Image Dataset Directories
train_images_dir_a = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_a_2'
train_images_dir_l = '/Users/cesaralmendarez/Desktop/DeepASL/train_images/train_images_l_2'

train_images_list = []

#Resizing and Grayscaling of Training Images and Appending Into List
for filename in os.listdir(train_images_dir_a):
    #Do Not Read Metadata
    if filename != '.DS_Store':
        readimg = cv2.imread(train_images_dir_a + '/' + filename)

        resized = cv2.resize(readimg, dsize=(28, 28), interpolation = cv2.INTER_AREA)

        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        train_images_list.append(grayscale)

for filename2 in os.listdir(train_images_dir_l):

    if filename2 != '.DS_Store':
        readimg2 = cv2.imread(train_images_dir_l + '/' + filename2)

        resized2 = cv2.resize(readimg2, dsize=(28, 28), interpolation = cv2.INTER_AREA)

        grayscale2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)

        train_images_list.append(grayscale2)

#Convert List To Numpy Array
train_images = np.array(train_images_list)

'''
Create Training Labels
1D Array, Labels Each Have A Set Number Of Repeats
0 = A, 1 = L
'''
zeros = np.full((1, 250), 0)
ones = np.full((1, 250), 1)
finalZeros = zeros.ravel()
finalOnes = ones.ravel()

#[0x250, 1x250]
train_labels = np.concatenate([finalZeros, finalOnes])

def forward(image, label):
    out = conv.forward((image / 255) - 0.5)

    out = pool.forward(out)

    out = softmax.forward(out)

    loss = -np.log(out[label])

    acc = 1 if np.argmax(out) == label else 0

    rightOrWrongEval = 'Correct Prediction' if np.argmax(out) == label else 'Incorrect Prediction'

    print("Network Prediction: ")
    if np.argmax(out) == 0:
        print('I think that sign language letter is: A')
    elif np.argmax(out) == 1:
        print('I think that sign language letter is: L')

    return out, loss, acc

def train(im, label, lr = .005):
    out, loss, acc = forward(im, label)

    gradient = np.zeros(2)

    gradient[label] = -1 / out[label]

    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc

print('DeepASL CNN Initialized')

#Train CNN for (args) epochs
for epoch in range(5000):
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

#Image Capture
testimgGetter.test_image_getter()

#Test CNN
print('\n--- Testing the CNN ---')

loss = 0
num_correct = 0

#Directories of Bulk Image Testing
testing_images_dir_a = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/test_images_a_resized'
testing_images_dir_l = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/test_images_l_resized'
live_image_testing = '/Users/cesaralmendarez/Desktop/DeepASL/test_images/specific_testing_image'
test_images_list = []
'''
for filename in os.listdir(testing_images_dir_a):
     #do not read .DS_Store file
     if filename != ".DS_Store":
         #read each image in directory a
         readimg = cv2.imread(testing_images_dir_a + '/' + filename)
         #convert each resized image in directory a to grayscale
         grayreadimg = cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
         #append each image(2x2) into 3D list
         test_images_list.append(grayreadimg)
'''
'''
for filename2 in os.listdir(testing_images_dir_l):
    #do not read .DS_Store file
     if filename2 != ".DS_Store":
         #read each image in directory l
         readimg2 = cv2.imread(testing_images_dir_l + '/' + filename2)
         #convert each resized image in directory a to grayscale
         grayreadimg2 = cv2.cvtColor(readimg2, cv2.COLOR_BGR2GRAY)
         #append each image(2x2) into 3D list
         test_images_list.append(grayreadimg2)
'''
#Iterationn Through Testing Image Directory, Resizing, Grayscaling and Appending to List
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

#Test Labels
#0 = a, 1 = l
zeros = np.full((1, 250), 0)
ones = np.full((1, 250), 1)
finalZeros = zeros.ravel()
finalOnes = ones.ravel()

#Prompt For Label
label = input('Enter Label for the Image...')

#Convertion of Input Into Readable Data
if label == 'a':
    label = 0
elif label == 'l':
    label = 1

#Insertion of Label Into 1D Numpy Array
zeros = np.full((1, 1), label)
ones = np.full((0, 0), 1)
finalZeros = zeros.ravel()
finalOnes = ones.ravel()

test_labels = np.concatenate([finalZeros, finalOnes])

for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += 1
    num_correct += acc

'''
num_tests = len(test_images)
print('Test Loss: ', loss / num_tests)
print('Test Accuracy: ', num_correct / num_tests)
print('Number of Tests:')
print(num_tests)
print('Number of Correct Predictions:')
print(num_correct)
'''
print('Deleting Previous Test Image...')

for toBeRemovedFile in os.listdir(live_image_testing):
    os.remove(live_image_testing + '/' + toBeRemovedFile)
