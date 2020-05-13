from training_images_fetcher import DataGetter

dataGetterClass = DataGetter

trainNewGesture = input('Create a new gesture(n) or add training images(a)?')

if trainNewGesture == 'a':
    dataGetterClass.image_taker()
elif trainNewGesture == 'n':
    newGestureTrainRepoLetter = input('What new gesture do you want to make?')
    dataGetterClass.createNewGestureRepo(newGestureTrainRepoLetter)

    addToNewDir = input('Do you want to add training images?(y/n)')

    if addToNewDir == 'y':
        dataGetterClass.image_taker()
    elif addToNewDir == 'n':
        print('Quitting...')
