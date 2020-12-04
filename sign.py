# Import Statements
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import cv2
import matplotlib.pyplot as plt

# Loading The Model
def loadModel():
    return keras.models.load_model("./web/model_v1")

# Constants
MODEL = loadModel()
CAPTURE = cv2.VideoCapture(0)

test_x = pd.read_csv("./sign_mnist_test/sign_mnist_test.csv").drop(["label"], axis=1)
test_y = pd.read_csv("./sign_mnist_test/sign_mnist_test.csv")["label"].values
testimg = test_x.values[567].reshape(28, 28)
# plt.imshow(testimg)
# plt.xlabel(np.argmax(MODEL.predict(testimg.reshape(-1, 28, 28, 1)), axis=1))
# plt.show()

labels = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z"
}

datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)

while CAPTURE.isOpened():
    ret, frame = CAPTURE.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        test_frame = frame[100:300, 100:300]
        cv2.imshow('test_image', test_frame)
        gray = cv2.cvtColor(cv2.resize(test_frame, (28, 28)), cv2.COLOR_BGR2GRAY)
        reshaped_frame = cv2.resize(gray, (28, 28))
        (thresh, blackAndWhite) = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        blackAndWhite = cv2.resize(blackAndWhite, (28, 28))
        print("Predicted: ", labels[MODEL.predict_classes(reshaped_frame.reshape(-1, 28, 28, 1))[0]], "Predicted Class ", MODEL.predict_classes(reshaped_frame.reshape(-1, 28, 28, 1))[0])
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

CAPTURE.release()
cv2.destroyAllWindows()