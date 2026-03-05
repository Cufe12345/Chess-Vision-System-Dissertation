import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
def trainModel(x,y,epochs=20):

    # --- Model Definition ---
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # binary output
    ])

    # --- Compile ---
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # --- Train ---
    model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.2, shuffle=True)
    return model


def trainingData():
    with open("training_dataRed.pkl", "rb") as f:
        training_dataRed = pickle.load(f)
    with open("training_dataZenodo.pkl", "rb") as f:
        training_dataZenodo = pickle.load(f)
    with open("training_dataMy.pkl", "rb") as f:
        training_dataMy = pickle.load(f)

    return training_dataRed, training_dataZenodo, training_dataMy

def formatTrainingData(training_dataRed, training_dataZenodo, training_dataMy):
    format1 = training_dataRed + training_dataMy
    format2 = training_dataZenodo

    x = []
    y = []
    piece_to_int = {
        '': 0,   # empty
        'P': 1,   # white pawn
        'N': 2,   # white knight
        'B': 3,   # white bishop
        'R': 4,   # white rook
        'Q': 5,   # white queen
        'K': 6,   # white king
        'p': 7,   # black pawn
        'n': 8,
        'b': 9,
        'r': 10,
        'q': 11,
        'k': 12
    }
    for image in format1:
        squares, labels = image
        for sq, label in zip(squares, labels):
            resized_sq = cv2.resize(sq, (64, 64))
            grey_sq = cv2.cvtColor(resized_sq, cv2.COLOR_BGR2GRAY)

            # if first:
            #     plt.imshow(grey_sq, cmap='gray')
            #     plt.title(f"Label: {labels[index]}")
            #     plt.axis('off')
            #     plt.show()
            #     first = False
            # if len(squares) -1 == index:
            #     plt.imshow(grey_sq, cmap='gray')
            #     plt.title(f"Label: {labels[index]}")
            #     plt.axis('off')
            #     plt.show()
            norm_sq = grey_sq / 255.0
            norm_sq = norm_sq[...,None] # add channel dimension
            x.append(norm_sq)
            label = piece_to_int[label]
            if label != 0:
                label = 1
            y.append(label)
    for image in format2:
        square, label = image
        resized_sq = cv2.resize(square, (64, 64))
        grey_sq = cv2.cvtColor(resized_sq, cv2.COLOR_BGR2GRAY)
        norm_sq = grey_sq / 255.0
        norm_sq = norm_sq[...,None] # add channel dimension
        x.append(norm_sq)
        if label != 0:
            label = 1
        y.append(label)

    x = np.array(x, dtype="float32")
    y = np.array(y, dtype="int32")
    return x,y

trainingDataRed, trainingDataZenodo, trainingDataMy = trainingData()
x,y = formatTrainingData(trainingDataRed, trainingDataZenodo, trainingDataMy)
model = trainModel(x,y,epochs=20)
model.save("binary_classification_model.h5")
model.save_weights("binary_classification_weights.h5")