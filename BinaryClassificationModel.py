import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import BoardSegmentation
from pathlib import Path

def trainModel(x,y,epochs=20):

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.2, shuffle=True)
    return model


def fen_to_labels(fen):
    """
    Convert a fen to a list of 64 labels representing the piece on the board
    """
    
    # Keep only the board part
    board_part = fen.split(',')[0]

    # Split into 8 rows
    rows = board_part.split('_')
    if len(rows) != 8:
        raise ValueError(f"Expected 8 rows, got {len(rows)} in {fen}")

    labels = []

    rows = rows[::-1]


    for row in rows:
        row = row[::-1]
        for char in row:
            if char.isdigit():
                labels.extend([''] * int(char))
            else:
                labels.append(char)
    
    if len(labels) != 64:
        raise ValueError(f"FEN did not expand to 64 squares, got {len(labels)}")
    
    return labels

def get_chessRedImages():
    """Load chessRed images
    """
    base_dir = Path(__file__).parent
    folder_path = base_dir / "trainingData" / "chessRed" / "FinalImages"
    folder_path2 = Path("trainingData") / "fillers"
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]

    image_files += [
        os.path.join(folder_path2, f)
        for f in os.listdir(folder_path2)
        if f.lower().endswith(image_extensions)
    ]

    images = []
    for i, image in enumerate(image_files):
        try:
            squares = BoardSegmentation.getSquaresFromImage(image,farChessTable=True,noModifySquares=True)
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            images.append((squares, labels))
        except Exception as e:
            continue
    return images


def get_myImages():
    """Load my images"""
    base_dir = Path(__file__).parent / "trainingData"
    path_opening = base_dir / "opening"
    path_midgame = base_dir / "midgame"
    path_endgame = base_dir / "endgame"

    image_filesO = [
    os.path.join(path_opening, f)
    for f in os.listdir(path_opening)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]

    image_filesM = [
        os.path.join(path_midgame, f)
        for f in os.listdir(path_midgame)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]

    image_filesE = [
        os.path.join(path_endgame, f)
        for f in os.listdir(path_endgame)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]


    image_files = image_filesO + image_filesM + image_filesE

    final_images = []
    for i, image in enumerate(image_files):
        try:
            squares = BoardSegmentation.getSquaresFromImage(image,noModifySquares=True)
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            # labels values need remapping as these images are side on
            remapped_labels = []
            for j in range(8):
                for i in range(7,-1,-1):
                    remapped_labels.append(labels[i*8 + j])

            final_images.append((squares, remapped_labels))
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    return final_images

def trainingData():
    print("Loading my images...")
    imagesMy = get_myImages()
    print(f"Loaded {len(imagesMy)} of my images.")
    
    print("Loading chessRed images...")
    imagesRed = get_chessRedImages()
    print(f"Loaded {len(imagesRed)} chessRed images.")

    return imagesRed, imagesMy

def formatTrainingData(training_dataRed, training_dataMy):
    format1 = training_dataRed + training_dataMy

    x = []
    y = []
    piece_to_int = {
        '': 0, 
        'P': 1, 
        'N': 2,
        'B': 3,  
        'R': 4,   
        'Q': 5,   
        'K': 6,   
        'p': 7,   
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
            norm_sq = grey_sq / 255.0
            norm_sq = norm_sq[...,None]
            x.append(norm_sq)
            label = piece_to_int[label]
            if label != 0:
                label = 1
            y.append(label)
    x = np.array(x, dtype="float32")
    y = np.array(y, dtype="int32")
    return x,y

def trainAndSaveModel():
    print("Training Binary Classification Model...")
    trainingDataRed, trainingDataMy = trainingData()
    x,y = formatTrainingData(trainingDataRed, trainingDataMy)
    model = trainModel(x,y,epochs=20)
    model.save("binary_classification_model.h5")
    model.save_weights("binary_classification_weights.h5")
    print("Binary Classification Model Trained and Saved!")