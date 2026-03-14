import BoardSegmentation
import os
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models 
import numpy as np
import pickle
from tensorflow.keras.applications import DenseNet121
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import math
from collections import Counter

import Visual_Representation

def recognizePieces(squares):
    
    piece_labels = []
    for sq in squares:
        # Placeholder for piece recognition logic
        piece_labels.append("Unknown")  # Replace with actual recognition result
    
    return piece_labels

def fen_to_labels(fen):
    """
    Convert FEN string to list of 64 square labels, top-left -> bottom-right.
    Empty squares are labeled as '1' or 'empty', pieces as letters.
    """
    
     # Step 1: Keep only the board part (before first comma)
    board_part = fen.split(',')[0]

    # Step 2: Split into 8 rows
    rows = board_part.split('_')
    if len(rows) != 8:
        raise ValueError(f"Expected 8 rows, got {len(rows)} in {fen}")

    labels = []

    rows = rows[::-1]


    for row in rows:
        row = row[::-1]
        for char in row:
            if char.isdigit():
                labels.extend([''] * int(char))  # '' for empty
            else:
                labels.append(char)
    
    if len(labels) != 64:
        raise ValueError(f"FEN did not expand to 64 squares, got {len(labels)}")
    
    return labels

def get_chessRedImages(colour):
    folder_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\chessRed\\FinalImages"
    folder_path2 = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\fillers"
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
        print(f"Processing image {i+1}/{len(image_files)}: {image}")
        try:
            squares = BoardSegmentation.getSquaresFromImage(image, colour=colour,farChessTable=True)
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            images.append((squares, labels))

            # for testing purpose only get 3
            # if len(images) >= 3:
            #     break
            # print(f"Piece labels for {image}: {piece_labels}")
        except Exception as e:
            # print(f"Error processing {image}: {e}")
            continue
    
    #check if each square with non empty label has a piece in it
    for squares, labels in images:
        for sq, label in zip(squares, labels):
            if label != '' and np.mean(sq) < 10:  # Assuming empty squares are mostly black
                print(f"Warning: Label '{label}' has an empty square with mean pixel value {np.mean(sq)}")
                plt.imshow(sq)
                plt.title(f"Label: {label}")
                plt.axis('off')
                plt.show()

            # #show the square and its label for testing
            # if label != '':
            #     plt.imshow(sq)
            #     plt.title(f"Label: {label}")
            #     plt.axis('off')
            #     plt.show()

    return images


def get_zenodoImages():
    pathTrain = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\zenodo\\chess_pieces\\train"
    pathValid = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\zenodo\\chess_pieces\\valid"

    #get all folders in pathTrain and pathValid
    foldersTrain = [os.path.join(pathTrain, f) for f in os.listdir(pathTrain) if os.path.isdir(os.path.join(pathTrain, f))]
    foldersValid = [os.path.join(pathValid, f) for f in os.listdir(pathValid) if os.path.isdir(os.path.join(pathValid, f))]

    images = []
    for folder in foldersTrain + foldersValid:
        label = os.path.basename(folder)
        label = int(label)
        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append((img, label))
                else:
                    print(f"Warning: Could not read image {img_path}")
    
    return images


def get_myImages(colour):
    path_opening = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\opening"
    path_midgame = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\midgame"
    path_endgame = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\endgame"

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
        print(f"Processing image {i+1}/{len(image_files)}: {image}")
        try:
            squares = BoardSegmentation.getSquaresFromImage(image, colour=colour)
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            # labels values need remapping as these images are side on
            remapped_labels = []
            for j in range(8):
                for i in range(7,-1,-1):
                    remapped_labels.append(labels[i*8 + j])

            # if(image == "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\opening\\1k1rq2r_1pp1bp2_p1N2n2_n2P1Qp1_3p3p_P5B1_1PPN1PPP_R3R1K1,b,-,-,3,18.jpg"):
            #     print(f"Labels for {image}: {remapped_labels}")
            #     for sq, label in zip(squares, remapped_labels):
            #         plt.imshow(sq)
            #         plt.title(f"Label: {label}")
            #         plt.axis('off')
            #         plt.show()
            final_images.append((squares, remapped_labels))
            print(f"Labels for {image}: {remapped_labels}")
        except Exception as e:
            print(f"Error processing {image}: {e}")
            continue
    
    return final_images

def get_testingImages(colour):
    path_testing = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\testingImages"


    image_files = [
    os.path.join(path_testing, f)
    for f in os.listdir(path_testing)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]



    final_images = []
    for i, image in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image}")
        try:
            squares = BoardSegmentation.getSquaresFromImage(image, colour=colour)
            labels = fen_to_labels(os.path.basename(image).split('.')[0])

            # labels values need remapping as these images are side on
            remapped_labels = []
            for j in range(8):
                for i in range(7,-1,-1):
                    remapped_labels.append(labels[i*8 + j])

            final_images.append((squares, remapped_labels))
            print(f"Labels for {image}: {remapped_labels}")
        except Exception as e:
            print(f"Error processing {image}: {e}")
            continue
    
    return final_images
    
def getTrainingData(colour):
    # Placeholder for loading and preprocessing training data

    print("Loading my images...")
    imagesMy = get_myImages(colour)
    print(f"Loaded {len(imagesMy)} of my images.")
    
    print("Loading chessRed images...")
    imagesRed = get_chessRedImages(colour)
    # imagesRed = []
    print(f"Loaded {len(imagesRed)} chessRed images.")

    print("Loading zenodo images...")
    imagesZenodo = get_zenodoImages()
    print(f"Loaded {len(imagesZenodo)} zenodo images.")

    #save the training data for later use
    with open("training_dataRed.pkl", "wb") as f:
        pickle.dump(imagesRed, f)
    with open("training_dataZenodo.pkl", "wb") as f:
        pickle.dump(imagesZenodo, f)
    with open("training_dataMy.pkl", "wb") as f:
        pickle.dump(imagesMy, f)
    return (imagesRed, imagesZenodo, imagesMy)


def loadTrainingData(colour):

    try:
        with open("training_dataRed.pkl", "rb") as f:
            training_dataRed = pickle.load(f)
        with open("training_dataZenodo.pkl", "rb") as f:
            training_dataZenodo = pickle.load(f)
        with open("training_dataMy.pkl", "rb") as f:
            training_dataMy = pickle.load(f)
    except FileNotFoundError:
        getTrainingData(colour)
        return loadTrainingData(colour)
    return (training_dataRed, training_dataZenodo, training_dataMy)

def dataPreprocessingMy(data,data_augmentation=False,colour=False):
    # Placeholder for data preprocessing logic

    new_data = []

    augmentation_pipeline = A.Compose([
    A.Rotate(limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
    A.GaussNoise(std_range=(0.05, 0.1), p=0.2),
    A.ImageCompression(quality_range=(60, 100), p=0.3),
    A.GaussianBlur(blur_limit=(3,5), p=0.2),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.25),
    A.ElasticTransform(alpha=1, sigma=5, p=0.1),  # subtle shape distortion
    A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.1),
    ])
    
    NUM_AUGMENTATIONS = 5
    
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


    #Reize images, grey scale, normalise
    for image in data:
        squares, labels = image
        first = True
        firstAug = True
        for index,sq in enumerate(squares):
            if labels[index] == '' and np.random.rand() < 0.5:  # Skip some empty squares to balance the dataset
                continue
            # if labels[index] == 'P' or labels[index] == 'p' and np.random.rand() < 0.7:  # Skip some empty squares to balance the dataset
            #     continue
            resized_sq = sq
            if colour:
                resized_sq = cv2.resize(sq, (128, 128))
            else:
                resized_sq = cv2.resize(sq, (64, 64))
            grey_sq = resized_sq
            if not colour:
                grey_sq = cv2.cvtColor(resized_sq, cv2.COLOR_BGR2GRAY)

            # if first and labels[index] != '':
            #     plt.imshow(grey_sq, cmap='gray')
            #     plt.title(f"Label: {labels[index]}")
            #     plt.axis('off')
            #     plt.show()
            #     first = True
            # if len(squares) -1 == index:
            #     plt.imshow(grey_sq, cmap='gray')
            #     plt.title(f"Label: {labels[index]}")
            #     plt.axis('off')
            #     plt.show()
            norm_sq = grey_sq
            if not colour:
                norm_sq = norm_sq / 255.0
                norm_sq = norm_sq[...,None] # add channel dimension
            new_data.append((norm_sq, piece_to_int[labels[index]]))

            if data_augmentation:
                for _ in range(NUM_AUGMENTATIONS):
                    

                    if labels[index] == '' and np.random.rand() <0.0:  # Skip some empty squares to balance the dataset
                        continue
                        # if labels[count] == 'P' or labels[count] == 'p' and np.random.rand() < 0.7:  # Skip some empty squares to balance the dataset
                        #     count += 1
                        #     continue
                    resized_aug = sq
                    if colour:
                        resized_aug = cv2.resize(sq, (128, 128))
                    else:    
                        resized_aug = cv2.resize(sq, (64, 64))
                    augmented = augmentation_pipeline(image=resized_aug)['image']
                    grey_aug = augmented
                    if not colour:
                        grey_aug = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
                        # if firstAug:
                        #     plt.imshow(grey_aug, cmap='gray')
                        #     plt.title(f"Augmented Label: {labels[count]}")
                        #     plt.axis('off')
                        #     plt.show()
                        #     firstAug = False
                        # if len(squares) -1 == count:
                        #     plt.imshow(grey_aug, cmap='gray')
                        #     plt.title(f"Label: {labels[count]} (Augmented)")
                        #     plt.axis('off')
                        #     plt.show()
                    norm_aug = grey_aug
                    if not colour:
                        norm_aug = norm_aug / 255.0
                        norm_aug = norm_aug[...,None] # add channel dimension
                    new_data.append((norm_aug, piece_to_int[labels[index]]))
    
    return new_data

def dataPreprocessingRed(data,data_augmentation=False,colour=False):
    # Placeholder for data preprocessing logic

    new_data = []

    augmentation_pipeline = A.Compose([
    A.Rotate(limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
    A.GaussNoise(std_range=(0.05, 0.1), p=0.2),
    A.ImageCompression(quality_range=(60, 100), p=0.3),
    A.GaussianBlur(blur_limit=(3,5), p=0.2),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.25),
    A.ElasticTransform(alpha=1, sigma=5, p=0.1),  # subtle shape distortion
    A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.1),
])
    
    NUM_AUGMENTATIONS = 5
    
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


    #Reize images, grey scale, normalise
    for image in data:
        squares, labels = image
        first = True
        firstAug = True
        for index,sq in enumerate(squares):
            if labels[index] == '' and np.random.rand() < 0.5:  # Skip some empty squares to balance the dataset
                continue
            # if labels[index] == 'P' or labels[index] == 'p' and np.random.rand() < 0.7:  # Skip some empty squares to balance the dataset
            #     continue
            resized_sq = sq
            if colour:
                resized_sq = cv2.resize(sq, (128, 128))
            else:
                resized_sq = cv2.resize(sq, (64, 64))
            grey_sq = resized_sq
            if not colour:
                grey_sq = cv2.cvtColor(resized_sq, cv2.COLOR_BGR2GRAY)

            # if first and labels[index] != '':
            #     plt.imshow(grey_sq, cmap='gray')
            #     plt.title(f"Label: {labels[index]}")
            #     plt.axis('off')
            #     plt.show()
            #     first = True
            # if len(squares) -1 == index:
            #     plt.imshow(grey_sq, cmap='gray')
            #     plt.title(f"Label: {labels[index]}")
            #     plt.axis('off')
            #     plt.show()
            norm_sq = grey_sq
            if not colour:
                norm_sq = norm_sq / 255.0
                norm_sq = norm_sq[...,None] # add channel dimension
            new_data.append((norm_sq, piece_to_int[labels[index]]))

            if data_augmentation:
                for _ in range(NUM_AUGMENTATIONS):
                    
                    count = 0
                    if labels[index] == '' and np.random.rand() <0.0:  # Skip empty squares to balance the dataset
                            count += 1
                            continue
                        # if labels[count] == 'P' or labels[count] == 'p' and np.random.rand() < 0.7:  # Skip some empty squares to balance the dataset
                        #     count += 1
                        #     continue
                    resized_aug = sq
                    if colour:
                        resized_aug = cv2.resize(sq, (128, 128))
                    else:    
                        resized_aug = cv2.resize(sq, (64, 64))
                    augmented = augmentation_pipeline(image=resized_aug)['image']
                    grey_aug = augmented
                    if not colour:
                        grey_aug = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
                        # if firstAug:
                        #     plt.imshow(grey_aug, cmap='gray')
                        #     plt.title(f"Augmented Label: {labels[count]}")
                        #     plt.axis('off')
                        #     plt.show()
                        #     firstAug = False
                        # if len(squares) -1 == count:
                        #     plt.imshow(grey_aug, cmap='gray')
                        #     plt.title(f"Label: {labels[count]} (Augmented)")
                        #     plt.axis('off')
                        #     plt.show()
                    norm_aug = grey_aug
                    if not colour:
                        norm_aug = norm_aug / 255.0
                        norm_aug = norm_aug[...,None] # add channel dimension
                    new_data.append((norm_aug, piece_to_int[labels[index]]))
                    count += 1
    
    return new_data

def dataPreprocessingZenodo(data,data_augmentation=False,colour=False):
    # Placeholder for data preprocessing logic

    new_data = []

    augmentation_pipeline = A.Compose([
    A.Rotate(limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
    A.GaussNoise(std_range=(0.05, 0.1), p=0.2),
    A.ImageCompression(quality_range=(60, 100), p=0.3),
    A.GaussianBlur(blur_limit=(3,5), p=0.2),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.25),
    A.ElasticTransform(alpha=1, sigma=5, p=0.1),  # subtle shape distortion
    A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.1),
])
    
    NUM_AUGMENTATIONS = 5

    #Reize images, grey scale, normalise
    for image in data:
        square, label = image
        resized_sq = square
        if colour:
                resized_sq = cv2.resize(square, (128, 128))
        else:
            resized_sq = cv2.resize(square, (64, 64))
        grey_sq = resized_sq
        if not colour:
            grey_sq = cv2.cvtColor(resized_sq, cv2.COLOR_BGR2GRAY)

        # plt.imshow(grey_sq, cmap='gray')
        # plt.title(f"Label: {label}")
        # plt.axis('off')
        # plt.show()
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
        norm_sq = grey_sq
        if not colour:
            norm_sq = norm_sq / 255.0
            norm_sq = norm_sq[...,None] # add channel dimension
        new_data.append((norm_sq, label))

        if data_augmentation:
            for _ in range(NUM_AUGMENTATIONS):
                
                resized_aug = square
                if colour:
                    resized_aug = cv2.resize(square, (128, 128))
                else:
                    resized_aug = cv2.resize(square, (64, 64))
                augmented = augmentation_pipeline(image=resized_aug)['image']
                grey_aug = augmented
                if not colour:
                    grey_aug = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
                    # if firstAug:
                # if label != "0":
                #     plt.imshow(grey_aug, cmap='gray')
                #     plt.title(f"Augmented Label: {label}")
                #     plt.axis('off')
                #     plt.show()
                    #     firstAug = False
                    # if len(squares) -1 == count:
                    #     plt.imshow(grey_aug, cmap='gray')
                    #     plt.title(f"Label: {labels[count]} (Augmented)")
                    #     plt.axis('off')
                    #     plt.show()
                norm_aug = grey_aug
                if not colour:
                    norm_aug = norm_aug / 255.0
                    norm_aug = norm_aug[...,None] # add channel dimension
                new_data.append((norm_aug, label))
    
    return new_data

def savePreprocessedData(images,validation=False):
    # clear processed folder first
    
    save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processed"

    if validation:
        save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processedValidation"
    print("Removing old processed images...")
    for filename in os.listdir(save_path):
        os.remove(os.path.join(save_path, filename))
    
    print("Finished clearing old images. Saving new preprocessed images...")
    count = 0
    paths = []
    for img, label in images:
        filename = f"img_{count}_{label}.png"
        cv2.imwrite(os.path.join(save_path, filename), img)
        paths.append(os.path.join(save_path, filename))
        count += 1
    return paths

def denseNetPieceRecognitionModel(train_dataset, val_dataset, num_classes=13, epochs=30, batch_size=32,class_weight_dict=None):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # freeze for initial training
    model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
    ])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=250,
        restore_best_weights=True
    )
    checkpoint_frozen = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_frozen.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,  # save full model
    verbose=1
)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # indices = np.arange(len(x))
    # train_idx, val_idx = train_test_split(
    # indices,
    # test_size=0.2,
    # stratify=y,
    # random_state=42
    # )

    # x_train, y_train = x[train_idx], y[train_idx]
    # x_val, y_val = x[val_idx], y[val_idx]

    # sw_train,sw_val = sample_weights[train_idx], sample_weights[val_idx]

    history_frozen = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_frozen]
    )
    # model.load_weights('best_model_frozen.keras')  # Load best weights before fine-tuning

    model = tf.keras.models.load_model('best_model_frozen.keras')
    base_model = model.layers[0]  # Extract the base model from the loaded model
    best_frozen_loss = min(history_frozen.history['val_loss'])
    
    if best_frozen_loss > 0.155:
        print("Frozen model did not perform well enough, skipping fine-tuning.")
        return (model,best_frozen_loss)
    
    checkpoint_finetuned = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_finetuned.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    initial_value_threshold=best_frozen_loss,  # must beat frozen best to save
    verbose=1
    )
    print("Unfreezing base model for fine-tuning...")
    base_model.trainable = True

    for layer in base_model.layers[:-200]:  # Freeze all but last 40 layers
        layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # x_train, x_val, y_train, y_val = train_test_split(
    # x, y,
    # test_size=0.2,
    # stratify=y,
    # random_state=42
    # )

    history_fineTuned = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=300,
        callbacks=[early_stop,checkpoint_finetuned]
    )
    if os.path.exists('best_model_finetuned.keras'):
        model = tf.keras.models.load_model('best_model_finetuned.keras')
    else:
        model = tf.keras.models.load_model('best_model_frozen.keras')
    return (model,min(history_fineTuned.history['val_loss']))

def trainPieceRecognitionModel(train_dataset, val_dataset, num_classes=13, epochs=45, batch_size=32):

    model = models.Sequential([
        layers.Input(shape=(64, 64, 1)),

        # Block 1
        layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # Block 2
        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # Block 3
        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # Reduce parameters drastically
        layers.GlobalAveragePooling2D(),

        # Dense layers
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()



    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
    )
    # Train the model
    # model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True)

    return model

augmentation_pipeline_128 = A.Compose([
    A.Rotate(limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
    A.GaussNoise(std_range=(0.05, 0.1), p=0.2),
    A.ImageCompression(quality_range=(60, 100), p=0.3),
    A.GaussianBlur(blur_limit=(3,5), p=0.2),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.25),
    A.ElasticTransform(alpha=1, sigma=5, p=0.1),  # subtle shape distortion
    A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.1),
    # A.RandomResizedCrop(size=(128,128), scale=(0.9,1.0), ratio=(0.95,1.05), p=0.3)
])
augmentation_pipeline_64 = A.Compose([
    A.Rotate(limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
    A.GaussNoise(std_range=(0.05, 0.1), p=0.2),
    A.ImageCompression(quality_range=(60, 100), p=0.3),
    A.GaussianBlur(blur_limit=(3,5), p=0.2),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.25),
    A.ElasticTransform(alpha=1, sigma=5, p=0.1),  # subtle shape distortion
    A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.1),
    # A.RandomResizedCrop(size=(64,64), scale=(0.9,1.0), ratio=(0.95,1.05), p=0.3)
])
def augment_numpy(img,colour):
    img = img.numpy()
    colour = bool(colour.numpy())
    img = img.astype(np.uint8)
    if colour:
        augmented = augmentation_pipeline_128(image=img)["image"]
    else:
        augmented = augmentation_pipeline_64(image=img)["image"]
        augmented = augmented / 255.0
    return augmented.astype(np.float32)
def augment_tf(img,colour):
    img = tf.py_function(
        func=augment_numpy,
        inp=[img,colour],
        Tout=tf.float32
    )

 # important
    return img
def loadImage(path,label,augmentation=False,colour=False):
    img = tf.io.read_file(path)
    if colour:
        img = tf.image.decode_png(img, channels=3)
    else:
        img = tf.image.decode_png(img, channels=1)
    if augmentation:
        img = augment_tf(img,colour)
    else:
        img = tf.cast(img, tf.float32)
    #write file to check if augmentation is working
    # img = tf.image.convert_image_dtype(img, tf.float32)
    if colour:
        img = preprocess_input(img)
    else:
        img = img / 255.0
    return img, label
def trainModel(color=False):
    print("Loading training data...")
    # (training_dataRed, training_dataZenodo, training_dataMy) = getTrainingData()
    (training_dataRed, training_dataZenodo, training_dataMy) = loadTrainingData(color)
    print(f"Loaded {len(training_dataRed) + len(training_dataZenodo) + len(training_dataMy)} total training samples.")

    print("Splitting data into training and validation sets...")

    training_my, val_my = train_test_split(training_dataMy, test_size=0.2, random_state=42)
    training_red, val_red = train_test_split(training_dataRed, test_size=0.2, random_state=42)
    training_zenodo, val_zenodo = train_test_split(training_dataZenodo, test_size=0.2, random_state=42)

    
    preprocessed_dataMy = dataPreprocessingMy(training_my, data_augmentation=False,colour=color)
    preprocessed_dataRed = dataPreprocessingRed(training_red, data_augmentation=False,colour=color)
    preprocessed_dataZenodo = dataPreprocessingZenodo(training_zenodo, data_augmentation=False,colour=color)

    preprocessed_dataMy_Val = dataPreprocessingMy(val_my, data_augmentation=False,colour=color)
    preprocessed_dataRed_Val = dataPreprocessingRed(val_red, data_augmentation=False,colour=color)
    preprocessed_dataZenodo_Val = dataPreprocessingZenodo(val_zenodo, data_augmentation=False,colour=color)
    print(f"Preprocessed data. Total samples after augmentation: {len(preprocessed_dataRed) + len(preprocessed_dataZenodo) + len(preprocessed_dataMy)}")
    
    # Count samples
    # train_counts = [
    #     len(preprocessed_dataMy),
    #     len(preprocessed_dataRed),
    #     len(preprocessed_dataZenodo)
    # ]

    # val_counts = [
    #     len(preprocessed_dataMy_Val),
    #     len(preprocessed_dataRed_Val),
    #     len(preprocessed_dataZenodo_Val)
    # ]

    # datasets = ['My', 'Red', 'Zenodo']

    # # X positions
    # x = np.arange(len(datasets))
    # width = 0.35

    # # Plot
    # plt.figure(figsize=(8,5))
    # plt.bar(x - width/2, train_counts, width, label='Training')
    # plt.bar(x + width/2, val_counts, width, label='Validation')

    # plt.xlabel("Dataset")
    # plt.ylabel("Number of Samples")
    # plt.title("Training vs Validation Samples per Dataset")
    # plt.xticks(x, datasets)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    print("Preprocessing data...")

    print("Getting ready to train the model...")

    # all_preprocessed_data = preprocessed_dataRed + preprocessed_dataZenodo
    all =  preprocessed_dataMy + preprocessed_dataRed

    all_val =   preprocessed_dataMy_Val + preprocessed_dataRed_Val

    # x = []
    # y = []
    # sample_weights = []

    # # for square, label in all_preprocessed_data:
    # #     x.append(square)
    # #     y.append(label)
    # #     sample_weights.append(1.0)
    
    # for square, label in all:
    #     x.append(square)
    #     y.append(label)
    #     sample_weights.append(1.0)  # Give more weight to my images to help the model generalize better to them

    labels = []
    for square, label in all:
        labels.append(label)
    
    labels_val = []
    for square, label in all_val:
        labels_val.append(label)
    
    int_to_piece = {
    0: 'empty', 1: 'P', 2: 'N', 3: 'B', 4: 'R',
    5: 'Q', 6: 'K', 7: 'p', 8: 'n', 9: 'b',
    10: 'r', 11: 'q', 12: 'k'
    }

    train_counts = Counter(label for _, label in all)
    val_counts = Counter(label for _, label in all_val)

    print("\n--- Class Distribution ---")
    print(f"{'Class':<10} {'Train':>8} {'Val':>8}")
    print("-" * 28)
    for cls in range(13):
        print(f"{int_to_piece[cls]:<10} {train_counts.get(cls, 0):>8} {val_counts.get(cls, 0):>8}")
    print(f"{'TOTAL':<10} {sum(train_counts.values()):>8} {sum(val_counts.values()):>8}")

    paths = savePreprocessedData(all)
    paths_val = savePreprocessedData(all_val, validation=True)
    
    paths = tf.constant(paths)
    labels = tf.constant(labels)
    paths_val = tf.constant(paths_val)
    labels_val = tf.constant(labels_val)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((paths_val, labels_val))

    train_ds = (dataset
               .map(lambda path, label: (loadImage(path, label,augmentation=True,colour=color)), num_parallel_calls=tf.data.AUTOTUNE)
               .shuffle(buffer_size=1000)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (val_dataset
               .map(lambda path, label: (loadImage(path, label,augmentation=False,colour=color)), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )

    # print(np.unique(y))
    # print(y[:20])


    # # Convert to numpy arrays
    # x = np.array(x, dtype="float32")
    # print(x.shape)
    # y = np.array(y, dtype="int32")
    # sample_weights = np.array(sample_weights, dtype="float32")
    # print(np.unique(y))
    # print(y[:20])
    # print(x.nbytes / 1e9, "GB")
    val = 0.0
    if not color:
        model = trainPieceRecognitionModel(train_ds,val_ds)
    else:
            
            all_labels = [label for _, label in all]
            class_weights = compute_class_weight('balanced', classes=np.arange(13), y=all_labels)
            class_weight_dict = dict(enumerate(class_weights))
            model, val = denseNetPieceRecognitionModel(train_ds,val_ds, class_weight_dict=class_weight_dict)
    # model.save("piece_recognition_model.h5")
    # model.save_weights("piece_recognition_weights.h5")
    return (model,val)
    # testModel(model)

def testModel(model, weight, color=False):
    model.load_weights(weight)

    test_data = get_testingImages(colour=color)
    test_data = dataPreprocessingRed(test_data, data_augmentation=False, colour=color)

    labels_test = [label for _, label in test_data]
    paths_test = savePreprocessedData(test_data, validation=True)

    paths_test = tf.constant(paths_test)
    labels_test = tf.constant(labels_test)

    test_ds = (tf.data.Dataset.from_tensor_slices((paths_test, labels_test))
               .map(lambda path, label: loadImage(path, label, augmentation=False, colour=color), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )

    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")
    return accuracy

def labels_to_fen(predicted_labels):
    fen_rows = []
    for row in range(8):
        fen_row = ""
        empty_count = 0
        for col in range(8):
            piece = predicted_labels[row * 8 + col]
            if piece == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    fen_string = "/".join(fen_rows)  # keep top-to-bottom as in your labels
    return fen_string

def makePredictions(model, weight, image, color=False):
    model.load_weights(weight)
    squares = BoardSegmentation.getSquaresFromImage(image, colour=color)
    
    # Save squares temporarily like the pipeline does
    square_data = [(sq, 0) for sq in squares]  # dummy label 0
    paths = savePreprocessedData(square_data, validation=True)
    
    paths_tf = tf.constant(paths)
    labels_tf = tf.constant([0] * len(paths))
    
    pred_ds = (tf.data.Dataset.from_tensor_slices((paths_tf, labels_tf))
               .map(lambda path, label: loadImage(path, label, augmentation=False, colour=color), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )
    
    predictions = model.predict(pred_ds)
    predicted_labels = np.argmax(predictions, axis=1)
    probs = np.max(predictions, axis=1)
    print(predicted_labels)
    print(probs)
    return (predicted_labels, predictions[np.newaxis, ...])

def visualisePredictions(predicted_labels,filePath):
    int_to_piece = {
        0: '',
        1: 'P',
        2: 'N',
        3: 'B',
        4: 'R',
        5: 'Q',
        6: 'K',
        7: 'p',
        8: 'n',
        9: 'b',
        10: 'r',
        11: 'q',
        12: 'k'
    }

    predicted_pieces = [int_to_piece[label] for label in predicted_labels]

    fen_string = labels_to_fen(predicted_pieces)


    print(predicted_pieces)
    print(fen_string)
    Visual_Representation.visualize_fen(fen_string, filePath)

def getValidationAccuracy(color=False):
    model = tf.keras.models.load_model('piece_recognition_model.h5')
    model.load_weights('piece_recognition_weights.h5')
    
    (training_dataRed, training_dataZenodo, training_dataMy) = loadTrainingData(color)
    
    _, val_my = train_test_split(training_dataMy, test_size=0.2, random_state=42)
    _, val_red = train_test_split(training_dataRed, test_size=0.2, random_state=42)

    preprocessed_val_my = dataPreprocessingMy(val_my, data_augmentation=False, colour=color)
    preprocessed_val_red = dataPreprocessingRed(val_red, data_augmentation=False, colour=color)

    all_val = preprocessed_val_my + preprocessed_val_red

    labels_val = [label for _, label in all_val]
    paths_val = savePreprocessedData(all_val, validation=True)

    paths_val = tf.constant(paths_val)
    labels_val = tf.constant(labels_val)

    val_ds = (tf.data.Dataset.from_tensor_slices((paths_val, labels_val))
               .map(lambda path, label: loadImage(path, label, augmentation=False, colour=color), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )

    loss, accuracy = model.evaluate(val_ds, verbose=1)
    print(f"Validation Loss: {loss:.4f} | Validation Accuracy: {accuracy:.4f}")
    return accuracy

# getValidationAccuracy(color=True)
# bestVal = float('inf')
# bestTestAcc = 0.0
# while True:
#     model,val = trainModel(True)
#     testAcc = testModel(tf.keras.models.load_model("piece_recognition_model.h5"), "piece_recognition_weights.h5",True)
#     if val < bestVal:
#         bestVal = val
#         model.save("piece_recognition_model.h5")
#         model.save_weights("piece_recognition_weights.h5")
#         print(f"New best validation loss: {bestVal}")
#         print(f"Test accuracy at this point: {testAcc}")
#     if math.isclose(val, bestVal, rel_tol=1e-4):
#         if testAcc > bestTestAcc:
#             model.save("piece_recognition_model.h5")
#             model.save_weights("piece_recognition_weights.h5")
#             bestTestAcc = testAcc
#         print(f"New best test accuracy: {bestTestAcc}")
# testModel(tf.keras.models.load_model("piece_recognition_model.h5"), "piece_recognition_weights.h5",True)
# visualisePredictions(makePredictions(tf.keras.models.load_model("piece_recognition_model.h5"), "piece_recognition_weights.h5", "C:\\Users\\Callu\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\testingImages\\rnbqk2r_ppp1bppp_3p1n2_4p3_2B1P3_P4N2_1PPP1PPP_RNBQK2R,w,KQkq,-,1,5.jpg", color=True)[0], "pipeline_prediction.png")