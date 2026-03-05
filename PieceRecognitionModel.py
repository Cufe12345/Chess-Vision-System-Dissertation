import BoardSegmentation
import os
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models 
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

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

def get_chessRedImages():
    folder_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\chessRed\\FinalImages"
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]

    images = []
    for i, image in enumerate(image_files):
        # print(f"Processing image {i+1}/{len(image_files)}: {image}")
        try:
            squares = BoardSegmentation.getSquaresFromImage(image)
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


def get_myImages():
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
            squares = BoardSegmentation.getSquaresFromImage(image)
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

def get_testingImages():
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
            squares = BoardSegmentation.getSquaresFromImage(image)
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
    
def getTrainingData():
    # Placeholder for loading and preprocessing training data

    print("Loading my images...")
    imagesMy = get_myImages()
    print(f"Loaded {len(imagesMy)} of my images.")
    
    print("Loading chessRed images...")
    imagesRed = get_chessRedImages()
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

def loadTrainingData():

    try:
        with open("training_dataRed.pkl", "rb") as f:
            training_dataRed = pickle.load(f)
        with open("training_dataZenodo.pkl", "rb") as f:
            training_dataZenodo = pickle.load(f)
        with open("training_dataMy.pkl", "rb") as f:
            training_dataMy = pickle.load(f)
    except FileNotFoundError:
        getTrainingData()
        return loadTrainingData()
    return (training_dataRed, training_dataZenodo, training_dataMy)

def dataPreprocessingRed(data,data_augmentation=False):
    # Placeholder for data preprocessing logic

    new_data = []

    augmentation_pipeline = A.Compose([
            A.Rotate(limit=5, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), per_channel=False, noise_scale_factor=1.0, p=0.3),
            A.CoarseDropout(num_holes_range=(1,1), hole_height_range=(5,8), hole_width_range=(5,8), fill=0, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.HorizontalFlip(p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
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
            if labels[index] == '' and np.random.rand() < 0.0:  # Skip some empty squares to balance the dataset
                continue
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
            new_data.append((norm_sq, piece_to_int[labels[index]]))

        if data_augmentation:
            for _ in range(NUM_AUGMENTATIONS):
                
                count = 0
                for sq in squares:
                    if labels[count] == '' and np.random.rand() <1:  # Skip some empty squares to balance the dataset
                        continue
                    resized_aug = cv2.resize(sq, (64, 64))
                    augmented = augmentation_pipeline(image=resized_aug)['image']
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
                    norm_aug = grey_aug / 255.0
                    norm_aug = norm_aug[...,None] # add channel dimension
                    new_data.append((norm_aug, piece_to_int[labels[count]]))
                    count += 1
    
    return new_data

def dataPreprocessingZenodo(data,data_augmentation=False):
    # Placeholder for data preprocessing logic

    new_data = []

    augmentation_pipeline = A.Compose([
            A.Rotate(limit=5, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), per_channel=False, noise_scale_factor=1.0, p=0.3),
            A.CoarseDropout(num_holes_range=(1,1), hole_height_range=(5,8), hole_width_range=(5,8), fill=0, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.HorizontalFlip(p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        ])
    
    NUM_AUGMENTATIONS = 5

    #Reize images, grey scale, normalise
    for image in data:
        square, label = image
        resized_sq = cv2.resize(square, (64, 64))
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
        norm_sq = grey_sq / 255.0
        norm_sq = norm_sq[...,None] # add channel dimension
        new_data.append((norm_sq, label))

        if data_augmentation:
            for _ in range(NUM_AUGMENTATIONS):
                
                
                resized_aug = cv2.resize(square, (64, 64))
                augmented = augmentation_pipeline(image=resized_aug)['image']
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
                norm_aug = grey_aug / 255.0
                norm_aug = norm_aug[...,None] # add channel dimension
                new_data.append((norm_aug, label))
    
    return new_data
def trainPieceRecognitionModel(x, y, num_classes=13, epochs=45, batch_size=32):

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

    x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=0.2,
    stratify=y,
    random_state=42
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True
    )
    # Train the model
    # model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True)

    return model

def trainModel():
    print("Loading training data...")
    # (training_dataRed, training_dataZenodo, training_dataMy) = getTrainingData()
    (training_dataRed, training_dataZenodo, training_dataMy) = loadTrainingData()
    print(f"Loaded {len(training_dataRed) + len(training_dataZenodo) + len(training_dataMy)} total training samples.")

    print("Preprocessing data...")
    preprocessed_dataRed = dataPreprocessingRed(training_dataRed, data_augmentation=True)
    preprocessed_dataZenodo = dataPreprocessingZenodo(training_dataZenodo, data_augmentation=True)
    preprocessed_dataMy = dataPreprocessingRed(training_dataMy, data_augmentation=True)
    print(f"Preprocessed data. Total samples after augmentation: {len(preprocessed_dataRed) + len(preprocessed_dataZenodo) + len(preprocessed_dataMy)}")
    
    print("Getting ready to train the model...")

    all_preprocessed_data = preprocessed_dataRed + preprocessed_dataMy + preprocessed_dataZenodo

    x = []
    y = []

    for square, label in all_preprocessed_data:
        x.append(square)
        y.append(label)

    print(np.unique(y))
    print(y[:20])


    # Convert to numpy arrays
    x = np.array(x, dtype="float32")
    y = np.array(y, dtype="int32")
    print(np.unique(y))
    print(y[:20])
    model = trainPieceRecognitionModel(x,y)
    model.save("piece_recognition_model.h5")
    model.save_weights("piece_recognition_weights.h5")

    # testModel(model)

def testModel(model, weight):
    model.load_weights(weight)

    test_data = get_testingImages()
    test_data = dataPreprocessingRed(test_data, data_augmentation=False)

    x_test, y_test = zip(*test_data)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model.evaluate(x_test, y_test)

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

def makePredictions(model, weight, image):
    model.load_weights(weight)
    squares = BoardSegmentation.getSquaresFromImage(image)
    preprocessed_squares = []
    for sq in squares:
        resized_sq = cv2.resize(sq, (64, 64))
        grey_sq = cv2.cvtColor(resized_sq, cv2.COLOR_BGR2GRAY)
        norm_sq = grey_sq / 255.0
        norm_sq = norm_sq[...,None] # add channel dimension
        preprocessed_squares.append(norm_sq)
    preprocessed_squares = np.array(preprocessed_squares)
    predictions = model.predict(preprocessed_squares)
    # print(predictions)
    predicted_labels = np.argmax(predictions, axis=1)
    probs = np.max(predictions, axis=1)
    print(predicted_labels)
    print(probs)
    return (predicted_labels,probs)
def visualisePredictions(predicted_labels):
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
    Visual_Representation.visualize_fen(fen_string)


# trainModel()
# testModel(tf.keras.models.load_model("piece_recognition_model.h5"), "piece_recognition_weights.h5")
visualisePredictions(makePredictions(tf.keras.models.load_model("piece_recognition_model.h5"), "piece_recognition_weights.h5", "C:\\Users\\Callu\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\testingImages\\1k6_1P2PK2_8_2bB4_8_8_8_8,b,-,-,0,70.jpg")[0])