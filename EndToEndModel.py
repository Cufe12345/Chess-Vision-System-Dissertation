import BoardSegmentation
import os
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models 
import numpy as np


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
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            image = cv2.imread(image)
            images.append((image, labels))

            # for testing purpose only get 3
            # if len(images) >= 3:
            #     break
            # print(f"Piece labels for {image}: {piece_labels}")
        except Exception as e:
            # print(f"Error processing {image}: {e}")
            continue

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
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            image = cv2.imread(image)
            # labels values need remapping as these images are side on
            remapped_labels = []
            for j in range(8):
                for i in range(7,-1,-1):
                    remapped_labels.append(labels[i*8 + j])
            final_images.append((image,remapped_labels ))
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
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            image = cv2.imread(image)
            # labels values need remapping as these images are side on
            remapped_labels = []
            for j in range(8):
                for i in range(7,-1,-1):
                    remapped_labels.append(labels[i*8 + j])

            final_images.append((image, remapped_labels))
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

    return (imagesRed, imagesMy)


def dataPreprocessingRed(data,data_augmentation=False):
    # Placeholder for data preprocessing logic

    new_data = []

    augmentation_pipeline = A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), per_channel=False, noise_scale_factor=1.0, p=0.3),
            A.CoarseDropout(num_holes_range=(1,1), hole_height_range=(5,8), hole_width_range=(5,8), fill=0, p=0.3)
        ])
    
    NUM_AUGMENTATIONS = 3
    
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
    for item in data:
        image, labels = item
        resized_img = cv2.resize(image, (400, 400))
        grey_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

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
        norm_img = grey_img / 255.0
        norm_img = norm_img[...,None] # add channel dimension
        intLabels = []
        for label in labels:
            intLabels.append(piece_to_int[label])
        new_data.append((norm_img, intLabels))

        if data_augmentation:
            for _ in range(NUM_AUGMENTATIONS):
                

                resized_aug = cv2.resize(image, (400, 400))
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
                new_data.append((norm_aug, intLabels))
    
    return new_data


def trainEndToEndModel(x, y, num_classes=13, epochs=45, batch_size=32):

    num_squares = 64
    model = models.Sequential([
        layers.Input(shape=(400, 400, 1)),

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
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_squares*num_classes, activation="softmax"),
        layers.Reshape((num_squares, num_classes))

    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train the model
    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True)

    return model

def trainModel():
    print("Loading training data...")
    (training_dataRed, training_dataMy) = getTrainingData()
    print(f"Loaded {len(training_dataRed) + len(training_dataMy)} total training samples.")

    print("Preprocessing data...")
    preprocessed_dataRed = dataPreprocessingRed(training_dataRed, data_augmentation=True)
    preprocessed_dataMy = dataPreprocessingRed(training_dataMy, data_augmentation=True)
    print(f"Preprocessed data. Total samples after augmentation: {len(preprocessed_dataRed) + len(preprocessed_dataMy)}")
    
    print("Getting ready to train the model...")

    all_preprocessed_data = preprocessed_dataRed + preprocessed_dataMy

    x = []
    y = []

    for image, label in all_preprocessed_data:
        x.append(image)
        y.append(label)

    print(np.unique(y))
    print(y[:20])


    # Convert to numpy arrays
    x = np.array(x, dtype="float32")
    y = np.array(y, dtype="int32")
    print(np.unique(y))
    print(y[:20])
    model = trainEndToEndModel(x,y)
    model.save("end_to_end_model.h5")
    model.save_weights("end_to_end_weights.h5")

    # testModel(model)

def testModel(model, weight):
    model.load_weights(weight)

    test_data = get_testingImages()
    test_data = dataPreprocessingRed(test_data, data_augmentation=False)

    x_test, y_test = zip(*test_data)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model.evaluate(x_test, y_test)

# trainModel()
testModel(tf.keras.models.load_model("end_to_end_model.h5"), "end_to_end_weights.h5")

