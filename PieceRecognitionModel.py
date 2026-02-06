import BoardSegmentation
import os
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models 
import numpy as np

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
    for image in image_files:
        print(f"Processing image: {image}")
        try:
            squares = BoardSegmentation.getSquaresFromImage(image)
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            images.append((squares, labels))

            # for testing purpose only get 3
            if len(images) >= 3:
                break
            # print(f"Piece labels for {image}: {piece_labels}")
        except Exception as e:
            print(f"Error processing {image}: {e}")
    
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

def getTrainingData():
    # Placeholder for loading and preprocessing training data

    images = get_chessRedImages()

    return images

def dataPreprocessing(data,data_augmentation=False):
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
    for image in data:
        squares, labels = image
        first = True
        firstAug = True
        for index,sq in enumerate(squares):
            resized_sq = cv2.resize(sq, (64, 64))
            grey_sq = cv2.cvtColor(resized_sq, cv2.COLOR_BGR2GRAY)

            if first:
                plt.imshow(grey_sq, cmap='gray')
                plt.title(f"Label: {labels[index]}")
                plt.axis('off')
                plt.show()
                first = False
            if len(squares) -1 == index:
                plt.imshow(grey_sq, cmap='gray')
                plt.title(f"Label: {labels[index]}")
                plt.axis('off')
                plt.show()
            norm_sq = grey_sq / 255.0
            norm_sq = norm_sq[...,None] # add channel dimension
            new_data.append((norm_sq, piece_to_int[labels[index]]))

        if data_augmentation:
            for _ in range(NUM_AUGMENTATIONS):
                
                count = 0
                for sq in squares:
                    resized_aug = cv2.resize(sq, (64, 64))
                    augmented = augmentation_pipeline(image=resized_aug)['image']
                    grey_aug = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
                    if firstAug:
                        plt.imshow(grey_aug, cmap='gray')
                        plt.title(f"Augmented Label: {labels[count]}")
                        plt.axis('off')
                        plt.show()
                        firstAug = False
                    if len(squares) -1 == count:
                        plt.imshow(grey_aug, cmap='gray')
                        plt.title(f"Label: {labels[count]} (Augmented)")
                        plt.axis('off')
                        plt.show()
                    norm_aug = grey_aug / 255.0
                    norm_aug = norm_aug[...,None] # add channel dimension
                    new_data.append((norm_aug, piece_to_int[labels[count]]))
                    count += 1
        

        
    
    return new_data
def trainPieceRecognitionModel(x, y, num_classes=13, epochs=400, batch_size=128):

    model = models.Sequential([
        layers.Input(shape=(64, 64, 1)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train the model
    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True)

    return model

def trainModel():
    print("Loading training data...")
    training_data = getTrainingData()
    print(f"Loaded {len(training_data)} training samples.")

    print("Preprocessing data...")
    preprocessed_data = dataPreprocessing(training_data, data_augmentation=True)
    print(f"Preprocessed data. Total samples after augmentation: {len(preprocessed_data)}")

    print(preprocessed_data[0])
    
    print("Getting ready to train the model...")

    x = []
    y = []

    for square, label in preprocessed_data:
        x.append(square)
        y.append(label)


    # Convert to numpy arrays
    x = np.array(x, dtype="float32")
    y = np.array(y, dtype="int32")
    model = trainPieceRecognitionModel(x,y)

trainModel()