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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,f1_score
import math

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
        try:
            squares = BoardSegmentation.getSquaresFromImage(image,farChessTable=True)
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            images.append((squares, labels))
        except Exception as e:
            continue
    return images


def get_myImages():
    """Load my images"""
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
        try:
            squares = BoardSegmentation.getSquaresFromImage(image)
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

def get_testingImages():
    """Load testing images"""
    path_testing = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\testingImages"


    image_files = [
    os.path.join(path_testing, f)
    for f in os.listdir(path_testing)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]



    final_images = []
    for i, image in enumerate(image_files):
        try:
            squares = BoardSegmentation.getSquaresFromImage(image)
            labels = fen_to_labels(os.path.basename(image).split('.')[0])

            # labels values need remapping as these images are side on
            remapped_labels = []
            for j in range(8):
                for i in range(7,-1,-1):
                    remapped_labels.append(labels[i*8 + j])

            final_images.append((squares, remapped_labels))
        except Exception as e:
            print(f"Error processing image{i}: {e}")
            continue
    
    return final_images
    
def getTrainingData():
    """Load training data from all sources and save to pickle files for later use"""

    print("Loading my images...")
    imagesMy = get_myImages()
    print(f"Loaded {len(imagesMy)} of my images.")
    
    print("Loading chessRed images...")
    imagesRed = get_chessRedImages()
    print(f"Loaded {len(imagesRed)} chessRed images.")

    #save the training data for later use
    with open("training_dataRed.pkl", "wb") as f:
        pickle.dump(imagesRed, f)
    with open("training_dataMy.pkl", "wb") as f:
        pickle.dump(imagesMy, f)
    return (imagesRed, imagesMy)


def loadTrainingData():
    """Load training data from pickle files if they exist, otherwise call getTrainingData to create them"""
    try:
        with open("training_dataRed.pkl", "rb") as f:
            training_dataRed = pickle.load(f)
        with open("training_dataMy.pkl", "rb") as f:
            training_dataMy = pickle.load(f)
    except FileNotFoundError:
        getTrainingData()
        return loadTrainingData()
    return (training_dataRed, training_dataMy)

def dataPreprocessing(data):
    """Resize images and covert labels to ints"""

    new_data = []
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


    for image in data:
        squares, labels = image
        for index,sq in enumerate(squares):
            if labels[index] == '' and np.random.rand() < 0.5:  # Skip some empty squares to balance the dataset
                continue
            resized_sq = cv2.resize(sq, (128, 128))


            new_data.append((resized_sq, piece_to_int[labels[index]]))
    
    return new_data

def savePreprocessedData(images,validation=False):
    """Save preprocessed images to disk for later use in training"""
    
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

def denseNetPieceRecognitionModel(train_dataset, val_dataset, num_classes=13, epochs=30, batch_size=32):
    """Train a piece recognition model with DenseNet121 as the base model"""
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

    history_frozen = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_frozen]
    )

    model = tf.keras.models.load_model('best_model_frozen.keras')
    base_model = model.layers[0]
    best_frozen_loss = min(history_frozen.history['val_loss'])
    
    if best_frozen_loss > 0.155:
        print("Frozen model did not perform well enough, skipping fine-tuning.")
        return (model,best_frozen_loss)
    
    checkpoint_finetuned = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_finetuned.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    initial_value_threshold=best_frozen_loss,
    verbose=1
    )

    print("Unfreezing base model for fine-tuning...")
    base_model.trainable = True

    for layer in base_model.layers[:-200]:
        layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    history_fineTuned = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=300,
        callbacks=[early_stop,checkpoint_finetuned]
    )

    # Return the best model based on validation loss, whether it is the frozen or fine-tuned version
    if os.path.exists('best_model_finetuned.keras'):
        model = tf.keras.models.load_model('best_model_finetuned.keras')
    else:
        model = tf.keras.models.load_model('best_model_frozen.keras')
    return (model,min(history_fineTuned.history['val_loss']))


augmentation_pipeline_128 = A.Compose([
    A.Rotate(limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
    A.GaussNoise(std_range=(0.05, 0.1), p=0.2),
    A.ImageCompression(quality_range=(60, 100), p=0.3),
    A.GaussianBlur(blur_limit=(3,5), p=0.2),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.25),
    A.ElasticTransform(alpha=1, sigma=5, p=0.1),
    A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.1),
])

def augment_numpy(img):
    """Actually does the augmentation to images, called by augment_tf"""
    img = img.numpy()
    img = img.astype(np.uint8)
    augmented = augmentation_pipeline_128(image=img)["image"]

    return augmented.astype(np.float32)
def augment_tf(img):
    """Applies augmentation to images"""
    img = tf.py_function(
        func=augment_numpy,
        inp=[img],
        Tout=tf.float32
    )
    return img
def loadImage(path,label,augmentation=False):
    """Loads an image from a path and applies augmentation if not validation data"""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)

    if augmentation:
        img = augment_tf(img)
    else:
        img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label

def trainModel():
    """Trains the end-to-end model, including data loading, preprocessing, and model training"""
    print("Loading training data...")
    (training_dataRed, training_dataMy) = loadTrainingData()
    print(f"Loaded {len(training_dataRed) + len(training_dataMy)} total training samples.")

    print("Splitting data into training and validation sets...")

    training_my, val_my = train_test_split(training_dataMy, test_size=0.2, random_state=42)
    training_red, val_red = train_test_split(training_dataRed, test_size=0.2, random_state=42)

    
    preprocessed_dataMy = dataPreprocessing(training_my)
    preprocessed_dataRed = dataPreprocessing(training_red)

    preprocessed_dataMy_Val = dataPreprocessing(val_my)
    preprocessed_dataRed_Val = dataPreprocessing(val_red)
    
    print("Getting ready to train the model...")

    all =  preprocessed_dataMy + preprocessed_dataRed

    all_val =   preprocessed_dataMy_Val + preprocessed_dataRed_Val

    labels = []
    for square, label in all:
        labels.append(label)
    
    labels_val = []
    for square, label in all_val:
        labels_val.append(label)


    paths = savePreprocessedData(all)
    paths_val = savePreprocessedData(all_val, validation=True)
    
    paths = tf.constant(paths)
    labels = tf.constant(labels)
    paths_val = tf.constant(paths_val)
    labels_val = tf.constant(labels_val)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((paths_val, labels_val))

    train_ds = (dataset
               .map(lambda path, label: (loadImage(path, label,augmentation=True)), num_parallel_calls=tf.data.AUTOTUNE)
               .shuffle(buffer_size=1000)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (val_dataset
               .map(lambda path, label: (loadImage(path, label,augmentation=False)), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )
    model, val = denseNetPieceRecognitionModel(train_ds,val_ds)

    return (model,val)

def testModel(model, weight):
    """Tests the model on the unseen testing set"""
    model.load_weights(weight)

    test_data = get_testingImages()
    test_data = dataPreprocessing(test_data)

    labels_test = [label for _, label in test_data]
    paths_test = savePreprocessedData(test_data, validation=True)

    paths_test = tf.constant(paths_test)
    labels_test = tf.constant(labels_test)

    test_ds = (tf.data.Dataset.from_tensor_slices((paths_test, labels_test))
               .map(lambda path, label: loadImage(path, label, augmentation=True), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )

    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")
    return accuracy

def makePredictions(model, weight, image):
    """Makes predictions on a single image using the trained model and returns the predicted labels and probabilities"""
    model.load_weights(weight)
    squares = BoardSegmentation.getSquaresFromImage(image)
    
    # Save squares temporarily like the pipeline does
    square_data = [(sq, 0) for sq in squares]
    paths = savePreprocessedData(square_data, validation=True)
    
    paths_tf = tf.constant(paths)
    labels_tf = tf.constant([0] * len(paths))
    
    pred_ds = (tf.data.Dataset.from_tensor_slices((paths_tf, labels_tf))
               .map(lambda path, label: loadImage(path, label, augmentation=False), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )
    
    predictions = model.predict(pred_ds)
    predicted_labels = np.argmax(predictions, axis=1)
    return (predicted_labels, predictions[np.newaxis, ...])



def getValidationAccuracy():
    """Evaluates the model on the validation set and returns the accuracy, F1 score, and confusion matrix"""

    model = tf.keras.models.load_model('piece_recognition_model.h5')
    model.load_weights('piece_recognition_weights.h5')
    
    (training_dataRed, training_dataMy) = loadTrainingData()
    
    training_my, val_my = train_test_split(training_dataMy, test_size=0.2, random_state=42)
    training_red, val_red = train_test_split(training_dataRed, test_size=0.2, random_state=42)

    preprocessed_train_my = dataPreprocessing(training_my)
    preprocessed_train_red = dataPreprocessing(training_red)
    preprocessed_val_my = dataPreprocessing(val_my)
    preprocessed_val_red = dataPreprocessing(val_red)

    all_train = preprocessed_train_my + preprocessed_train_red
    all_val = preprocessed_val_my + preprocessed_val_red

    #Gets training and validation data ready for evaluation by saving the preprocessed images to disk and creating tf datasets from those saved images, just like the training pipeline does
    labels_train = [label for _, label in all_train]
    paths_train = savePreprocessedData(all_train)

    paths_train_tf = tf.constant(paths_train)
    labels_train_tf = tf.constant(labels_train)

    train_ds = (tf.data.Dataset.from_tensor_slices((paths_train_tf, labels_train_tf))
                .map(lambda path, label: loadImage(path, label, augmentation=False),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .batch(32)
                .prefetch(tf.data.AUTOTUNE)
    )

    labels_val = [label for _, label in all_val]
    paths_val = savePreprocessedData(all_val, validation=True)

    paths_val_tf = tf.constant(paths_val)
    labels_val_tf = tf.constant(labels_val)

    val_ds = (tf.data.Dataset.from_tensor_slices((paths_val_tf, labels_val_tf))
               .map(lambda path, label: loadImage(path, label, augmentation=False),
                    num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )

    
    train_loss, train_accuracy = model.evaluate(train_ds, verbose=1)
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")

    loss, accuracy = model.evaluate(val_ds, verbose=1)
    print(f"Validation Loss: {loss:.4f} | Validation Accuracy: {accuracy:.4f}")

    # Collect predictions and labels for the entire validation set
    all_preds = []
    for images, _ in val_ds:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))

    all_preds = np.array(all_preds)
    labels_val = np.array(labels_val)


    f1 = f1_score(labels_val, all_preds, average='weighted')
    f1_per_class = f1_score(labels_val, all_preds, average=None, labels=list(range(13)))
    print(f"Weighted F1 Score: {f1:.4f}")

    int_to_piece = {
        0: 'empty', 1: 'P', 2: 'N', 3: 'B', 4: 'R',
        5: 'Q', 6: 'K', 7: 'p', 8: 'n', 9: 'b',
        10: 'r', 11: 'q', 12: 'k'
    }

    print("\n--- Per-Class F1 Scores ---")
    for i, score in enumerate(f1_per_class):
        print(f"  {int_to_piece[i]:<8}: {score:.4f}")
    print("\n--- Per-Class Accuracy Scores ---")
    for i in range(13):
        class_mask = (labels_val == i)
        if class_mask.sum() == 0:
            print(f"  {int_to_piece[i]:<8}: No samples")
            continue
        class_acc = (all_preds[class_mask] == labels_val[class_mask]).sum() / class_mask.sum()
        print(f"  {int_to_piece[i]:<8}: {class_acc:.4f}  ({class_mask.sum()} samples)")
        
    # Confusion Matrix
    cm = confusion_matrix(labels_val, all_preds, labels=list(range(13)))
    display_labels = [int_to_piece[i] for i in range(13)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to confusion_matrix.png")

    return accuracy, f1, cm


def SingleTrainingRun():
    model,val = trainModel()
    model.save("piece_recognition_model.h5")
    model.save_weights("piece_recognition_weights.h5")
def MultipleTrainingRuns(count):
    bestVal = float('inf')
    bestTestAcc = 0.0
    while count > 0:
        model,val = trainModel()
        model.save("temp_model.h5")
        model.save_weights("temp_weights.h5")
        testAcc = testModel(tf.keras.models.load_model("temp_model.h5"), "temp_weights.h5")
        if val < bestVal:
            bestVal = val
            model.save("piece_recognition_model.h5")
            model.save_weights("piece_recognition_weights.h5")
            print(f"New best validation loss: {bestVal}")
            print(f"Test accuracy at this point: {testAcc}")
        if math.isclose(val, bestVal, rel_tol=1e-4):
            if testAcc > bestTestAcc:
                model.save("piece_recognition_model.h5")
                model.save_weights("piece_recognition_weights.h5")
                bestTestAcc = testAcc
            print(f"New best test accuracy: {bestTestAcc}")
        count -= 1

def UnseenTest():
    testModel(tf.keras.models.load_model("piece_recognition_model.h5"), "piece_recognition_weights.h5")