import BoardSegmentation
import os
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models 
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score


def fen_to_labels(fen):
    """
    Convert a fen to a list of 64 labels representing the piece on the board
    """
    
     #Keep only the board part
    board_part = fen.split(',')[0]

    #Split into 8 rows
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
    folder_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\chessRed\\FinalImages_end"
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
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            image = cv2.imread(image)
            image = BoardSegmentation.getBoardFromImage(image, debug=False,farChessTable=True)
            images.append((image, labels))
        except Exception as e:
            # when an image fails to be extracted
            continue
    return images


def get_myImages():
    """Loads my images"""
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
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            image = cv2.imread(image)
            image = BoardSegmentation.getBoardFromImage(image, debug=False)
            # labels values need remapping as these images are side on
            remapped_labels = []
            for j in range(8):
                for i in range(7,-1,-1):
                    remapped_labels.append(labels[i*8 + j])
            final_images.append((image,remapped_labels ))
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    return final_images

def get_testingImages():
    """Fetches the testing images and its labels"""
    path_testing = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\testingImages2"


    image_files = [
    os.path.join(path_testing, f)
    for f in os.listdir(path_testing)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]



    final_images = []
    for i, image in enumerate(image_files):
        try:
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            image = cv2.imread(image)
            remapped_labels = []
            for j in range(8):
                for i in range(7,-1,-1):
                    remapped_labels.append(labels[i*8 + j])

            final_images.append((image, remapped_labels))
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    return final_images
    
def getTrainingData():
    """Fetches all the training data"""

    print("Loading my images...")
    imagesMy = get_myImages()
    print(f"Loaded {len(imagesMy)} of my images.")
    
    print("Loading chessRed images...")
    imagesRed = get_chessRedImages()

    print(f"Loaded {len(imagesRed)} chessRed images.")

    return (imagesRed, imagesMy)


def dataPreprocessingRed(data):
    """Resize images and convert labels to integers"""

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


    #Reize images, grey scale, normalise
    for item in data:
        image, labels = item
        resized_img = cv2.resize(image, (512, 512))

        intLabels = []
        for label in labels:
            intLabels.append(piece_to_int[label])
        new_data.append((resized_img, intLabels))
    
    return new_data

augmentation_pipeline_128 = A.Compose([
    A.Rotate(limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
    A.ImageCompression(quality_range=(60, 100), p=0.3),
    A.GaussianBlur(blur_limit=(3,5), p=0.2),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.25),
    A.ElasticTransform(alpha=1, sigma=5, p=0.1),
    A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.1),
])

def clearProcessedFolder(validation=False):
    """Clears the processed folder of old images before saving new ones"""
    save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processed_end_to_end"

    if validation:
        save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processedValidation_end_to_end"
    print("Removing old processed images...")
    for filename in os.listdir(save_path):
        os.remove(os.path.join(save_path, filename))
    
    print("Finished clearing old images.")

def savePreprocessedData(images,validation=False):
    """Saves the preprocessed images to a folder and returns the paths to those images"""
    
    save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processed_end_to_end"

    if validation:
        save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processedValidation_end_to_end"
    count = 0
    paths = []
    for img, label in images:
        filename = f"img_{count}.png"
        cv2.imwrite(os.path.join(save_path, filename), img)
        paths.append(os.path.join(save_path, filename))
        count += 1
    return paths
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
def loadImage(path,label,augment):
    """Loads an image from a path and applies augmentation if not validation data"""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    if augment:
        img = augment_tf(img)
    else:
        img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label

def denseNetPieceRecognitionModel(train_dataset, val_dataset, num_classes=13, epochs=60, batch_size=32,class_weight_dict=None):
    """Trains the piece recognition model using a DenseNet121 base and custom top layers"""
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    base_model.trainable = False
    num_squares = 64

    inputs = tf.keras.Input(shape=(512, 512, 3))
    x = base_model(inputs, training=False)
    x = layers.Conv2D(512, (3,3), padding="same", activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Resizing(8, 8)(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Conv2D(num_classes, (1,1), padding="same", activation="softmax")(x)
    outputs = layers.Reshape((num_squares, num_classes))(x)
    model = tf.keras.Model(inputs, outputs)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6
    )
    checkpoint_frozen = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model_frozen.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history_frozen = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[reduce_lr, checkpoint_frozen]
    )

    model = tf.keras.models.load_model('best_model_frozen.keras')
    best_frozen_loss = min(history_frozen.history['val_loss'])
    print(f"Best frozen val_loss: {best_frozen_loss:.5f}")

    # Fine Tuning Phase
    base_model = model.get_layer('densenet121')
    base_model.trainable = True
    for layer in base_model.layers[:-50]:  # freeze all but last 50 layers
        layer.trainable = False

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    checkpoint_finetuned = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model_finetuned.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        initial_value_threshold=best_frozen_loss,
        verbose=1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_finetuned = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=300,
        callbacks=[ checkpoint_finetuned]
    )

    # Return the best model based on validation loss across both phases
    if os.path.exists('best_model_finetuned.keras'):
        model = tf.keras.models.load_model('best_model_finetuned.keras')
    else:
        model = tf.keras.models.load_model('best_model_frozen.keras')

    return (model,min(history_finetuned.history['val_loss']))
def trainModel():
    """Trains the end-to-end model, including data loading, preprocessing, and model training"""
    clearProcessedFolder(validation=False)
    clearProcessedFolder(validation=True)
    print("Loading training data...")
    (training_dataRed, training_dataMy) = getTrainingData()
    print(f"Loaded {len(training_dataRed) + len(training_dataMy)} total training samples.")

    print("Splitting data into training and validation sets...")

    training_my, val_my = train_test_split(training_dataMy, test_size=0.2, random_state=42)
    training_red, val_red = train_test_split(training_dataRed, test_size=0.2, random_state=42)

    print("Preprocessing data...")
    preprocessed_dataRed = dataPreprocessingRed(training_red)
    preprocessed_dataMy = dataPreprocessingRed(training_my)

    preprocessed_dataRed_Val = dataPreprocessingRed(val_red)
    preprocessed_dataMy_Val = dataPreprocessingRed(val_my)


    print(f"Preprocessed data. Total samples after augmentation: {len(preprocessed_dataRed) + len(preprocessed_dataMy)}")
    
    print("Getting ready to train the model...")

    all =  preprocessed_dataMy + preprocessed_dataRed

    all_val =   preprocessed_dataMy_Val + preprocessed_dataRed_Val

    labels = []
    for image, label in all:
        labels.append(label)
    
    labels_val = []
    for image, label in all_val:
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
               .map(lambda path, label: (loadImage(path, label,True)), num_parallel_calls=tf.data.AUTOTUNE)
               .shuffle(buffer_size=1000)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (val_dataset
               .map(lambda path, label: (loadImage(path, label,False)), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )

    model,val = denseNetPieceRecognitionModel(train_ds,val_ds)
    return (model,val)

def testModel(model, weight):
    """Tests the end-to-end model on the testing set and returns the accuracy"""
    model.load_weights(weight)

    test_data = get_testingImages()
    test_data = dataPreprocessingRed(test_data)

    labels_test = [label for _, label in test_data]
    paths_test = savePreprocessedData(test_data, validation=True)

    paths_test = tf.constant(paths_test)
    labels_test = tf.constant(labels_test)

    test_ds = (tf.data.Dataset.from_tensor_slices((paths_test, labels_test))
               .map(lambda path, label: loadImage(path, label, True), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE)
    )

    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")
    return accuracy

def makePredictions(model, weight, image):
    """Makes predictions on a single image using the trained model and returns the predicted labels and probabilities"""
    image = cv2.imread(image)
    model.load_weights(weight)
    image = cv2.resize(image, (512, 512))

    # Save to disk and run through pipeline like training does
    dummy_data = [(image, [0] * 64)]
    paths = savePreprocessedData(dummy_data, validation=True)

    paths_tf = tf.constant(paths)
    labels_tf = tf.constant([[0] * 64])

    pred_ds = (tf.data.Dataset.from_tensor_slices((paths_tf, labels_tf))
               .map(lambda path, label: loadImage(path, label, False), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(1)
               .prefetch(tf.data.AUTOTUNE)
    )

    predictions = model.predict(pred_ds)

    predicted_labels = np.argmax(predictions, axis=-1)[0]
    return (predicted_labels, predictions)


def getValidationAccuracy():
    """Evaluates the model on the validation set and returns the accuracy, F1 score, and confusion matrix"""

    model = tf.keras.models.load_model('end_to_end_model.h5')
    model.load_weights('end_to_end_weights.h5')

    (training_dataRed, training_dataMy) = getTrainingData()

    training_my, val_my = train_test_split(training_dataMy, test_size=0.2, random_state=42)
    training_red, val_red = train_test_split(training_dataRed, test_size=0.2, random_state=42)

    preprocessed_train_my = dataPreprocessingRed(training_my)
    preprocessed_train_red = dataPreprocessingRed(training_red)
    preprocessed_val_my = dataPreprocessingRed(val_my)
    preprocessed_val_red = dataPreprocessingRed(val_red)

    all_train = preprocessed_train_my + preprocessed_train_red
    all_val = preprocessed_val_my + preprocessed_val_red

    #Gets training and validation data ready for evaluation by saving the preprocessed images to disk and creating tf datasets from those saved images, just like the training pipeline does
    labels_train = [label for _, label in all_train]
    paths_train = savePreprocessedData(all_train, validation=False)

    paths_train_tf = tf.constant(paths_train)
    labels_train_tf = tf.constant(labels_train)

    train_ds = (tf.data.Dataset.from_tensor_slices((paths_train_tf, labels_train_tf))
                .map(lambda path, label: loadImage(path, label, False), num_parallel_calls=tf.data.AUTOTUNE)
                .batch(32)
                .prefetch(tf.data.AUTOTUNE)
    )

    labels_val = [label for _, label in all_val]
    paths_val = savePreprocessedData(all_val, validation=True)

    paths_val_tf = tf.constant(paths_val)
    labels_val_tf = tf.constant(labels_val)

    val_ds = (tf.data.Dataset.from_tensor_slices((paths_val_tf, labels_val_tf))
              .map(lambda path, label: loadImage(path, label, False), num_parallel_calls=tf.data.AUTOTUNE)
              .batch(32)
              .prefetch(tf.data.AUTOTUNE)
    )

    train_loss, train_accuracy = model.evaluate(train_ds, verbose=1)
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")

    val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)
    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")


    # Collect predictions and true labels for the entire validation set

    all_preds = []
    all_true = []

    for images, label_batch in val_ds:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=-1)
        all_preds.extend(pred_classes.flatten().tolist())
        all_true.extend(label_batch.numpy().flatten().tolist())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    int_to_piece = {
        0: 'empty', 1: 'P', 2: 'N', 3: 'B', 4: 'R',
        5: 'Q',     6: 'K', 7: 'p', 8: 'n', 9: 'b',
        10: 'r',    11: 'q', 12: 'k'
    }

    f1 = f1_score(all_true, all_preds, average='weighted')
    f1_per_class = f1_score(all_true, all_preds, average=None, labels=list(range(13)))
    print(f"Weighted F1 Score: {f1:.4f}")

    print("\n--- Per-Class F1 Scores ---")
    for i, score in enumerate(f1_per_class):
        print(f"{int_to_piece[i]:<8}: {score:.4f}")

    print("\n--- Per-Class Accuracy Scores ---")
    for i in range(13):
        class_mask = (all_true == i)
        if class_mask.sum() == 0:
            print(f"  {int_to_piece[i]:<8}: No samples")
            continue
        class_acc = (all_preds[class_mask] == all_true[class_mask]).sum() / class_mask.sum()
        print(f"  {int_to_piece[i]:<8}: {class_acc:.4f}  ({class_mask.sum()} samples)")

    confusionMatrix = confusion_matrix(all_true, all_preds, labels=list(range(13)))
    display_labels = [int_to_piece[i] for i in range(13)]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title("Validation Confusion Matrix (End-to-End)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_end_to_end.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to confusion_matrix_end_to_end.png")

    return val_accuracy, f1, confusionMatrix

def SingleTrainingRun():
    model,val = trainModel()
    model.save("end_to_end_model.h5")
    model.save_weights("end_to_end_weights.h5")

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
            model.save("end_to_end_model.h5")
            model.save_weights("end_to_end_weights.h5")
            print(f"New best validation loss: {bestVal}")
            print(f"Test accuracy at this point: {testAcc}")
        if math.isclose(val, bestVal, rel_tol=1e-4):
            if testAcc > bestTestAcc:
                model.save("end_to_end_model.h5")
                model.save_weights("end_to_end_weights.h5")
                bestTestAcc = testAcc
            print(f"New best test accuracy: {bestTestAcc}")
        count -= 1

def UnseenTest():
    testModel(tf.keras.models.load_model("end_to_end_model.h5"), "end_to_end_weights.h5")