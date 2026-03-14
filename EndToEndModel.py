import BoardSegmentation
import Visual_Representation
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
def get_chessRenderImages():
    folder_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\ChessRender360\\ChessRender360\\ChessRender360\\rgb"
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]

    images = []
    labelsTrain = []
    imagesVal = []
    labelsVal = []
    total = len(image_files)

    for i, image in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}")
        if i > 1000:
            break
        try:
            labels = fen_to_labels(os.path.basename(image).split('.')[0])
            image = cv2.imread(image)
            preproccessed = dataPreprocessingRed([(image, labels)], data_augmentation=False)
            preproccessed = preproccessed[0]  # get the single item from the list
            validation = False
            if i*10/total > 0.8:
                validation = True
            path = saveDataChessRender(preproccessed[0], i, validation=validation)
            if validation:
                imagesVal.append(path)
                labelsVal.append(preproccessed[1])
                continue
            images.append(path)
            labelsTrain.append(preproccessed[1])
            

            # for testing purpose only get 3
            # if len(images) >= 3:
            #     break
            # print(f"Piece labels for {image}: {piece_labels}")
        except Exception as e:
            print(f"Error processing {image}: {e}")
            continue

    return (images,labelsTrain, imagesVal, labelsVal)





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

    print("Loading chessRender images...")
    # imagesChessRender = get_chessRenderImages()
    imagesChessRender = [[None],[None],[None],[None]]
    print(f"Loaded {len(imagesChessRender)} chessRender images.")

    return (imagesRed, imagesMy, imagesChessRender)


def dataPreprocessingRed(data,data_augmentation=False):
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
        resized_img = cv2.resize(image, (512, 512))
        if data_augmentation:
            # grey_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            grey_img = resized_img
        else:
            grey_img = resized_img

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
        if data_augmentation:
            # norm_img = grey_img / 255.0
            # norm_img = norm_img[...,None]
            norm_img = grey_img
        else:
            norm_img = grey_img
        intLabels = []
        for label in labels:
            intLabels.append(piece_to_int[label])
        new_data.append((norm_img, intLabels))

        if data_augmentation:
            for _ in range(NUM_AUGMENTATIONS):
                

                resized_aug = cv2.resize(image, (512, 512))
                augmented = augmentation_pipeline(image=resized_aug)['image']
                # grey_aug = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
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
                
                # norm_aug = grey_aug / 255.0
                # norm_aug = norm_aug[...,None] # add channel dimension
                new_data.append((augmented, intLabels))
    
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

augmentation_pipeline_128 = A.Compose([
    A.Rotate(limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
    # A.GaussNoise(std_range=(0.05, 0.1), p=0.2),
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

def clearProcessedFolder(validation=False):
    save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processed_end_to_end"

    if validation:
        save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processedValidation_end_to_end"
    print("Removing old processed images...")
    for filename in os.listdir(save_path):
        os.remove(os.path.join(save_path, filename))
    
    print("Finished clearing old images.")
def saveDataChessRender(image,count,validation=False):
    # clear processed folder first
    
    save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processed_end_to_end"

    if validation:
        save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processedValidation_end_to_end"
    filename = f"img_{count}_Render.png"
    cv2.imwrite(os.path.join(save_path, filename), image)
    path = os.path.join(save_path, filename)
    return path
def savePreprocessedData(images,validation=False):
    # clear processed folder first
    
    save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processed_end_to_end"

    if validation:
        save_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\processedValidation_end_to_end"
    # print("Removing old processed images...")
    # for filename in os.listdir(save_path):
    #     os.remove(os.path.join(save_path, filename))
    
    # print("Finished clearing old images. Saving new preprocessed images...")
    count = 0
    paths = []
    for img, label in images:
        filename = f"img_{count}.png"
        cv2.imwrite(os.path.join(save_path, filename), img)
        paths.append(os.path.join(save_path, filename))
        count += 1
    return paths
def augment_numpy(img):
    img = img.numpy()

    img = img.astype(np.uint8)

    augmented = augmentation_pipeline_128(image=img)["image"]
    return augmented.astype(np.float32)
def augment_tf(img):
    img = tf.py_function(
        func=augment_numpy,
        inp=[img],
        Tout=tf.float32
    )

 # important
    return img
def loadImage(path,label,augment):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    if augment:
        img = augment_tf(img)
    else:
        img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label
def denseNetPieceRecognitionModel(train_dataset, val_dataset, num_classes=13, epochs=60, batch_size=32,class_weight_dict=None):
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

    # --- Fine-tuning ---
    # Access base model by name rather than index, which breaks after reload
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # was 1e-6, too slow
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_finetuned = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=300,
        callbacks=[early_stop, checkpoint_finetuned]
    )

    if os.path.exists('best_model_finetuned.keras'):
        model = tf.keras.models.load_model('best_model_finetuned.keras')
    else:
        model = tf.keras.models.load_model('best_model_frozen.keras')

    return model
def trainModel():

    clearProcessedFolder(validation=False)
    clearProcessedFolder(validation=True)
    print("Loading training data...")
    (training_dataRed, training_dataMy, training_dataChessRender) = getTrainingData()
    chessRenderPaths, chessRenderLabels, chessRenderPathsVal, chessRenderLabelsVal = training_dataChessRender
    print(f"Loaded {len(training_dataRed) + len(training_dataMy) + len(chessRenderPaths)} total training samples.")

    print("Splitting data into training and validation sets...")

    training_my, val_my = train_test_split(training_dataMy, test_size=0.2, random_state=42)
    training_red, val_red = train_test_split(training_dataRed, test_size=0.2, random_state=42)
    # training_chessRender, val_chessRender = train_test_split(training_dataChessRender, test_size=0.2, random_state=42)
    print("Preprocessing data...")
    preprocessed_dataRed = dataPreprocessingRed(training_red, data_augmentation=False)
    preprocessed_dataMy = dataPreprocessingRed(training_my, data_augmentation=False)
    # preprocessed_chessRender = dataPreprocessingRed(training_chessRender, data_augmentation=False)
    preprocessed_dataRed_Val = dataPreprocessingRed(val_red, data_augmentation=False)
    preprocessed_dataMy_Val = dataPreprocessingRed(val_my, data_augmentation=False)
    # preprocessed_chessRender_Val = dataPreprocessingRed(val_chessRender, data_augmentation=False)

    print(f"Preprocessed data. Total samples after augmentation: {len(preprocessed_dataRed) + len(preprocessed_dataMy)}")
    
    print("Getting ready to train the model...")

    all =  preprocessed_dataMy + preprocessed_dataRed

    all_val =   preprocessed_dataMy_Val + preprocessed_dataRed_Val

    # x = []
    # y = []

    # for image, label in all_preprocessed_data:
    #     x.append(image)
    #     y.append(label)

    # print(np.unique(y))
    # print(y[:20])


    # # Convert to numpy arrays
    # x = np.array(x, dtype="float32")
    # y = np.array(y, dtype="int32")
    # print(np.unique(y))
    # print(y[:20])

    labels = []
    for image, label in all:
        labels.append(label)
    
    labels_val = []
    for image, label in all_val:
        labels_val.append(label)
    
    paths = savePreprocessedData(all)
    paths_val = savePreprocessedData(all_val, validation=True)

    # paths += chessRenderPaths
    # labels += chessRenderLabels
    # paths_val += chessRenderPathsVal
    # labels_val += chessRenderLabelsVal

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
    # model = trainEndToEndModel(x,y)
    model = denseNetPieceRecognitionModel(train_ds,val_ds)
    model.save("end_to_end_model.h5")
    model.save_weights("end_to_end_weights.h5")

    # testModel(model)

def testModel(model, weight):
    model.load_weights(weight)

    test_data = get_testingImages()
    test_data = dataPreprocessingRed(test_data, data_augmentation=False)

    labels_test = [label for _, label in test_data]
    paths_test = savePreprocessedData(test_data, validation=True)

    paths_test = tf.constant(paths_test)
    labels_test = tf.constant(labels_test)

    test_ds = (tf.data.Dataset.from_tensor_slices((paths_test, labels_test))
               .map(lambda path, label: loadImage(path, label, False), num_parallel_calls=tf.data.AUTOTUNE)
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


def makePredictions(model, weight, image):
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

    predictions = model.predict(pred_ds)  # (1, 64, 13)

    predicted_labels = np.argmax(predictions, axis=-1)[0]  # (64,)
    probs = np.max(predictions, axis=-1)[0]                # (64,)

    print(predicted_labels)
    print(probs)
    return (predicted_labels, predictions)
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
    Visual_Representation.visualize_fen(fen_string,filePath)
# trainModel()
# testModel(tf.keras.models.load_model("end_to_end_model.h5"), "end_to_end_weights.h5")
# visualisePredictions(makePredictions(tf.keras.models.load_model("end_to_end_model.h5"), "end_to_end_weights.h5", "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\testingImages\\rnbqk2r_ppp1bppp_3p1n2_4p3_2B1P3_P4N2_1PPP1PPP_RNBQK2R,w,KQkq,-,1,5.jpg")[0], "end_to_end_prediction.png")

def test():
    fen = "1k6_1P2PK2_8_2bB4_8_8_8_8,b,-,-,0,70"
    print(fen)
    labels = fen_to_labels(fen)
    print(labels)
    fen_string = labels_to_fen(labels)
    print(fen_string)
# test()