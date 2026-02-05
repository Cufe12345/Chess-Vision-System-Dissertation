import BoardSegmentation
import os
import matplotlib.pyplot as plt
import cv2
import albumentations as A

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
        print(f"Processing row: {row}")
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
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, p=0.3)
        ])
    #Reize images, grey scale, normalise
    for image in data:
        squares, labels = image
        for idx, sq in enumerate(squares):
            resized_sq = cv2.resize(sq, (64, 64))

            if data_augmentation:
                augmented = augmentation_pipeline(image=resized_sq)
                resized_sq = augmented['image']

            gray_sq = cv2.cvtColor(resized_sq, cv2.COLOR_BGR2GRAY)
            norm_sq = gray_sq / 255.0

            squares[idx] = norm_sq

        new_data.append((squares, labels))
    
    return new_data
def trainPieceRecognitionModel(training_data):
    # Placeholder for training logic
    pass

def trainModel():
    training_data = getTrainingData()
    preprocessed_data = dataPreprocessing(training_data)
    trainPieceRecognitionModel(preprocessed_data)

trainModel()