import BoardSegmentation
import PieceRecognitionModel
import matplotlib.pyplot as plt
import cv2
import os

def processChessboardImage(image_path):
    # Step 1: Segment the chessboard into squares
    squares = BoardSegmentation.getSquaresFromImage(image_path)
    if len(squares) != 64:
        print(f"image name: {image_path} - Warning: Expected 64 squares, but detected {len(squares)}. Check the segmentation results.")
        raise ValueError("Incorrect number of squares detected. Expected 64.")

    #display the detected squares for debugging
    # for idx,sq in enumerate(squares):
            
    #         plt.figure(figsize=(2,2))  # optional: control size
    #         plt.imshow(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
    #         plt.axis('off')
    #         plt.title(f"Square {idx}")
    #         plt.show()
    



    # Step 2: Recognize pieces in each square
    piece_labels = PieceRecognitionModel.recognizePieces(squares)
    
    return piece_labels

def get_chessRedImages():
    folder_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\chessRed\\FinalImages"
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]
    for image in image_files:
        print(f"Processing image: {image}")
        try:
            piece_labels = processChessboardImage(image)
            # print(f"Piece labels for {image}: {piece_labels}")
        except Exception as e:
            print(f"Error processing {image}: {e}")

# processChessboardImage("C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\test2.jpg")
get_chessRedImages()