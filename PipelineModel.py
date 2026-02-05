import BoardSegmentation
import PieceRecognitionModel
import matplotlib.pyplot as plt
import cv2

def processChessboardImage(image_path):
    # Step 1: Segment the chessboard into squares
    squares = BoardSegmentation.getSquaresFromImage(image_path)
    
    #display the detected squares for debugging
    for idx,sq in enumerate(squares):
            
            plt.figure(figsize=(2,2))  # optional: control size
            plt.imshow(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"Square {idx}")
            plt.show()
    



    # Step 2: Recognize pieces in each square
    piece_labels = PieceRecognitionModel.recognizePieces(squares)
    
    return piece_labels

processChessboardImage("C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\test3.jpg")