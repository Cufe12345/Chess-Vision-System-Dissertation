import comparisonModel
import cv2
import time as thread
import numpy as np
import Visual_Representation

def cameraSetup():
    print("Initializing camera...")
    cam = cv2.VideoCapture(0)
    print("Camera initialized")
    return cam

def captureBoardImage(cam):
    result, image = cam.read()
    # save the image to a temp file
    if result:
        cv2.imwrite(f"C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\temp.jpg", image)
        print(f"Image Captured")

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
    
    fen_string = "/".join(fen_rows)
    return fen_string

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

def MakePrediction(cam):
    previousImg = cv2.imread("C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\temp.jpg")
    if previousImg is not None:
        result, image = cam.read()
        if result:
            diff = np.mean(np.abs(image.astype(np.float32) - previousImg.astype(np.float32)))
            if diff > 0.0:  # tweak this threshold as needed
                print(f"Change detected (diff={diff:.2f}), making prediction...")
                captureBoardImage(cam)
                print("Making prediction...")
                final_predictions, final_probs = comparisonModel.prediction("C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\temp.jpg")
                print("Final Predictions: ", final_predictions)
                print("Final Probabilities: ", final_probs)
                visualisePredictions(final_predictions, "current_prediction.png")
            else:
                print(f"No significant change (diff={diff:.2f}), skipping prediction.")
    else:
        print("No previous image found, capturing initial image.")
        captureBoardImage(cam)
        print("Making initial prediction...")
        final_predictions, final_probs = comparisonModel.prediction("C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\temp.jpg")
        print("Final Predictions: ", final_predictions)
        print("Final Probabilities: ", final_probs)
        visualisePredictions(final_predictions, "initial_prediction.png")

print("Performing setup...")

previousImg = None
camera = cameraSetup()
print("Setup complete. Displaying random chess positions:")
while True:
    user_input = input("Press Enter to predict (or 'q' to quit): ")
    if user_input.lower() == 'q':
        print("Exiting...")
        camera.release()
        break
    MakePrediction(camera)
