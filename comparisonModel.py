import numpy as np
import Visual_Representation
import PieceRecognitionModel
import EndToEndModel
import tensorflow as tf
import os
import pickle
from sklearn.linear_model import LogisticRegression


# --- Trained combiner ---

def train_combiner_from_data(X, y):
    from sklearn.linear_model import LogisticRegression
    import pickle
    combiner = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
    combiner.fit(X, y)
    with open("combiner_model.pkl", "wb") as f:
        pickle.dump(combiner, f)
    print("Combiner saved.")
    return combiner


def load_combiner():
    with open("combiner_model.pkl", "rb") as f:
        return pickle.load(f)


def evaluateOptimalModel(pipelinePrediction, endToEndPrediction, combiner=None):
    pipeline_probs = np.squeeze(np.array(pipelinePrediction[1]))   # (64, 13)
    e2e_probs = np.squeeze(np.array(endToEndPrediction[1]))        # (64, 13)

    if combiner is not None:
        # Use learned combiner
        features = np.concatenate([pipeline_probs, e2e_probs], axis=1)  # (64, 26)
        final_predictions = combiner.predict(features).astype(int)       # (64,)
        final_probs = combiner.predict_proba(features)                   # (64, 13)
    else:
        # Fallback to equal weighted average
        final_probs = (pipeline_probs * 0.8) + (e2e_probs * 0.2)
        final_predictions = np.argmax(final_probs, axis=1).flatten().astype(int)

    return final_predictions, final_probs


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

def get_combiner_training_data(pieceModel, pieceWeights, endToEndModel, endToEndWeights):
    import EndToEndModel as E2E
    import PieceRecognitionModel as PRM
    from sklearn.model_selection import train_test_split

    piece_to_int = {
        '': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4,
        'Q': 5, 'K': 6, 'p': 7, 'n': 8, 'b': 9,
        'r': 10, 'q': 11, 'k': 12
    }

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    # --- ChessRed ---
    folder_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\chessRed\\FinalImages"
    folder_path2 = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\fillers"
    all_red = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]
    all_red += [
        os.path.join(folder_path2, f)
        for f in os.listdir(folder_path2)
        if f.lower().endswith(image_extensions)
    ]
    _, val_red = train_test_split(all_red, test_size=0.2, random_state=42)

    # --- My images ---
    path_opening = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\opening"
    path_midgame = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\midgame"
    path_endgame = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\endgame"
    all_my = []
    for folder in [path_opening, path_midgame, path_endgame]:
        all_my += [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(image_extensions)]
    _, val_my = train_test_split(all_my, test_size=0.2, random_state=42)

    X, y = [], []

    def process_image_file(img_path, int_labels):
        try:
            pipe_pred = PRM.makePredictions(pieceModel, pieceWeights, img_path, color=True)
            e2e_pred = E2E.makePredictions(endToEndModel, endToEndWeights, img_path)

            pipeline_probs = np.squeeze(np.array(pipe_pred[1]))   # (64, 13)
            e2e_probs = np.squeeze(np.array(e2e_pred[1]))         # (64, 13)

            for sq in range(64):
                features = np.concatenate([pipeline_probs[sq], e2e_probs[sq]])  # (26,)
                X.append(features)
                y.append(int_labels[sq])
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    # Process chessRed val images
    print(f"Processing {len(val_red)} chessRed validation images...")
    for img_path in val_red:
        try:
            true_labels = fen_to_labels(os.path.basename(img_path).split('.')[0])
            int_labels = [piece_to_int[l] for l in true_labels]
            process_image_file(img_path, int_labels)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    # Process my val images with column remap
    print(f"Processing {len(val_my)} my validation images...")
    for img_path in val_my:
        try:
            labels = fen_to_labels(os.path.basename(img_path).split('.')[0])
            remapped = []
            for j in range(8):
                for i in range(7, -1, -1):
                    remapped.append(labels[i * 8 + j])
            int_labels = [piece_to_int[l] for l in remapped]
            process_image_file(img_path, int_labels)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    print(f"Collected {len(X)} square samples from {len(val_red) + len(val_my)} images.")
    return np.array(X), np.array(y)

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

pieceModel = tf.keras.models.load_model("piece_recognition_model.h5")
pieceWeights = "piece_recognition_weights.h5"
endToEndModel = tf.keras.models.load_model("end_to_end_model.h5")
endToEndWeights = "end_to_end_weights.h5"

combiner = None
if os.path.exists("combiner_model.pkl"):
    combiner = load_combiner()
    print("Loaded trained combiner.")
else:
    print("No combiner found, using equal weighted average.")

def prediction(img_path):
    pipe_pred = PieceRecognitionModel.makePredictions(pieceModel, pieceWeights, img_path, color=True)
    e2e_pred = EndToEndModel.makePredictions(endToEndModel, endToEndWeights, img_path)

    # visualisePredictions(pipe_pred[0], "pipeline_prediction.png")
    # visualisePredictions(e2e_pred[0], "end_to_end_prediction.png")

    final_predictions, final_probs = evaluateOptimalModel(pipe_pred, e2e_pred, combiner)
    print("Final:", final_predictions)
    # visualisePredictions(final_predictions, "final_prediction.png")

    return final_predictions, final_probs

# prediction("C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\testingImages\\rnbqk2r_ppp1bppp_3p1n2_4p3_2B1P3_P4N2_1PPP1PPP_RNBQK2R,w,KQkq,-,1,5.jpg") 
# X, y = get_combiner_training_data(pieceModel, pieceWeights, endToEndModel, endToEndWeights)
# combiner = train_combiner_from_data(X, y)