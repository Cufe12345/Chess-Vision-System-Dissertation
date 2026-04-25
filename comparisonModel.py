import numpy as np
import Visual_Representation
import PieceRecognitionModel
import EndToEndModel
import tensorflow as tf
import os
import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
import EndToEndModel as E2E
import PieceRecognitionModel as PRM
import tensorflow as tf


def trainCombiner(X, y):
    combiner = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
    combiner.fit(X, y)
    with open("combiner_model.pkl", "wb") as f:
        pickle.dump(combiner, f)
    print("Combiner saved.")


def loadCombiner():
    with open("combiner_model.pkl", "rb") as f:
        return pickle.load(f)


def evaluateOptimalModel(pipelinePrediction, endToEndPrediction, combiner=None):
    pipeline_probs = np.squeeze(np.array(pipelinePrediction[1]))
    e2e_probs = np.squeeze(np.array(endToEndPrediction[1]))

    features = np.concatenate([pipeline_probs, e2e_probs], axis=1)
    final_predictions = combiner.predict(features).astype(int)
    final_probs = combiner.predict_proba(features)

    return final_predictions, final_probs


def fen_to_labels(fen):
    board_part = fen.split(',')[0]

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

def getCombinerTrainingData(pieceModel, pieceWeights, endToEndModel, endToEndWeights, returnFeatures=True):

    piece_to_int = {
        '': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4,
        'Q': 5, 'K': 6, 'p': 7, 'n': 8, 'b': 9,
        'r': 10, 'q': 11, 'k': 12
    }

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    #ChessReD
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

    #My Data
    path_opening = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\opening"
    path_midgame = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\midgame"
    path_endgame = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\endgame"
    all_my = []
    for folder in [path_opening, path_midgame, path_endgame]:
        all_my += [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(image_extensions)]
    _, val_my = train_test_split(all_my, test_size=0.2, random_state=42)


    if not returnFeatures:
        # Return raw paths + int labels per square
        def get_labels(img_path, remap=False):
            labels = fen_to_labels(os.path.basename(img_path).split('.')[0])
            if remap:
                remapped = []
                for j in range(8):
                    for i in range(7, -1, -1):
                        remapped.append(labels[i * 8 + j])
                labels = remapped
            return [piece_to_int[l] for l in labels]

        val_data_red = [(p, get_labels(p)) for p in val_red]
        val_data_my = [(p, get_labels(p, remap=True)) for p in val_my]
        return val_data_red, val_data_my
    
    X, y = [], []

    def process_image_file(img_path, int_labels):
        try:
            pipeline_pred = PieceRecognitionModel.makePredictions(pieceModel, pieceWeights, img_path)
            endToEnd_pred = EndToEndModel.makePredictions(endToEndModel, endToEndWeights, img_path)

            pipeline_probs = np.squeeze(np.array(pipeline_pred[1]))
            e2e_probs = np.squeeze(np.array(endToEnd_pred[1]))

            for sq in range(64):
                features = np.concatenate([pipeline_probs[sq], e2e_probs[sq]])
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

    print(f"Collected {len(X)} square samples from {len(val_red) + len(val_my)} images")
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

def prediction(img_path):

    combiner = None
    if os.path.exists("combiner_model.pkl"):
        combiner = loadCombiner()
        print("Loaded trained combiner.")
    else:
        print("No trained combiner found")
        return
    
    pieceModel = tf.keras.models.load_model("piece_recognition_model.h5")
    pieceWeights = "piece_recognition_weights.h5"
    endToEndModel = tf.keras.models.load_model("end_to_end_model.h5")
    endToEndWeights = "end_to_end_weights.h5"
    if pieceModel is None or endToEndModel is None:
        print("Error loading models. Ensure piece_recognition_model.h5 and end_to_end_model.h5 exist.")
        return
    if pieceWeights is None or endToEndWeights is None:
        print("Error loading weights. Ensure piece_recognition_weights.h5 and end_to_end_weights.h5 exist.")
        return
    # SHow original image
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Input Image")
    plt.show()

    pipeline_pred = PieceRecognitionModel.makePredictions(pieceModel, pieceWeights, img_path)
    endToEnd_pred = EndToEndModel.makePredictions(endToEndModel, endToEndWeights, img_path)

    final_predictions, final_probs = evaluateOptimalModel(pipeline_pred, endToEnd_pred, combiner)
    print("Final:", final_predictions)
    visualisePredictions(final_predictions, "final_prediction.png")

    return final_predictions, final_probs


def evaluateCombinerOnValidation():

    pieceModel = tf.keras.models.load_model("piece_recognition_model.h5")
    pieceWeights = "piece_recognition_weights.h5"
    endToEndModel = tf.keras.models.load_model("end_to_end_model.h5")
    endToEndWeights = "end_to_end_weights.h5"
    if pieceModel is None or endToEndModel is None:
        print("Error loading models. Ensure piece_recognition_model.h5 and end_to_end_model.h5 exist.")
        return
    if pieceWeights is None or endToEndWeights is None:
        print("Error loading weights. Ensure piece_recognition_weights.h5 and end_to_end_weights.h5 exist.")
        return

    combiner = None
    if os.path.exists("combiner_model.pkl"):
        combiner = loadCombiner()
        print("Loaded trained combiner.")
    else:
        print("No trained combiner found")
        return

    piece_to_int = {
        '': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4,
        'Q': 5, 'K': 6, 'p': 7, 'n': 8, 'b': 9,
        'r': 10, 'q': 11, 'k': 12
    }
    int_to_piece = {v: k if k != '' else 'empty' for k, v in piece_to_int.items()}
    int_to_piece[0] = 'empty'

    val_red, val_my = getCombinerTrainingData(pieceModel, pieceWeights, endToEndModel, endToEndWeights, returnFeatures=False)
    
    all_preds = []
    all_true = []
    # Also track per-model predictions for comparison
    pipeline_preds_all = []
    e2e_preds_all = []

    def process_image(img_path, int_labels):
        try:
            pipe_pred = PRM.makePredictions(pieceModel, pieceWeights, img_path)
            e2e_pred  = E2E.makePredictions(endToEndModel, endToEndWeights, img_path)

            final_predictions, _ = evaluateOptimalModel(pipe_pred, e2e_pred, combiner)

            pipeline_preds_all.extend(pipe_pred[0].flatten().tolist())
            e2e_preds_all.extend(np.argmax(np.squeeze(e2e_pred[1]), axis=-1).flatten().tolist())
            all_preds.extend(final_predictions.flatten().tolist())
            all_true.extend(int_labels)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    print(f"Processing {len(val_red)} chessRed validation images...")
    for img_path in val_red:
        img_path = img_path[0]
        try:
            true_labels = fen_to_labels(os.path.basename(img_path).split('.')[0])
            int_labels = [piece_to_int[l] for l in true_labels]
            process_image(img_path, int_labels)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    print(f"Processing {len(val_my)} my validation images...")
    for img_path in val_my:
        img_path = img_path[0]
        try:
            labels = fen_to_labels(os.path.basename(img_path).split('.')[0])
            remapped = []
            for j in range(8):
                for i in range(7, -1, -1):
                    remapped.append(labels[i * 8 + j])
            int_labels = [piece_to_int[l] for l in remapped]
            process_image(img_path, int_labels)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    all_preds        = np.array(all_preds)
    all_true         = np.array(all_true)
    pipeline_preds_all = np.array(pipeline_preds_all)
    e2e_preds_all    = np.array(e2e_preds_all)


    combined_acc  = accuracy_score(all_true, all_preds)
    pipeline_acc  = accuracy_score(all_true, pipeline_preds_all)
    e2e_acc       = accuracy_score(all_true, e2e_preds_all)

    print(f"\n--- Overall Accuracy ---")
    print(f"  Pipeline only:  {pipeline_acc:.4f}")
    print(f"  End-to-end only:{e2e_acc:.4f}")
    print(f"  Combined:       {combined_acc:.4f}")


    combined_f1 = f1_score(all_true, all_preds, average='weighted')
    pipeline_f1 = f1_score(all_true, pipeline_preds_all, average='weighted')
    e2e_f1      = f1_score(all_true, e2e_preds_all, average='weighted')

    print(f"\n--- Weighted F1 Score ---")
    print(f"  Pipeline only:  {pipeline_f1:.4f}")
    print(f"  End-to-end only:{e2e_f1:.4f}")
    print(f"  Combined:       {combined_f1:.4f}")

    combined_f1_per  = f1_score(all_true, all_preds,          average=None, labels=list(range(13)))
    pipeline_f1_per  = f1_score(all_true, pipeline_preds_all, average=None, labels=list(range(13)))
    e2e_f1_per       = f1_score(all_true, e2e_preds_all,      average=None, labels=list(range(13)))

    print(f"\n--- Per-Class Accuracy & F1 (Pipeline | E2E | Combined) ---")
    print(f"  {'Class':<8} {'Pipe Acc':>9} {'E2E Acc':>9} {'Comb Acc':>9} {'Pipe F1':>9} {'E2E F1':>9} {'Comb F1':>9}")
    print("  " + "-" * 62)
    for i in range(13):
        mask = (all_true == i)
        if mask.sum() == 0:
            print(f"  {int_to_piece[i]:<8} {'N/A':>9} {'N/A':>9} {'N/A':>9}")
            continue
        p_acc = (pipeline_preds_all[mask] == all_true[mask]).sum() / mask.sum()
        e_acc = (e2e_preds_all[mask]      == all_true[mask]).sum() / mask.sum()
        c_acc = (all_preds[mask]          == all_true[mask]).sum() / mask.sum()
        print(f"  {int_to_piece[i]:<8} {p_acc:>9.4f} {e_acc:>9.4f} {c_acc:>9.4f} "
              f"{pipeline_f1_per[i]:>9.4f} {e2e_f1_per[i]:>9.4f} {combined_f1_per[i]:>9.4f}  ({mask.sum()} samples)")

    cm = confusion_matrix(all_true, all_preds, labels=list(range(13)))
    display_labels = [int_to_piece[i] for i in range(13)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title("Combined Model - Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_combined.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to confusion_matrix_combined.png")

    return combined_acc, combined_f1, cm

def SingleTrainingRun():
    pieceModel = tf.keras.models.load_model("piece_recognition_model.h5")
    pieceWeights = "piece_recognition_weights.h5"
    endToEndModel = tf.keras.models.load_model("end_to_end_model.h5")
    endToEndWeights = "end_to_end_weights.h5"
    if pieceModel is None or endToEndModel is None:
        print("Error loading models. Ensure piece_recognition_model.h5 and end_to_end_model.h5 exist.")
        return
    if pieceWeights is None or endToEndWeights is None:
        print("Error loading weights. Ensure piece_recognition_weights.h5 and end_to_end_weights.h5 exist.")
        return
    X, y = getCombinerTrainingData(pieceModel, pieceWeights, endToEndModel, endToEndWeights)
    trainCombiner(X, y)
