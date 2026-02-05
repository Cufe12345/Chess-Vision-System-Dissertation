import chess
import random
import cv2
import os
def print_ascii(fen):
    board = chess.Board(fen)
    print(board)   # python-chess prints a nice ASCII board
def loadFensFromFile(filename):
    listOfFens = []
    with open(filename, 'r') as file:
        for line in file:
            fen = line.strip()
            if fen:
                listOfFens.append(fen)
    return listOfFens

def cameraSetup():
    print("Initializing camera...")
    cam = cv2.VideoCapture(0)
    print("Camera initialized")
    return cam

def FEN_To_Filename(fen):
    return fen.replace("/", "_").replace(" ", "_")

def captureBoardImage(cam, fen):
    result, image = cam.read()
    if result:
        state = getGameStateFromFEN(fen)
        cv2.imwrite(f"C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\{state}\\{FEN_To_Filename(fen)}.jpg", image)
        print(f"Image Captured")
    else:
        print("Error capturing image")

def getGameStateFromFEN(fen):
    
    numOfPieces = {
        "queens":0,
        "rooks":0,
        "bishops":0,
        "knights":0,
        "pawns":0
    }

    pieceValues = {
        "queens":4,
        "rooks":2,
        "bishops":1,
        "knights":1,
    }
    for char in fen:
        if char == 'Q' or char == 'q':
            numOfPieces["queens"]+= 1
        elif char == 'R' or char == 'r':
            numOfPieces["rooks"] += 1
        elif char == 'B' or char == 'b':
            numOfPieces["bishops"] += 1
        elif char == 'N' or char == 'n':
            numOfPieces["knights"] += 1
        elif char == 'P' or char == 'p':
            numOfPieces["pawns"] += 1
    
    #stockfish method
    totalPhase = 24
    phase = 0

    for piece, value in pieceValues.items():
        phase += numOfPieces[piece] * value


    phase = (phase * 256 + (totalPhase / 2)) // totalPhase

    if phase < 64:
        return "endgame"
    elif phase <= 192:
        return "midgame"
    else:
        return "opening"

print("Performing setup...")

print("Loading FENs from file...")
fens = loadFensFromFile("C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\OutputFENs.txt")
print(f"Loaded {len(fens)} FENs Successfully")

camera = cameraSetup()

print("Setup complete. Displaying random chess positions:")
while True:
    fen = random.choice(fens)

    state = getGameStateFromFEN(fen)
    if os.path.exists(f"C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\{state}\\{FEN_To_Filename(fen)}.jpg"):
        continue

    print(f"Displaying position for FEN: {fen}")
    print(f"Game State: {state}")
    print_ascii(fen)
    input("Press Enter to capture image...")
    captureBoardImage(camera, fen)
    option = input("Press Anything to see another position or 'q' to quit: ")
    if option.lower() == 'q':
        break

