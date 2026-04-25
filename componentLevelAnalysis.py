import BoardSegmentation
import os
def boardSegmentationAndLineSegmentation():
    path_opening = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\opening"
    path_midgame = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\midgame"
    path_endgame = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\endgame"

    folder_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\chessRed\\FinalImages"
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    folder_path_fillers = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\fillers"

    image_files_fillers = [
        os.path.join(folder_path_fillers, f)
        for f in os.listdir(folder_path_fillers)
        if f.lower().endswith(image_extensions)
    ]

    image_files_My = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]


    image_filesO = [
        os.path.join(path_opening, f)
        for f in os.listdir(path_opening)
        if f.lower().endswith(image_extensions)
    ]

    image_filesM = [
        os.path.join(path_midgame, f)
        for f in os.listdir(path_midgame)
        if f.lower().endswith(image_extensions)
    ]

    image_filesE = [
        os.path.join(path_endgame, f)
        for f in os.listdir(path_endgame)
        if f.lower().endswith(image_extensions)
    ]


    image_files = image_filesO + image_filesM + image_filesE + image_files_fillers + image_files_My

    offset = 0
    perfectBoardCount = 0
    perfectLineCount = 0
    nearPerfectBoardCount = 0
    nearPerfectLineCount = 0
    failedBoardCount = 0
    failedLineCount = 0
    for i, image in enumerate(image_files):
        if i < offset:
            continue
        print(f"Processing image {i+1}/{len(image_files)}: {image}")
        try:
            if i > 87:
                squares = BoardSegmentation.getSquaresFromImage(image, debug=False,colour=True, farChessTable=True)
            else:
                squares = BoardSegmentation.getSquaresFromImage(image, debug=False,colour=True)
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            continue
        result = int(input("Enter board result 1 for perfect, 2 for near perfect, 3 for failed: 4 for exit"))
        if result == 1:
            perfectBoardCount += 1
        elif result == 2:
            nearPerfectBoardCount += 1
        elif result == 3:
            failedBoardCount += 1
        elif result == 4:
            print(f"Perfect Boards: {perfectBoardCount}")
            print(f"Near Perfect Boards: {nearPerfectBoardCount}")
            print(f"Failed Boards: {failedBoardCount}")
            print(f"Perfect Line Segmentations: {perfectLineCount}")
            print(f"Near Perfect Line Segmentations: {nearPerfectLineCount}")
            print(f"Failed Line Segmentations: {failedLineCount}")
            print(f"Total Images Processed: {len(image_files) - offset}")
            break
        line_result = int(input("Enter line segmentation result 1 for perfect, 2 for near perfect, 3 for failed: "))
        if line_result == 1:
            perfectLineCount += 1
        elif line_result == 2:
            nearPerfectLineCount += 1
        else:
            failedLineCount += 1
    print(f"Perfect Boards: {perfectBoardCount}")
    print(f"Near Perfect Boards: {nearPerfectBoardCount}")
    print(f"Failed Boards: {failedBoardCount}")
    print(f"Perfect Line Segmentations: {perfectLineCount}")
    print(f"Near Perfect Line Segmentations: {nearPerfectLineCount}")
    print(f"Failed Line Segmentations: {failedLineCount}")
    print(f"Total Images Processed: {len(image_files) - offset}")

print(f"Perfect Boards: 20")
print(f"Near Perfect Boards: 140")
print(f"Failed Boards: 3")
print(f"Perfect Line Segmentations: 163")
print(f"Near Perfect Line Segmentations: 0")
print(f"Failed Line Segmentations: 0")
print(f"Total Images Processed: 163")
