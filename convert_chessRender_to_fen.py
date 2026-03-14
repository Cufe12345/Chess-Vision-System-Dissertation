import os
import csv
import re
import cv2
import shutil
import matplotlib.pyplot as plt
fenPath = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\ChessRender360\\ChessRender360\\ChessRender360\\FENs.csv"
rgbFolderPath = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\ChessRender360\\ChessRender360\\ChessRender360\\rgb"
def rgb_sort_key(path):
    filename = os.path.basename(path)
    number = int(filename.split("_")[1].split(".")[0])
    return number

def rename_images_to_fen():
    image_files = sorted([
        os.path.join(rgbFolderPath, f)
        for f in os.listdir(rgbFolderPath)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ],key=rgb_sort_key)

    listFens = []
    with open(fenPath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            print("Row: ", row)
            listFens.append(row)
    print(listFens)
    dictFilenames = {}
    for i, image in enumerate(image_files):
        fen = listFens[i]
        print(f"Image: {image} - FEN: {fen}")
        #rename the image file to match the FEN format
        new_filename = fen[0].replace("/", "_").replace(" ", ",") + ".jpg"
        new_filepath = os.path.join(rgbFolderPath, new_filename)
        if os.path.exists(new_filepath):
            if new_filename in dictFilenames:
                dictFilenames[new_filename] = dictFilenames[new_filename] + 1
            else:
                dictFilenames[new_filename] = 1
            new_filename = new_filename.replace(".jpg", f"_{dictFilenames[new_filename]}.jpg")
            new_filepath = os.path.join(rgbFolderPath, new_filename)
        os.rename(image, new_filepath)


rgbFolderPath = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\ChessRender360\\ChessRender360\\ChessRender360\\rgb"
rgbFinalPath = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\ChessRender360\\ChessRender360\\ChessRender360\\rgbFinal"

offset = 752  # change if resuming

images = sorted([
    f for f in os.listdir(rgbFolderPath)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

for i in range(offset, len(images)):

    filename = images[i]
    path = os.path.join(rgbFolderPath, filename)

    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Image {i} / {len(images)} : {filename}")
    ax.axis("off")

    key_pressed = {"key": None}

    def on_key(event):
        key_pressed["key"] = event.key
        plt.close()

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    if key_pressed["key"] == "enter":
        dst = os.path.join(rgbFinalPath, filename)
        shutil.copy(path, dst)
        print(f"Copied: {filename}")
    else:
        print(f"Skipped: {filename}")

print("Finished.")