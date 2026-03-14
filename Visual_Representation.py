import chessboard_image as cbi
from PIL import Image
import cv2

def visualize_fen(fen,fileName):
    # Generate a PIL.Image object
    img = cbi.generate_image(fen, fileName, size=400)  # Pass None as filename

    # img = Image.open(img)  # Open the generated image using PIL
    Image.open(img).show()  # Display the image
