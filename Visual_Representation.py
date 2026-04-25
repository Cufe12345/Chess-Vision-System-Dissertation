import chessboard_image as cbi
from PIL import Image

def visualize_fen(fen,fileName):
    img = cbi.generate_image(fen, fileName, size=400)
    Image.open(img).show()  # Display the image
