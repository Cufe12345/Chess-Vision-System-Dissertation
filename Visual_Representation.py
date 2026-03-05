import chessboard_image as cbi
from PIL import Image

def visualize_fen(fen):
    # Generate a PIL.Image object
    img = cbi.generate_image(fen, None, size=400)  # Pass None as filename

    img = Image.open(img)  # Open the generated image using PIL
    # Show it
    img.show()