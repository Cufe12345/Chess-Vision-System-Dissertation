import chessboard_image as cbi
from PIL import Image

fen = "rnbqkbnr/ppppppp1/7p/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Generate a PIL.Image object
img = cbi.generate_image(fen, None, size=400)  # Pass None as filename

img = Image.open(img)  # Open the generated image using PIL
# Show it
img.show()