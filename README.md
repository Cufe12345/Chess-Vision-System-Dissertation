# Chess Vision System for Identifying Chess Pieces from a Top-Down Perspective

This project contains the scripts for training, testing, and running a live prediction for a vision system that converts images of a chessboard into standard FEN notation. 

The vision system is divided into two approaches, and the results from each are combined to produce a final prediction.

The first approach is a pipeline model which breaks the problem down into board segmentation identifying the board from an image), square segmentation (splitting the board into 64 Squares) and piece recognition, which identifies the piece in an input square.

The second approach is an end-to-end model which aims to identify all pieces from the whole image without breaking it down.

The predictions from both models, with their given certainty for each square is passed into a logistic regression model to give a final output prediction.

