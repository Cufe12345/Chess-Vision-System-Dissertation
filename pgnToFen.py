import chess
import chess.pgn
import sys

def pgn_to_fens(pgn_path, output_path, max_games=None):
    with open(pgn_path, 'r') as pgn_file, open(output_path, 'w') as output_file:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                output_file.write(board.fen() + '\n')
            
            game_count += 1
            if game_count % 1000 == 0:
                print(f"Processed {game_count} games...")
            if max_games is not None and game_count >= max_games:
                break


input_pgn_file = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\lichess_db_standard_rated_2025-11.pgn\\lichess_db_standard_rated_2025-11.pgn"
output_fen_file = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\OutputFENs.txt"
max_games = 100000

pgn_to_fens(input_pgn_file, output_fen_file, max_games)