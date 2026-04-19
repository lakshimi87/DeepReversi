"""Headless evaluation harness: AI vs baseline opponents."""

import os
import random

import numpy as np

from board import ReversiBoard
from checkpoint import load_model_weights, resolve_model_path
from config import DEVICE
from mcts import MCTS
from network import ReversiNet
from opponents import gt_get_move, scoreboard_get_move


def test_games(config, opponent_type, num_games=100, gt_depth=5):
	"""Play num_games between AI (MCTS) and opponent, report win rate."""
	net = ReversiNet(config).to(DEVICE)
	path = resolve_model_path(config)
	if os.path.exists(path):
		load_model_weights(net, path)
		print(f"Model loaded: {path}")
	else:
		print(f"Warning: no model at {path}, using random weights")
	net.eval()
	mcts = MCTS(net, config)

	wins = draws = losses = 0
	print(f"Testing AI vs {opponent_type} ({num_games} games, "
		f"{config.num_simulations} MCTS sims)")
	if opponent_type == "gt":
		print(f"Game tree depth: {gt_depth}")
	print()

	for g in range(num_games):
		board = ReversiBoard()
		ai_color = 1 if g % 2 == 0 else 2

		while True:
			valid = board.get_valid_moves()
			if not valid:
				break
			if board.turn == ai_color:
				if len(valid) == 1:
					move = valid[0]
				else:
					policy = mcts.search(board, board.turn)
					move = int(np.argmax(policy))
			else:
				if opponent_type == "random":
					move = random.choice(valid)
				elif opponent_type == "scoreboard":
					move = scoreboard_get_move(board)
				else:
					move = gt_get_move(board, gt_depth)
			if not board.place(move):
				break

		winner = board.get_winner()
		if winner == ai_color:
			wins += 1
			result = "Win"
		elif winner == 0:
			draws += 1
			result = "Draw"
		else:
			losses += 1
			result = "Loss"

		w, b = board.get_score()
		ai_label = "W" if ai_color == 1 else "B"
		print(f"Game {g+1:3d}/{num_games}: {result:4s} "
			f"(AI={ai_label} W:{w:2d} B:{b:2d}) "
			f"[W:{wins} D:{draws} L:{losses}]")

	print(f"\n{'='*50}")
	print(f"AI vs {opponent_type}: {wins}W {draws}D {losses}L / {num_games} games")
	print(f"Win rate: {wins/num_games*100:.1f}%  "
		f"Draw rate: {draws/num_games*100:.1f}%  "
		f"Loss rate: {losses/num_games*100:.1f}%")
