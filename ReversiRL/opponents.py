"""Baseline opponents used for evaluation: positional scoreboard and
minimax game-tree search with alpha-beta pruning."""

import random

from board import NUM_CELLS


SCORE_BOARD = [
	10,  1,  3,  2,  2,  3,  1, 10,
	 1, -5, -1, -1, -1, -1, -5,  1,
	 3, -1,  0,  0,  0,  0, -1,  3,
	 2, -1,  0,  0,  0,  0, -1,  2,
	 2, -1,  0,  0,  0,  0, -1,  2,
	 3, -1,  0,  0,  0,  0, -1,  3,
	 1, -5, -1, -1, -1, -1, -5,  1,
	10,  1,  3,  2,  2,  3,  1, 10,
]


def _gt_evaluate(board):
	"""Evaluate board from white's perspective using positional weights."""
	score = 0
	for i in range(NUM_CELLS):
		if board.white & (1 << i):
			score += SCORE_BOARD[i]
		elif board.black & (1 << i):
			score -= SCORE_BOARD[i]
	return score


def _gt_minimax(board, depth, alpha, beta):
	"""Minimax with alpha-beta pruning. Returns (score, move)."""
	valid = board.get_valid_moves()
	if not valid:
		w, b = board.get_score()
		if w > b:
			return 10000, -1
		elif b > w:
			return -10000, -1
		return 0, -1
	if depth == 0:
		return _gt_evaluate(board), -1

	maximizing = (board.turn == 1)
	best_move = valid[0]

	if maximizing:
		best = -100000
		for move in valid:
			child = board.copy()
			child.place(move)
			score, _ = _gt_minimax(child, depth - 1, alpha, beta)
			if score > best:
				best = score
				best_move = move
			alpha = max(alpha, score)
			if alpha >= beta:
				break
		return best, best_move
	else:
		best = 100000
		for move in valid:
			child = board.copy()
			child.place(move)
			score, _ = _gt_minimax(child, depth - 1, alpha, beta)
			if score < best:
				best = score
				best_move = move
			beta = min(beta, score)
			if alpha >= beta:
				break
		return best, best_move


def gt_get_move(board, depth=5):
	"""Get best move using game tree search."""
	valid = board.get_valid_moves()
	if len(valid) == 1:
		return valid[0]
	_, move = _gt_minimax(board, depth, -100000, 100000)
	return move


def scoreboard_get_move(board):
	"""Pick valid move with highest SCORE_BOARD weight (tie → random)."""
	valid = board.get_valid_moves()
	best = max(SCORE_BOARD[m] for m in valid)
	top = [m for m in valid if SCORE_BOARD[m] == best]
	return random.choice(top)
