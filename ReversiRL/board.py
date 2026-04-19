"""Reversi game engine and shared board constants."""

import numpy as np


BOARD_SIZE = 8
NUM_CELLS = 64

# Direction offsets: N, NE, E, SE, S, SW, W, NW
DXY = [-8, -7, 1, 9, 8, 7, -1, -9]
DR = [-1, -1, 0, 1, 1, 1, 0, -1]
DC = [0, 1, 1, 1, 0, -1, -1, -1]

# Precompute limit table (same as C++ limittbl.h)
LIMIT = [[0] * 8 for _ in range(NUM_CELLS)]
for _p in range(NUM_CELLS):
	_r, _c = _p // 8, _p % 8
	for _d in range(8):
		_steps = 0
		_nr, _nc = _r + DR[_d], _c + DC[_d]
		while 0 <= _nr < 8 and 0 <= _nc < 8:
			_steps += 1
			_nr += DR[_d]
			_nc += DC[_d]
		LIMIT[_p][_d] = _steps


class ReversiBoard:
	"""Reversi game engine using bitboard representation."""

	def __init__(self):
		self.white = 0
		self.black = 0
		self.turn = 1	# 1=white, 2=black
		self.valid_moves = []
		self._init_start()

	def _init_start(self):
		"""Set up the standard starting position."""
		self.white = (1 << (3 * 8 + 3)) | (1 << (4 * 8 + 4))
		self.black = (1 << (3 * 8 + 4)) | (1 << (4 * 8 + 3))
		self.turn = 1
		self._calc_valid_moves()

	def copy(self):
		b = ReversiBoard.__new__(ReversiBoard)
		b.white = self.white
		b.black = self.black
		b.turn = self.turn
		b.valid_moves = list(self.valid_moves)
		return b

	@classmethod
	def from_bitboards(cls, white, black, hint, turn):
		"""Create board from server bitboard state."""
		b = cls.__new__(cls)
		b.white = white
		b.black = black
		b.turn = turn
		b.valid_moves = []
		for i in range(NUM_CELLS):
			if hint & (1 << i):
				b.valid_moves.append(i)
		return b

	def _my_board(self):
		return self.white if self.turn == 1 else self.black

	def _opp_board(self):
		return self.black if self.turn == 1 else self.white

	def _set_boards(self, my_board, opp_board):
		if self.turn == 1:
			self.white, self.black = my_board, opp_board
		else:
			self.black, self.white = my_board, opp_board

	def _calc_valid_moves(self):
		"""Calculate valid moves for current player."""
		self.valid_moves = []
		my = self._my_board()
		opp = self._opp_board()
		occupied = my | opp

		for p in range(NUM_CELLS):
			if occupied & (1 << p):
				continue
			# Check if placing here flips at least one opponent piece
			for d in range(8):
				lim = LIMIT[p][d]
				if lim < 2:
					continue
				r = p + DXY[d]
				k = 1
				while k < lim and (opp & (1 << r)):
					r += DXY[d]
					k += 1
				if k > 1 and (my & (1 << r)):
					self.valid_moves.append(p)
					break

	def get_valid_moves(self):
		return self.valid_moves

	def place(self, pos):
		"""Place a piece at pos. Returns True if game continues, False if over."""
		if pos not in self.valid_moves:
			return False

		my = self._my_board()
		opp = self._opp_board()
		my |= (1 << pos)

		# Flip opponent pieces in each direction
		for d in range(8):
			lim = LIMIT[pos][d]
			if lim == 0:
				continue
			r = pos + DXY[d]
			k = 1
			while k < lim and (opp & (1 << r)):
				r += DXY[d]
				k += 1
			if my & (1 << r):
				# Flip back
				while k > 1:
					k -= 1
					r -= DXY[d]
					opp &= ~(1 << r)
					my |= (1 << r)

		self._set_boards(my, opp)

		# Try switching to opponent
		for t in range(2):
			self.turn = 3 - self.turn
			self._calc_valid_moves()
			if self.valid_moves:
				return True
			# Check if board is full
			empty = NUM_CELLS - bin(self.white | self.black).count('1')
			if empty == 0:
				return False

		return False

	def is_game_over(self):
		if not self.valid_moves:
			# Also check opponent
			saved_turn = self.turn
			self.turn = 3 - self.turn
			self._calc_valid_moves()
			has_moves = len(self.valid_moves) > 0
			self.turn = saved_turn
			self._calc_valid_moves()
			return not has_moves
		return False

	def get_score(self):
		"""Returns (white_count, black_count)."""
		return bin(self.white).count('1'), bin(self.black).count('1')

	def get_winner(self):
		"""Returns 1 if white wins, 2 if black wins, 0 if tie."""
		w, b = self.get_score()
		if w > b:
			return 1
		elif b > w:
			return 2
		return 0

	def encode(self, player):
		"""Encode board as neural network input (4, 8, 8) float32 array.
		Planes: [my pieces, opponent pieces, valid moves, player indicator]
		"""
		state = np.zeros((4, 8, 8), dtype=np.float32)
		my_bb = self.white if player == 1 else self.black
		opp_bb = self.black if player == 1 else self.white

		for i in range(NUM_CELLS):
			r, c = i // 8, i % 8
			if my_bb & (1 << i):
				state[0, r, c] = 1.0
			if opp_bb & (1 << i):
				state[1, r, c] = 1.0

		for pos in self.valid_moves:
			state[2, pos // 8, pos % 8] = 1.0

		if player == 1:
			state[3, :, :] = 1.0

		return state
