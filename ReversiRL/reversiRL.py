"""
AlphaZero-style Reversi RL Agent
- Self-play training with MCTS + ResNet (policy + value heads)
- GameCenter client mode for evaluation
"""

import argparse
import collections
import json
import math
import os
import random
import socket
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Configuration ---

@dataclass
class Config:
	# Network
	num_res_blocks: int = 5
	num_filters: int = 64
	# MCTS
	num_simulations: int = 200
	c_puct: float = 1.5
	dirichlet_alpha: float = 0.3
	dirichlet_epsilon: float = 0.25
	temperature_threshold: int = 15
	# Training
	num_iterations: int = 100
	num_episodes: int = 50
	num_epochs: int = 10
	batch_size: int = 64
	learning_rate: float = 0.001
	weight_decay: float = 1e-4
	replay_buffer_size: int = 50000
	# GameCenter
	server_host: str = "127.0.0.1"
	server_port: int = 8888
	model_path: str = "reversi_model.pt"


# --- Constants ---

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

# Auto-detect device (verify CUDA actually works before using it)
def _detect_device():
	if torch.cuda.is_available():
		try:
			# Test that CUDA kernels actually run on this GPU
			t = torch.zeros(1, device="cuda")
			_ = t + 1
			return torch.device("cuda")
		except RuntimeError:
			pass
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")

DEVICE = _detect_device()


# --- Game Engine ---

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


# --- Neural Network ---

class ResBlock(nn.Module):
	"""Residual block with two conv layers and skip connection."""

	def __init__(self, filters):
		super().__init__()
		self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(filters)
		self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(filters)

	def forward(self, x):
		residual = x
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += residual
		return F.relu(out)


class ReversiNet(nn.Module):
	"""ResNet with policy and value heads for Reversi."""

	def __init__(self, config):
		super().__init__()
		nf = config.num_filters

		# Stem
		self.stem = nn.Sequential(
			nn.Conv2d(4, nf, 3, padding=1, bias=False),
			nn.BatchNorm2d(nf),
			nn.ReLU(),
		)

		# Residual tower
		self.res_tower = nn.Sequential(
			*[ResBlock(nf) for _ in range(config.num_res_blocks)]
		)

		# Policy head
		self.policy_conv = nn.Sequential(
			nn.Conv2d(nf, 2, 1, bias=False),
			nn.BatchNorm2d(2),
			nn.ReLU(),
		)
		self.policy_fc = nn.Linear(2 * NUM_CELLS, NUM_CELLS)

		# Value head
		self.value_conv = nn.Sequential(
			nn.Conv2d(nf, 1, 1, bias=False),
			nn.BatchNorm2d(1),
			nn.ReLU(),
		)
		self.value_fc1 = nn.Linear(NUM_CELLS, NUM_CELLS)
		self.value_fc2 = nn.Linear(NUM_CELLS, 1)

	def forward(self, x):
		# Shared trunk
		out = self.stem(x)
		out = self.res_tower(out)

		# Policy
		p = self.policy_conv(out)
		p = p.view(p.size(0), -1)
		p = F.log_softmax(self.policy_fc(p), dim=1)

		# Value
		v = self.value_conv(out)
		v = v.view(v.size(0), -1)
		v = F.relu(self.value_fc1(v))
		v = torch.tanh(self.value_fc2(v))

		return p, v

	@torch.no_grad()
	def predict(self, board, player):
		"""Single board inference. Returns (policy[64], value)."""
		state = torch.from_numpy(board.encode(player)).unsqueeze(0).to(DEVICE)
		log_policy, value = self(state)
		policy = torch.exp(log_policy).squeeze(0).cpu().numpy()

		# Mask invalid moves and renormalize
		mask = np.zeros(NUM_CELLS, dtype=np.float32)
		for m in board.get_valid_moves():
			mask[m] = 1.0
		policy *= mask
		total = policy.sum()
		if total > 0:
			policy /= total
		else:
			# Uniform over valid moves
			n = mask.sum()
			if n > 0:
				policy = mask / n
		return policy, value.item()


# --- MCTS ---

class MCTSNode:
	"""A node in the Monte Carlo search tree."""

	__slots__ = [
		'parent', 'children', 'visit_count', 'value_sum',
		'prior', 'board', 'player', 'is_expanded', 'move'
	]

	def __init__(self, board, player, prior=0.0, parent=None, move=-1):
		self.parent = parent
		self.children = {}
		self.visit_count = 0
		self.value_sum = 0.0
		self.prior = prior
		self.board = board
		self.player = player
		self.is_expanded = False
		self.move = move

	def q_value(self):
		if self.visit_count == 0:
			return 0.0
		return self.value_sum / self.visit_count

	def ucb_score(self, c_puct):
		parent_n = self.parent.visit_count if self.parent else 1
		exploration = c_puct * self.prior * math.sqrt(parent_n) / (1 + self.visit_count)
		return self.q_value() + exploration


class MCTS:
	"""Monte Carlo Tree Search with neural network evaluation."""

	def __init__(self, net, config):
		self.net = net
		self.config = config

	def search(self, board, player):
		"""Run MCTS simulations and return visit-count policy (64,)."""
		root = MCTSNode(board.copy(), player)

		# Expand root
		self._expand(root)

		# Add Dirichlet noise to root priors
		valid_moves = board.get_valid_moves()
		if valid_moves:
			noise = np.random.dirichlet(
				[self.config.dirichlet_alpha] * len(valid_moves)
			)
			eps = self.config.dirichlet_epsilon
			for i, move in enumerate(valid_moves):
				if move in root.children:
					child = root.children[move]
					child.prior = (1 - eps) * child.prior + eps * noise[i]

		# Run simulations
		for _ in range(self.config.num_simulations):
			node = self._select(root)
			value = self._expand(node)
			self._backup(node, value)

		# Build policy from visit counts
		policy = np.zeros(NUM_CELLS, dtype=np.float32)
		for move, child in root.children.items():
			policy[move] = child.visit_count
		total = policy.sum()
		if total > 0:
			policy /= total
		return policy

	def _select(self, node):
		"""Traverse tree selecting highest UCB child until leaf."""
		while node.is_expanded and node.children:
			best_score = -float('inf')
			best_child = None
			for child in node.children.values():
				score = child.ucb_score(self.config.c_puct)
				if score > best_score:
					best_score = score
					best_child = child
			if best_child is None:
				break
			node = best_child
		return node

	def _expand(self, node):
		"""Expand node using neural network. Returns value estimate."""
		if node.is_expanded:
			return node.q_value()

		board = node.board
		valid_moves = board.get_valid_moves()

		# Terminal node
		if not valid_moves and board.is_game_over():
			node.is_expanded = True
			winner = board.get_winner()
			if winner == 0:
				return 0.0
			return 1.0 if winner == node.player else -1.0

		# Pass: no valid moves but game not over
		if not valid_moves:
			node.is_expanded = True
			passed = board.copy()
			passed.turn = 3 - passed.turn
			passed._calc_valid_moves()
			child = MCTSNode(passed, 3 - node.player, prior=1.0, parent=node, move=-1)
			node.children[-1] = child
			# Evaluate from opponent's perspective, negate
			policy, value = self.net.predict(passed, 3 - node.player)
			return -value

		# Neural network evaluation
		policy, value = self.net.predict(board, node.player)

		node.is_expanded = True
		for move in valid_moves:
			child_board = board.copy()
			game_continues = child_board.place(move)
			child = MCTSNode(
				child_board, child_board.turn,
				prior=policy[move], parent=node, move=move
			)
			node.children[move] = child

		return value

	def _backup(self, node, value):
		"""Propagate value up the tree. Each child's q accumulates from the
		parent's (move-maker's) perspective, so MAX-UCB selects a parent's
		best move. We flip before adding: the leaf's NN value is from the
		leaf's own perspective, so negating once before storing at the leaf
		shifts it to the leaf's parent's perspective, and further flips up
		the chain alternate correctly."""
		while node is not None:
			value = -value
			node.visit_count += 1
			node.value_sum += value
			node = node.parent


# --- Self-Play ---

class SelfPlay:
	"""Generate training data through self-play games."""

	def __init__(self, net, config):
		self.net = net
		self.config = config
		self.mcts = MCTS(net, config)

	def play_game(self):
		"""Play one game via MCTS, return training examples."""
		board = ReversiBoard()
		history = []
		move_count = 0

		while True:
			valid_moves = board.get_valid_moves()
			if not valid_moves:
				if board.is_game_over():
					break
				# Pass
				board.turn = 3 - board.turn
				board._calc_valid_moves()
				continue

			# MCTS search
			policy = self.mcts.search(board, board.turn)

			# Record training example
			state = board.encode(board.turn)
			history.append((state, policy, board.turn))

			# Select move
			if move_count < self.config.temperature_threshold:
				# Sample proportional to policy (temperature = 1)
				move = np.random.choice(NUM_CELLS, p=policy)
			else:
				# Greedy
				move = int(np.argmax(policy))

			# Apply move
			game_continues = board.place(move)
			move_count += 1
			if not game_continues:
				break

		# Assign outcomes
		winner = board.get_winner()
		examples = []
		for state, policy, player in history:
			if winner == 0:
				value = 0.0
			elif winner == player:
				value = 1.0
			else:
				value = -1.0
			examples.append((state, policy, value))

		return examples

	def generate_data(self, num_episodes=None):
		"""Generate training data from multiple self-play games."""
		if num_episodes is None:
			num_episodes = self.config.num_episodes
		all_examples = []
		for ep in range(num_episodes):
			examples = self.play_game()
			all_examples.extend(examples)
			if (ep + 1) % 10 == 0:
				print(f"  Self-play: {ep + 1}/{num_episodes} games, "
					f"{len(all_examples)} examples")
		return all_examples


# --- Trainer ---

class Trainer:
	"""Training loop: self-play + network updates."""

	def __init__(self, config):
		self.config = config
		self.net = ReversiNet(config).to(DEVICE)
		self.optimizer = torch.optim.Adam(
			self.net.parameters(),
			lr=config.learning_rate,
			weight_decay=config.weight_decay,
		)
		self.replay_buffer = collections.deque(maxlen=config.replay_buffer_size)
		# Load existing model for continued training
		if os.path.exists(config.model_path):
			self.net.load_state_dict(
				torch.load(config.model_path, map_location=DEVICE, weights_only=True)
			)
			print(f"Resumed from existing model: {config.model_path}")

	def train(self):
		"""Main training loop."""
		print(f"Training on device: {DEVICE}")
		print(f"Config: {self.config.num_iterations} iterations, "
			f"{self.config.num_episodes} episodes/iter, "
			f"{self.config.num_simulations} MCTS sims")

		for iteration in range(1, self.config.num_iterations + 1):
			print(f"\n=== Iteration {iteration}/{self.config.num_iterations} ===")

			# Self-play
			self.net.eval()
			selfplay = SelfPlay(self.net, self.config)
			examples = selfplay.generate_data()
			self.replay_buffer.extend(examples)
			print(f"  Buffer size: {len(self.replay_buffer)}")

			# Train
			self.net.train()
			total_p_loss = 0.0
			total_v_loss = 0.0
			num_batches = 0
			for epoch in range(self.config.num_epochs):
				p_loss, v_loss = self._train_epoch()
				total_p_loss += p_loss
				total_v_loss += v_loss
				num_batches += 1

			avg_p = total_p_loss / max(num_batches, 1)
			avg_v = total_v_loss / max(num_batches, 1)
			print(f"  Loss: policy={avg_p:.4f}, value={avg_v:.4f}")

			# Save checkpoint
			torch.save(self.net.state_dict(), self.config.model_path)
			print(f"  Model saved: {self.config.model_path}")

	def _train_epoch(self):
		"""Train one epoch on replay buffer samples."""
		if len(self.replay_buffer) < self.config.batch_size:
			batch = list(self.replay_buffer)
		else:
			batch = random.sample(list(self.replay_buffer), self.config.batch_size)

		states = torch.from_numpy(
			np.array([s for s, _, _ in batch])
		).to(DEVICE)
		target_policies = torch.from_numpy(
			np.array([p for _, p, _ in batch])
		).to(DEVICE)
		target_values = torch.FloatTensor(
			[v for _, _, v in batch]
		).unsqueeze(1).to(DEVICE)

		# Forward pass
		log_policies, values = self.net(states)

		# Loss
		policy_loss = -torch.mean(
			torch.sum(target_policies * log_policies, dim=1)
		)
		value_loss = F.mse_loss(values, target_values)
		loss = policy_loss + value_loss

		# Backward pass
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return policy_loss.item(), value_loss.item()


# --- GameCenter Client ---

class GameCenterClient:
	"""Connect to GameCenter server and play using trained model + MCTS."""

	def __init__(self, config):
		self.config = config
		self.net = ReversiNet(config).to(DEVICE)
		if os.path.exists(config.model_path):
			self.net.load_state_dict(
				torch.load(config.model_path, map_location=DEVICE, weights_only=True)
			)
			print(f"Model loaded: {config.model_path}")
		else:
			print(f"Warning: no model at {config.model_path}, using random weights")
		self.net.eval()
		self.mcts = MCTS(self.net, config)

	def _send_json(self, sock, obj):
		"""Send length-prefixed JSON message."""
		data = json.dumps(obj).encode("utf-8")
		header = struct.pack("!I", len(data))
		sock.sendall(header + data)

	def _recv_json(self, sock):
		"""Receive length-prefixed JSON message."""
		header = self._recv_exact(sock, 4)
		if not header:
			return None
		length = struct.unpack("!I", header)[0]
		if length <= 0 or length > 65536:
			return None
		data = self._recv_exact(sock, length)
		if not data:
			return None
		return json.loads(data.decode("utf-8"))

	def _recv_exact(self, sock, n):
		"""Receive exactly n bytes."""
		buf = b""
		while len(buf) < n:
			chunk = sock.recv(n - len(buf))
			if not chunk:
				return None
			buf += chunk
		return buf

	def play_game(self):
		"""Play one game against the server. Returns winner (1/2) or 0 for tie."""
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
			sock.connect((self.config.server_host, self.config.server_port))

			# Receive setup
			msg = self._recv_json(sock)
			if not msg or msg.get("cmd") != "S":
				return None
			my_color = msg["player"]
			print(f"  Turn color: {'White' if my_color == 1 else 'Black'}")

			while True:
				msg = self._recv_json(sock)
				if not msg:
					return None

				cmd = msg.get("cmd")
				if cmd == "Q":
					w, b = msg["white"], msg["black"]
					print(f"  Game over - White: {w}, Black: {b}")
					if w > b:
						return 1
					elif b > w:
						return 2
					return 0

				if cmd != "T":
					return None

				# Parse bitboard state
				board = ReversiBoard.from_bitboards(
					msg["white"], msg["black"], msg["hint"], my_color
				)

				# Choose move via MCTS
				valid = board.get_valid_moves()
				if not valid:
					continue

				if len(valid) == 1:
					choice = valid[0]
				else:
					policy = self.mcts.search(board, my_color)
					choice = int(np.argmax(policy))

				# Send move
				self._send_json(sock, {"cmd": "P", "pos": choice})
				w_count = bin(board.white).count('1')
				b_count = bin(board.black).count('1')
				print(f"  Place: {choice} (W:{w_count} B:{b_count})")
		finally:
			sock.close()

	def run(self, num_games=0):
		"""Play games in a loop. 0 = infinite until Ctrl+C."""
		wins = [0, 0, 0]	# [losses, wins, ties]
		game_num = 0
		try:
			while num_games == 0 or game_num < num_games:
				game_num += 1
				print(f"Game {game_num}")
				result = self.play_game()
				if result is None:
					print("  Connection failed")
					break
				if result == 0:
					wins[2] += 1
				else:
					wins[1 if result == 1 else 0] += 1
		except KeyboardInterrupt:
			pass
		print(f"\nResults: Wins={wins[1]}, Losses={wins[0]}, Ties={wins[2]}")


# --- Game Tree AI (Minimax + Alpha-Beta) ---

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


# --- Test Harness ---

def test_games(config, opponent_type, num_games=100, gt_depth=5):
	"""Play num_games between AI (MCTS) and opponent, report win rate."""
	net = ReversiNet(config).to(DEVICE)
	if os.path.exists(config.model_path):
		net.load_state_dict(
			torch.load(config.model_path, map_location=DEVICE, weights_only=True)
		)
		print(f"Model loaded: {config.model_path}")
	else:
		print(f"Warning: no model at {config.model_path}, using random weights")
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


# --- Pygame Human Play ---

def play_human_pygame(config, human_color=2):
	"""Play against AI using pygame-ce GUI."""
	try:
		import pygame
	except ImportError:
		print("Error: pygame-ce required. Install: pip install pygame-ce")
		return

	CELL = 70
	BOARD_PX = CELL * 8
	PANEL_W = 220
	WIN_W = BOARD_PX + PANEL_W
	WIN_H = BOARD_PX

	GREEN = (0, 120, 0)
	BLACK = (0, 0, 0)
	WHITE = (255, 255, 255)
	PANEL_BG = (32, 32, 32)
	GRAY = (180, 180, 180)
	GOLD = (255, 215, 0)
	RED = (220, 50, 50)

	pygame.init()
	screen = pygame.display.set_mode((WIN_W, WIN_H))
	pygame.display.set_caption("DeepReversi")
	font_lg = pygame.font.SysFont(None, 36)
	font_md = pygame.font.SysFont(None, 28)
	font_sm = pygame.font.SysFont(None, 22)
	clock = pygame.time.Clock()

	net = ReversiNet(config).to(DEVICE)
	if os.path.exists(config.model_path):
		net.load_state_dict(
			torch.load(config.model_path, map_location=DEVICE, weights_only=True)
		)
		print(f"Model loaded: {config.model_path}")
	else:
		print("Warning: no model found, using random weights")
	net.eval()
	mcts = MCTS(net, config)

	ai_color = 3 - human_color
	board = ReversiBoard()
	game_over = False
	last_move = -1

	def draw(status=""):
		screen.fill(GREEN, (0, 0, BOARD_PX, BOARD_PX))
		for i in range(9):
			pygame.draw.line(screen, BLACK,
				(i * CELL, 0), (i * CELL, BOARD_PX), 2)
			pygame.draw.line(screen, BLACK,
				(0, i * CELL), (BOARD_PX, i * CELL), 2)

		for r in range(8):
			for c in range(8):
				pos = r * 8 + c
				cx = c * CELL + CELL // 2
				cy = r * CELL + CELL // 2
				if board.white & (1 << pos):
					pygame.draw.circle(screen, WHITE, (cx, cy), CELL // 2 - 5)
					pygame.draw.circle(screen, BLACK, (cx, cy), CELL // 2 - 5, 2)
					if pos == last_move:
						pygame.draw.circle(screen, RED, (cx, cy), 6)
				elif board.black & (1 << pos):
					pygame.draw.circle(screen, (20, 20, 20),
						(cx, cy), CELL // 2 - 5)
					pygame.draw.circle(screen, (80, 80, 80),
						(cx, cy), CELL // 2 - 5, 2)
					if pos == last_move:
						pygame.draw.circle(screen, RED, (cx, cy), 6)
				elif (not game_over and board.turn == human_color
						and pos in board.valid_moves):
					s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
					pygame.draw.circle(s, (255, 255, 255, 50),
						(CELL // 2, CELL // 2), 12)
					screen.blit(s, (c * CELL, r * CELL))

		# Side panel
		screen.fill(PANEL_BG, (BOARD_PX, 0, PANEL_W, WIN_H))
		x0 = BOARD_PX + 15
		y = 20
		screen.blit(font_lg.render("DeepReversi", True, GOLD), (x0, y))
		y += 50

		w_count, b_count = board.get_score()

		pygame.draw.circle(screen, WHITE, (x0 + 14, y + 14), 14)
		pygame.draw.circle(screen, BLACK, (x0 + 14, y + 14), 14, 2)
		label = "You" if human_color == 1 else "AI"
		screen.blit(font_md.render(f"  {label}: {w_count}", True, GRAY),
			(x0 + 32, y + 2))
		y += 42

		pygame.draw.circle(screen, (20, 20, 20), (x0 + 14, y + 14), 14)
		label = "You" if human_color == 2 else "AI"
		screen.blit(font_md.render(f"  {label}: {b_count}", True, GRAY),
			(x0 + 32, y + 2))
		y += 60

		if game_over:
			if w_count == b_count:
				result = "Draw!"
			elif ((w_count > b_count) == (human_color == 1)):
				result = "You Win!"
			else:
				result = "AI Wins!"
			screen.blit(font_lg.render(result, True, GOLD), (x0, y))
			y += 36
			screen.blit(font_sm.render(
				f"Score: {w_count} - {b_count}", True, GRAY), (x0, y))
			y += 28
			screen.blit(font_sm.render(
				"Click: new game", True, GRAY), (x0, y))
			y += 22
			screen.blit(font_sm.render("ESC: quit", True, GRAY), (x0, y))
		elif status:
			screen.blit(font_md.render(
				status, True, (255, 200, 100)), (x0, y))
		else:
			if board.turn == human_color:
				screen.blit(font_md.render(
					"Your turn", True, (100, 255, 100)), (x0, y))
			else:
				screen.blit(font_md.render(
					"AI thinking...", True, (255, 200, 100)), (x0, y))
		pygame.display.flip()

	running = True
	while running:
		if (not game_over and board.turn == ai_color
				and board.get_valid_moves()):
			draw("AI thinking...")
			policy = mcts.search(board, board.turn)
			move = int(np.argmax(policy))
			if not board.place(move):
				game_over = True
			last_move = move
			draw()

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_n:
					board = ReversiBoard()
					game_over = False
					last_move = -1
			elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				mx, my = event.pos
				if game_over:
					board = ReversiBoard()
					game_over = False
					last_move = -1
				elif mx < BOARD_PX and board.turn == human_color:
					col = mx // CELL
					row = my // CELL
					if 0 <= row < 8 and 0 <= col < 8:
						pos = row * 8 + col
						if pos in board.get_valid_moves():
							if not board.place(pos):
								game_over = True
							last_move = pos
		draw()
		clock.tick(30)

	pygame.quit()


# --- Entry Point ---

def main():
	parser = argparse.ArgumentParser(
		description="AlphaZero-style Reversi RL Agent")
	parser.add_argument("mode",
		choices=["train", "play", "play-human",
			"test-gt", "test-random", "test-scoreboard"],
		help="train | play | play-human | test-gt | "
			"test-random | test-scoreboard")
	parser.add_argument("--model", default="reversi_model.pt",
		help="Model file path (default: reversi_model.pt)")
	parser.add_argument("--host", default="127.0.0.1",
		help="GameCenter host (default: 127.0.0.1)")
	parser.add_argument("--port", type=int, default=8888,
		help="GameCenter port (default: 8888)")
	parser.add_argument("--simulations", type=int, default=200,
		help="MCTS simulations per move (default: 200)")
	parser.add_argument("--iterations", type=int, default=100,
		help="Training iterations (default: 100)")
	parser.add_argument("--episodes", type=int, default=50,
		help="Self-play games per iteration (default: 50)")
	parser.add_argument("--games", type=int, default=0,
		help="Number of games (0=infinite for play, 100 for test)")
	parser.add_argument("--gt-depth", type=int, default=5,
		help="Game tree search depth (default: 5)")
	parser.add_argument("--color", type=int, default=2, choices=[1, 2],
		help="Human color: 1=white(first), 2=black (default: 2)")
	args = parser.parse_args()

	config = Config(
		model_path=args.model,
		server_host=args.host,
		server_port=args.port,
		num_simulations=args.simulations,
		num_iterations=args.iterations,
		num_episodes=args.episodes,
	)

	if args.mode == "train":
		Trainer(config).train()
	elif args.mode == "play":
		GameCenterClient(config).run(args.games)
	elif args.mode == "play-human":
		play_human_pygame(config, args.color)
	elif args.mode == "test-gt":
		n = args.games if args.games > 0 else 100
		test_games(config, "gt", n, args.gt_depth)
	elif args.mode == "test-random":
		n = args.games if args.games > 0 else 100
		test_games(config, "random", n)
	elif args.mode == "test-scoreboard":
		n = args.games if args.games > 0 else 100
		test_games(config, "scoreboard", n)


if __name__ == "__main__":
	main()
