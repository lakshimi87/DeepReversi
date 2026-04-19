"""GameCenter TCP client: connects, plays via MCTS-backed policy."""

import json
import os
import socket
import struct

import numpy as np

from board import ReversiBoard
from checkpoint import load_model_weights, resolve_model_path
from config import DEVICE
from mcts import MCTS
from network import ReversiNet


class GameCenterClient:
	"""Connect to GameCenter server and play using trained model + MCTS."""

	def __init__(self, config):
		self.config = config
		self.net = ReversiNet(config).to(DEVICE)
		path = resolve_model_path(config)
		if os.path.exists(path):
			load_model_weights(self.net, path)
			print(f"Model loaded: {path}")
		else:
			print(f"Warning: no model at {path}, using random weights")
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
