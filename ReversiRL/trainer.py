"""Self-play data generation and training loop."""

import collections
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from board import NUM_CELLS, ReversiBoard
from checkpoint import load_model_weights
from config import DEVICE
from mcts import MCTS
from network import ReversiNet


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
		self.start_iteration = 0
		self._resume()

	def _resume(self):
		"""Resume from checkpoints/latest.pt if available (full state),
		otherwise seed weights from the legacy flat model file."""
		latest = os.path.join(self.config.checkpoint_dir, "latest.pt")
		if os.path.exists(latest):
			ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
			if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
				self.net.load_state_dict(ckpt["model_state_dict"])
				opt_state = ckpt.get("optimizer_state_dict")
				if opt_state is not None:
					self.optimizer.load_state_dict(opt_state)
				buf = ckpt.get("replay_buffer") or []
				if buf:
					self.replay_buffer.extend(buf)
				self.start_iteration = int(ckpt.get("iteration", 0))
				print(f"Resumed checkpoint {latest} "
					f"(iteration {self.start_iteration}, "
					f"buffer {len(self.replay_buffer)})")
				return
			# Raw state_dict saved as latest.pt — treat as weights-only seed
			self.net.load_state_dict(ckpt)
			print(f"Seeded weights from {latest}")
			return
		if os.path.exists(self.config.model_path):
			load_model_weights(self.net, self.config.model_path)
			print(f"Seeded weights from {self.config.model_path}")

	def _save_checkpoint(self, iteration):
		"""Write iter_XXXX.pt and latest.pt with full training state."""
		os.makedirs(self.config.checkpoint_dir, exist_ok=True)
		checkpoint = {
			"iteration": iteration,
			"model_state_dict": self.net.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"replay_buffer": list(self.replay_buffer),
		}
		iter_path = os.path.join(
			self.config.checkpoint_dir, f"iter_{iteration:04d}.pt"
		)
		latest_path = os.path.join(self.config.checkpoint_dir, "latest.pt")
		torch.save(checkpoint, iter_path)
		torch.save(checkpoint, latest_path)
		print(f"  Checkpoint saved: {iter_path} (latest.pt updated)")

	def train(self):
		"""Main training loop."""
		print(f"Training on device: {DEVICE}")
		start = self.start_iteration + 1
		end = self.start_iteration + self.config.num_iterations
		print(f"Config: iterations {start}..{end}, "
			f"{self.config.num_episodes} episodes/iter, "
			f"{self.config.num_simulations} MCTS sims, "
			f"checkpoint every {self.config.checkpoint_interval} iters")

		for iteration in range(start, end + 1):
			print(f"\n=== Iteration {iteration} ===")

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

			# Save checkpoint every N iterations (and on the final iter)
			if (iteration % self.config.checkpoint_interval == 0
					or iteration == end):
				self._save_checkpoint(iteration)

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
