"""Global configuration and device detection."""

from dataclasses import dataclass

import torch


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
	# Batched MCTS: gather up to this many leaves per NN forward pass.
	# Diversity during gather is enforced via virtual loss. Benchmarking
	# (CPU, 5 resblocks / 64 filters) found 8 to be the sweet spot; larger
	# batches see diminishing returns from gather collisions.
	mcts_batch_size: int = 8
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
	# Checkpoints
	checkpoint_dir: str = "checkpoints"
	checkpoint_interval: int = 10


# Auto-detect device. MPS is intentionally skipped: this network is
# small (few resblocks, 64 filters) and MCTS issues many single-board
# evaluations, where MPS kernel-launch + host↔device sync overhead
# makes it measurably slower than CPU (benchmarked ~2.5x slower).
def _detect_device():
	if torch.cuda.is_available():
		try:
			t = torch.zeros(1, device="cuda")
			_ = t + 1
			return torch.device("cuda")
		except RuntimeError:
			pass
	return torch.device("cpu")


DEVICE = _detect_device()
