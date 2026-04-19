"""Checkpoint / model-weight I/O helpers."""

import os

import torch

from config import DEVICE


def load_model_weights(net, path):
	"""Load model weights from `path`, accepting either a raw state_dict
	file or a full-checkpoint dict with a 'model_state_dict' key."""
	try:
		state = torch.load(path, map_location=DEVICE, weights_only=True)
	except Exception:
		# Full checkpoint dicts contain mixed Python objects (optimizer
		# state, replay buffer) which the safe loader rejects. Fall back
		# to the permissive loader — checkpoint files are written by us.
		state = torch.load(path, map_location=DEVICE, weights_only=False)
	if isinstance(state, dict) and "model_state_dict" in state:
		state = state["model_state_dict"]
	net.load_state_dict(state)


def resolve_model_path(config):
	"""Return the model file to load from, preferring a fresh checkpoint
	under `checkpoint_dir` over the legacy flat `model_path`."""
	latest = os.path.join(config.checkpoint_dir, "latest.pt")
	if os.path.exists(latest):
		return latest
	return config.model_path
