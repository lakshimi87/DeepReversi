"""ResNet with policy + value heads for Reversi."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import NUM_CELLS
from config import DEVICE


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
