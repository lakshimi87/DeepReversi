"""Monte Carlo Tree Search with batched leaf evaluation."""

import math

import numpy as np
import torch

from board import NUM_CELLS
from config import DEVICE


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


# Virtual loss applied along a path while another simulation in the
# same gather batch is still pending. Discourages the next select from
# reusing the same path and keeps the batch diverse.
VIRTUAL_LOSS = 3


class MCTS:
	"""Monte Carlo Tree Search with batched neural network evaluation.

	Each outer iteration gathers up to `mcts_batch_size` leaves — during
	the gather, virtual loss is applied along every selected path so
	subsequent selects prefer different branches. Terminal leaves are
	resolved in place; non-terminal leaves are evaluated in one batched
	NN forward pass, then expanded and backed up (which also reverts
	the virtual loss)."""

	def __init__(self, net, config):
		self.net = net
		self.config = config

	def search(self, board, player):
		"""Run MCTS simulations and return visit-count policy (64,)."""
		root = MCTSNode(board.copy(), player)
		self._expand_root(root)

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

		remaining = self.config.num_simulations
		bs = max(1, self.config.mcts_batch_size)

		while remaining > 0:
			target = min(bs, remaining)
			pending = []  # list of (leaf, path, is_pass)

			for _ in range(target):
				path, leaf = self._select_leaf(root)

				# Same unexpanded leaf selected twice in this gather —
				# flush the batch now rather than evaluate it twice.
				if any(item[0] is leaf for item in pending):
					break

				# Apply virtual loss on the descent path
				for n in path:
					n.visit_count += VIRTUAL_LOSS
					n.value_sum -= VIRTUAL_LOSS

				valid = leaf.board.get_valid_moves()
				if not valid:
					if leaf.board.is_game_over():
						winner = leaf.board.get_winner()
						if winner == 0:
							value = 0.0
						else:
							value = 1.0 if winner == leaf.player else -1.0
						leaf.is_expanded = True
						self._revert_and_backup(path, value)
						remaining -= 1
						continue
					# Pass — NN evaluates the pass-successor board
					pending.append((leaf, path, True))
				else:
					pending.append((leaf, path, False))

			if pending:
				self._batch_expand_and_backup(pending)
				remaining -= len(pending)

		# Build policy from visit counts (skip the pass pseudo-move)
		policy = np.zeros(NUM_CELLS, dtype=np.float32)
		for move, child in root.children.items():
			if move >= 0:
				policy[move] = child.visit_count
		total = policy.sum()
		if total > 0:
			policy /= total
		return policy

	def _select_leaf(self, root):
		"""Descend by max UCB until we reach an unexpanded (or childless)
		node. Returns (path, leaf); path[0] is root, path[-1] is leaf."""
		node = root
		path = [node]
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
			path.append(node)
		return path, node

	def _expand_root(self, root):
		"""Synchronous NN evaluation for the root so Dirichlet noise has
		priors to perturb before the main batched loop begins."""
		board = root.board
		valid = board.get_valid_moves()
		if not valid:
			root.is_expanded = True
			return
		policy, _ = self.net.predict(board, root.player)
		root.is_expanded = True
		for m in valid:
			cb = board.copy()
			cb.place(m)
			root.children[m] = MCTSNode(
				cb, cb.turn, prior=float(policy[m]), parent=root, move=m
			)

	def _batch_expand_and_backup(self, pending):
		"""Single NN forward on all pending leaves, then expand + backup."""
		inputs = []
		# Per-item context retained in the same order as `inputs`.
		ctx = []
		for leaf, path, is_pass in pending:
			if is_pass:
				# Eval the pass-successor board (opponent-to-move).
				succ = leaf.board.copy()
				succ.turn = 3 - succ.turn
				succ._calc_valid_moves()
				eval_board, eval_player = succ, 3 - leaf.player
			else:
				eval_board, eval_player = leaf.board, leaf.player
			inputs.append(eval_board.encode(eval_player))
			ctx.append((leaf, path, is_pass, eval_board, eval_player))

		tensor = torch.from_numpy(np.stack(inputs)).to(DEVICE)
		with torch.no_grad():
			log_policies, values = self.net(tensor)
		policies = torch.exp(log_policies).cpu().numpy()
		values = values.squeeze(-1).cpu().numpy()

		for (leaf, path, is_pass, eval_board, eval_player), raw_policy, value in zip(
				ctx, policies, values):
			if is_pass:
				# Single pass child; value is from successor's perspective,
				# so leaf itself should store +value (its parent's view).
				# _revert_and_backup flips once before storing at leaf, so
				# pass -value in to end up with +value there.
				leaf.children[-1] = MCTSNode(
					eval_board, eval_player, prior=1.0, parent=leaf, move=-1
				)
				leaf.is_expanded = True
				self._revert_and_backup(path, -float(value))
			else:
				valid = eval_board.get_valid_moves()
				mask = np.zeros(NUM_CELLS, dtype=np.float32)
				for m in valid:
					mask[m] = 1.0
				masked = raw_policy * mask
				tot = masked.sum()
				if tot > 0:
					masked = masked / tot
				else:
					n = mask.sum()
					masked = mask / n if n > 0 else mask

				for m in valid:
					cb = eval_board.copy()
					cb.place(m)
					leaf.children[m] = MCTSNode(
						cb, cb.turn, prior=float(masked[m]),
						parent=leaf, move=m
					)
				leaf.is_expanded = True
				self._revert_and_backup(path, float(value))

	def _revert_and_backup(self, path, value):
		"""Revert virtual loss on every node in `path` and apply the real
		(visit +1, value_sum += signed-value) update. Sign alternates per
		ply, same convention as the original sequential backup."""
		v = float(value)
		for node in reversed(path):
			v = -v
			node.visit_count = node.visit_count - VIRTUAL_LOSS + 1
			node.value_sum = node.value_sum + VIRTUAL_LOSS + v
