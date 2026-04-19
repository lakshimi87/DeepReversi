"""AlphaZero-style Reversi RL Agent — CLI entry point.

Functional modules:
  board.py       — ReversiBoard + shared board constants
  network.py     — ResBlock / ReversiNet (policy + value heads)
  mcts.py        — MCTS with batched leaf evaluation
  trainer.py     — SelfPlay + training loop
  client.py      — GameCenter TCP client
  opponents.py   — scoreboard and game-tree baselines
  harness.py     — headless AI-vs-opponent evaluation
  gui.py         — pygame human-vs-AI GUI (uses resource/board.png)
  checkpoint.py  — model / full-state checkpoint I/O
  config.py      — Config dataclass + device detection
"""

import argparse

from config import Config


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
	parser.add_argument("--mcts-batch", type=int, default=8,
		help="MCTS leaves batched per NN forward (default: 8)")
	parser.add_argument("--iterations", type=int, default=100,
		help="Training iterations (default: 100)")
	parser.add_argument("--episodes", type=int, default=50,
		help="Self-play games per iteration (default: 50)")
	parser.add_argument("--games", type=int, default=0,
		help="Number of games (0=infinite for play, 100 for test)")
	parser.add_argument("--gt-depth", type=int, default=5,
		help="Game tree search depth (default: 5)")
	parser.add_argument("--color", type=int, default=0, choices=[0, 1, 2],
		help="Human color: 0=random (default), 1=white(first), 2=black")
	args = parser.parse_args()

	config = Config(
		model_path=args.model,
		server_host=args.host,
		server_port=args.port,
		num_simulations=args.simulations,
		mcts_batch_size=args.mcts_batch,
		num_iterations=args.iterations,
		num_episodes=args.episodes,
	)

	if args.mode == "train":
		from trainer import Trainer
		Trainer(config).train()
	elif args.mode == "play":
		from client import GameCenterClient
		GameCenterClient(config).run(args.games)
	elif args.mode == "play-human":
		from gui import play_human_pygame
		play_human_pygame(config, args.color)
	elif args.mode == "test-gt":
		from harness import test_games
		n = args.games if args.games > 0 else 100
		test_games(config, "gt", n, args.gt_depth)
	elif args.mode == "test-random":
		from harness import test_games
		n = args.games if args.games > 0 else 100
		test_games(config, "random", n)
	elif args.mode == "test-scoreboard":
		from harness import test_games
		n = args.games if args.games > 0 else 100
		test_games(config, "scoreboard", n)


if __name__ == "__main__":
	main()
