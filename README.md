# DeepReversi

An AlphaZero-style deep reinforcement learning agent for Reversi (Othello), featuring self-play training with MCTS, a tournament server, and a classical game-tree AI baseline.

## Components

### ReversiRL (Python)

The core ML component — an AlphaZero-inspired agent built with PyTorch.

- **ResNet** with 5 residual blocks, dual policy + value heads
- **MCTS** with 200 simulations per move, Dirichlet noise for exploration
- **Self-play training** loop: generate games → train on replay buffer → iterate
- **GPU acceleration**: auto-detects CUDA, MPS, or CPU
- Modes: `train`, `play`, `play-human`, `test-gt`, `test-random`

### GameCenter (C++)

A Windows desktop application that hosts Reversi games over the network.

- GUI with an 8x8 board display
- TCP server on port 8888, supports 2 player connections
- JSON protocol with 4-byte length framing and 64-bit bitboard encoding
- Supports human vs AI, AI vs AI, and human vs human

### ReversiGT (C++)

A classical game-tree AI player for benchmarking.

- Minimax search with alpha-beta pruning (configurable depth)
- Positional evaluation with weighted scoring (corners, edges)
- Cross-platform socket support (Windows + Unix)
- Connects to GameCenter as a network client

## Setup

```bash
./setup.sh
```

Creates a Python virtual environment at `~/venvs/torch/` and installs dependencies (PyTorch, NumPy, Pygame-CE).

## Usage

**Train the model:**

```bash
./train.sh
```

Runs 100 iterations of self-play training (50 episodes each). Saves a full checkpoint (model + optimizer + replay buffer + iteration) every 10 iterations to `checkpoints/iter_XXXX.pt`, with `checkpoints/latest.pt` tracking the most recent. Training auto-resumes from `checkpoints/latest.pt` if present.

**Play against the AI (interactive GUI):**

```bash
./play.sh
```

Opens a Pygame window where you play as black against the trained model.

**Benchmark against random opponent:**

```bash
./test_random.sh
```

**Benchmark against game-tree AI:**

```bash
./test_gt.sh
```

**Network tournament mode:**

1. Start GameCenter (Windows GUI server on port 8888)
2. Connect clients:
   ```bash
   python ReversiRL/reversiRL.py play --host 127.0.0.1 --port 8888
   ```

## Network Protocol

Communication uses TCP with JSON messages, framed by a 4-byte big-endian length prefix.

| Command | Direction | Description |
|---------|-----------|-------------|
| `S` | Server → Client | Setup: assigns player color (`1`=white, `2`=black) |
| `T` | Server → Client | Turn: sends board state as 64-bit bitboards (`white`, `black`, `hint`) |
| `P` | Client → Server | Place: client's chosen move position (0-63) |
| `Q` | Server → Client | Quit: game over with final scores |

## License

GPL-3.0
