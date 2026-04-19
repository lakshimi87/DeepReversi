"""Pygame human-vs-AI GUI."""

import os
import random

import numpy as np

from board import ReversiBoard
from checkpoint import load_model_weights, resolve_model_path
from config import DEVICE
from mcts import MCTS
from network import ReversiNet


# resource/board.png sits one level up from this module.
RESOURCE_DIR = os.path.normpath(os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "..", "resource"))
BOARD_IMAGE_PATH = os.path.join(RESOURCE_DIR, "board.png")


def _load_board_background(pygame, size):
	"""Return a Surface the size of the board, textured with board.png
	tiled/scaled to fit. Falls back to a solid green surface if the
	image is missing or fails to load."""
	fallback = pygame.Surface((size, size))
	fallback.fill((0, 120, 0))
	if not os.path.exists(BOARD_IMAGE_PATH):
		print(f"Warning: {BOARD_IMAGE_PATH} not found, using solid green.")
		return fallback
	try:
		img = pygame.image.load(BOARD_IMAGE_PATH).convert()
	except Exception as exc:
		print(f"Warning: failed to load {BOARD_IMAGE_PATH}: {exc}")
		return fallback
	return pygame.transform.smoothscale(img, (size, size))


def play_human_pygame(config, color_mode=0):
	"""Play against AI using pygame-ce GUI.

	color_mode: 0=random (re-picked each new game), 1=white, 2=black.
	"""
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

	board_bg = _load_board_background(pygame, BOARD_PX)

	net = ReversiNet(config).to(DEVICE)
	path = resolve_model_path(config)
	if os.path.exists(path):
		load_model_weights(net, path)
		print(f"Model loaded: {path}")
	else:
		print("Warning: no model found, using random weights")
	net.eval()
	mcts = MCTS(net, config)

	def pick_human_color():
		if color_mode in (1, 2):
			return color_mode
		return random.choice([1, 2])

	human_color = pick_human_color()
	ai_color = 3 - human_color
	board = ReversiBoard()
	game_over = False
	last_move = -1

	def new_game():
		nonlocal board, game_over, last_move, human_color, ai_color
		human_color = pick_human_color()
		ai_color = 3 - human_color
		board = ReversiBoard()
		game_over = False
		last_move = -1

	def draw(status=""):
		# Board background from board.png (or solid fallback)
		screen.blit(board_bg, (0, 0))

		# Grid lines on top of the wood texture
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
					pygame.draw.circle(s, (255, 255, 255, 80),
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
					new_game()
			elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
				mx, my = event.pos
				if game_over:
					new_game()
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
