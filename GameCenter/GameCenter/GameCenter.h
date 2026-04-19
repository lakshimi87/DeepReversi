#pragma once

#include "framework.h"
#include "Resource.h"
#include "Reversi.h"

// Custom socket notifications — listen accept / per-client IO
#define	WM_ACCEPT			(WM_USER + 1)
#define	WM_CLIENT			(WM_USER + 2)

// Board geometry (cell size in px; board origin offset; overall window size)
#define	CSIZE	64
#define	XOFFSET	10
#define	YOFFSET	10
#define	WIDTH	(RSIZE * CSIZE + XOFFSET * 2 + 130)
#define	HEIGHT	(RSIZE * CSIZE + YOFFSET * 2)

#define	BUTTON_USER_GAME	(2001)

#define PLAYER_NOTASSIGN	0
#define PLAYER_NETWORK		1
#define PLAYER_USER			2

// Shared globals owned by GameCenter.cpp
extern HINSTANCE hInst;
extern int players[2];
extern Reversi game;
extern HWND startGame;

// Cross-module game-flow helpers
int GetPlayer(int player);
void StartGame(HWND hWnd);
void OnClose(HWND hWnd);
