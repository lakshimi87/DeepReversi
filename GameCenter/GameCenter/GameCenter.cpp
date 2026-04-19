#define _CRT_SECURE_NO_WARNINGS
#include "GameCenter.h"
#include "Network.h"
#include "Draw.h"
#include <stdlib.h>
#include <time.h>

// Globals declared in GameCenter.h
HINSTANCE hInst;
int players[2];
Reversi game;
HWND startGame;

ATOM MyRegisterClass(HINSTANCE hInstance);
HWND InitInstance(HINSTANCE, int);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow)
{
	srand((unsigned)time(0));

	MyRegisterClass(hInstance);

	HWND hWnd = InitInstance(hInstance, nCmdShow);
	if (!hWnd) return FALSE;

	if (!InitNetwork(hWnd)) return FALSE;

	// Main message loop
	MSG msg;
	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	ShutdownNetwork();
	ShutdownDraw();

	return (int)msg.wParam;
}

ATOM MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEX wcex;

	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_GAMECENTER));
	wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCE(IDC_GAMECENTER);
	wcex.lpszClassName = L"ReversiGC";
	wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

	return RegisterClassEx(&wcex);
}

HWND InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	hInst = hInstance;

	RECT rt = { 0, 0, WIDTH, HEIGHT };
	AdjustWindowRect(&rt, WS_OVERLAPPEDWINDOW, TRUE);

	HWND hWnd = CreateWindow(L"ReversiGC", L"Reversi Game Center",
		WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, rt.right - rt.left, rt.bottom - rt.top,
		NULL, NULL, hInstance, NULL);

	if (!hWnd) return 0;

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	return hWnd;
}

// Assign a player slot (random if both empty)
int GetPlayer(int player)
{
	int cand = (!players[0] && !players[1]) ? rand() & 1 : !players[1];
	players[cand] = player;
	return cand;
}

void StartGame(HWND hWnd)
{
	EnableWindow(startGame, FALSE);
	game.Start();
	int slot = game.GetTurn() - 1;
	if (players[slot] == PLAYER_NETWORK) SendBoard(slot);
	InvalidateRect(hWnd, 0, FALSE);
}

void OnClose(HWND hWnd)
{
	SendQuit();
	EnableWindow(startGame, TRUE);
	InvalidateRect(hWnd, 0, FALSE);
}

static void OnLButtonDown(HWND hWnd, int x, int y)
{
	int slot = game.GetTurn() - 1;
	if (players[slot] != PLAYER_USER) return;
	int nx = (x - XOFFSET) / CSIZE, ny = (y - YOFFSET) / CSIZE;
	if (nx < 0 || nx >= 8 || ny < 0 || ny >= 8) return;
	int p = ny * 8 + nx;
	if (game.GetBoard()[p] != 0) return;
	if (!game.Place(p)) { OnClose(hWnd); return; }
	slot = game.GetTurn() - 1;
	if (players[slot] == PLAYER_NETWORK) SendBoard(slot);
	InvalidateRect(hWnd, 0, FALSE);
}

static void OnCreate(HWND hWnd)
{
	InitDraw(hWnd);
	startGame = CreateWindow(L"BUTTON", L"User Game",
		WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
		WIDTH - 120, HEIGHT - 50, 110, 30,
		hWnd, (HMENU)BUTTON_USER_GAME,
		(HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
		NULL);
	EnableWindow(startGame, TRUE);
}

static int OnCommand(HWND hWnd, WPARAM wParam, LPARAM lParam)
{
	int wmId = LOWORD(wParam);
	int wmEvent = HIWORD(wParam);

	if (wmId == IDM_ABOUT) DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
	else if (wmId == IDM_EXIT) DestroyWindow(hWnd);
	else if (wmId == BUTTON_USER_GAME)
	{
		if (wmEvent == BN_CLICKED && players[0] != PLAYER_USER && players[1] != PLAYER_USER)
		{
			int cand = GetPlayer(PLAYER_USER);
			game.SetPlayer(cand, PLAYER_USER);
			EnableWindow(startGame, FALSE);
			if (players[0] && players[1]) StartGame(hWnd);
			InvalidateRect(hWnd, 0, TRUE);
		}
	}
	else return (int)DefWindowProc(hWnd, WM_COMMAND, wParam, lParam);
	return 0;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_CREATE:
		OnCreate(hWnd);
		break;
	case WM_COMMAND:
		return OnCommand(hWnd, wParam, lParam);
	case WM_PAINT:
		OnPaint(hWnd);
		break;
	case WM_LBUTTONDOWN:
		OnLButtonDown(hWnd, LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	case WM_ACCEPT:
		OnAccept(hWnd);
		break;
	case WM_CLIENT:
	case WM_CLIENT + 1:
		OnClient(hWnd, LOWORD(lParam), message - WM_CLIENT);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM)
{
	if (message == WM_INITDIALOG) return TRUE;
	if (message == WM_COMMAND)
	{
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
	}
	return (INT_PTR)FALSE;
}
