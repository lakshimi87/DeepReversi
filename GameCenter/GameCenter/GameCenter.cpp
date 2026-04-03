#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#include "framework.h"
#include "GameCenter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <commdlg.h>
#include <winsock2.h>
#include <time.h>
#include <cstdint>
#include "Reversi.h"

#define	WM_ACCEPT			(WM_USER+1)
#define	WM_CLIENT			(WM_USER+2)

#define	CSIZE	64
#define	XOFFSET	10
#define	YOFFSET	10
#define	WIDTH	(RSIZE * CSIZE + XOFFSET * 2 + 130)
#define	HEIGHT	(RSIZE * CSIZE + YOFFSET * 2)

#define	BUTTON_USER_GAME	(2001)

#define PLAYER_NOTASSIGN	0
#define PLAYER_NETWORK		1
#define PLAYER_USER			2

// Global Variables
HINSTANCE hInst;
SOCKET hListen = INVALID_SOCKET;
SOCKET sockClient[2] = { INVALID_SOCKET, INVALID_SOCKET };
int players[2];
HDC memDC;
HWND startGame;
Reversi game;

int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow)
{
	ATOM MyRegisterClass(HINSTANCE hInstance);
	HWND InitInstance(HINSTANCE, int);
	srand((unsigned)time(0));

	// WSA network initialize
	WSADATA wsa;
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return FALSE;

	MyRegisterClass(hInstance);

	HWND hWnd = InitInstance(hInstance, nCmdShow);
	if (!hWnd) return FALSE;

	// Create listen socket
	hListen = socket(AF_INET, SOCK_STREAM, 0);
	if (hListen == INVALID_SOCKET) return FALSE;

	SOCKADDR_IN addr;
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	addr.sin_port = htons(8888);
	bind(hListen, (SOCKADDR*)&addr, sizeof(addr));

	listen(hListen, 2);
	WSAAsyncSelect(hListen, hWnd, WM_ACCEPT, FD_ACCEPT);

	// Main message loop
	MSG msg;
	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	closesocket(hListen);
	WSACleanup();

	return (int)msg.wParam;
}

ATOM MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEX wcex;
	LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

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

// --- JSON protocol helpers ---
// Message framing: 4-byte length (network byte order) + JSON string

bool SendJson(SOCKET sock, const char *json)
{
	int len = (int)strlen(json);
	uint32_t netLen = htonl((uint32_t)len);
	if (send(sock, (const char *)&netLen, 4, 0) != 4) return false;
	if (send(sock, json, len, 0) != len) return false;
	return true;
}

bool RecvJson(SOCKET sock, char *buf, int bufSize)
{
	// Read 4-byte length header
	uint32_t netLen = 0;
	int received = 0;
	while (received < 4)
	{
		int c = recv(sock, ((char *)&netLen) + received, 4 - received, 0);
		if (c <= 0) return false;
		received += c;
	}
	int len = (int)ntohl(netLen);
	if (len <= 0 || len >= bufSize) return false;

	// Read JSON payload
	received = 0;
	while (received < len)
	{
		int c = recv(sock, buf + received, len - received, 0);
		if (c <= 0) return false;
		received += c;
	}
	buf[len] = '\0';
	return true;
}

// Send board state as JSON with bitboards
void SendBoard(int slot)
{
	char json[256];
	sprintf(json, "{\"cmd\":\"T\",\"white\":%llu,\"black\":%llu,\"hint\":%llu}",
		game.GetWhiteBits(), game.GetBlackBits(), game.GetHintBits());
	SendJson(sockClient[slot], json);
}

// Send quit message with final scores
void SendQuit()
{
	char json[128];
	sprintf(json, "{\"cmd\":\"Q\",\"white\":%d,\"black\":%d}",
		game.GetScores()[1], game.GetScores()[2]);
	for (int i = 0; i < 2; i++)
	{
		if (players[i] == PLAYER_NETWORK && sockClient[i] != INVALID_SOCKET)
		{
			SendJson(sockClient[i], json);
			closesocket(sockClient[i]);
			sockClient[i] = INVALID_SOCKET;
		}
		players[i] = PLAYER_NOTASSIGN;
	}
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

void OnAccept(HWND hWnd)
{
	if (players[0] && players[1]) return;
	int cand = GetPlayer(PLAYER_NETWORK);
	game.SetPlayer(cand, PLAYER_NETWORK);

	SOCKADDR_IN addr;
	int len = sizeof(addr);
	sockClient[cand] = accept(hListen, (SOCKADDR*)&addr, &len);
	WSAAsyncSelect(sockClient[cand], hWnd, WM_CLIENT + cand, FD_READ | FD_CLOSE);

	// Send setup message with player number
	char json[64];
	sprintf(json, "{\"cmd\":\"S\",\"player\":%d}", cand + 1);
	SendJson(sockClient[cand], json);

	if (players[0] && players[1]) StartGame(hWnd);
	InvalidateRect(hWnd, 0, FALSE);
}

void OnClient(HWND hWnd, int issue, int slot)
{
	if (sockClient[slot] == INVALID_SOCKET) return;
	if (issue == FD_CLOSE) { OnClose(hWnd); return; }

	char buf[1024];
	if (!RecvJson(sockClient[slot], buf, sizeof(buf))) { OnClose(hWnd); return; }

	// Parse JSON command
	const char *cmdStr = strstr(buf, "\"cmd\":\"");
	if (!cmdStr || cmdStr[7] != 'P') { OnClose(hWnd); return; }

	// Parse position value
	const char *posStr = strstr(buf, "\"pos\":");
	if (!posStr) { OnClose(hWnd); return; }
	int place = atoi(posStr + 6);

	if (!game.Place(place)) { OnClose(hWnd); return; }
	int nextSlot = game.GetTurn() - 1;
	if (players[nextSlot] == PLAYER_NETWORK) SendBoard(nextSlot);
	InvalidateRect(hWnd, 0, FALSE);
}

void OnLButtonDown(HWND hWnd, int x, int y)
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

void OnCreate(HWND hWnd)
{
	HDC hdc = GetDC(hWnd);
	HBITMAP hBitmap = CreateCompatibleBitmap(hdc, WIDTH, HEIGHT);
	memDC = CreateCompatibleDC(hdc);
	SelectObject(memDC, hBitmap);
	ReleaseDC(hWnd, hdc);
	startGame = CreateWindow(L"BUTTON", L"User Game",
		WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
		WIDTH - 120, HEIGHT - 50, 110, 30,
		hWnd, (HMENU)BUTTON_USER_GAME,
		(HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
		NULL);
	EnableWindow(startGame, TRUE);
}

int OnCommand(HWND hWnd, WPARAM wParam, LPARAM lParam)
{
	INT_PTR CALLBACK About(HWND, UINT, WPARAM, LPARAM);
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

void OnPaint(HWND hWnd)
{
	void Draw(HDC hdc);

	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(hWnd, &ps);
	Draw(memDC);
	BitBlt(hdc, 0, 0, WIDTH, HEIGHT, memDC, 0, 0, SRCCOPY);
	EndPaint(hWnd, &ps);
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

void Draw(HDC hdc)
{
	const char *board = game.GetBoard();
	const int *scores = game.GetScores();
	HGDIOBJ oldObj = SelectObject(hdc, GetStockObject(DC_BRUSH));
	SelectObject(hdc, GetStockObject(DC_PEN));

	// Clear background
	SetDCBrushColor(hdc, RGB(255, 255, 255));
	SetDCPenColor(hdc, RGB(255, 0, 0));
	Rectangle(hdc, 0, 0, WIDTH, HEIGHT);

	// Draw board background
	SetDCPenColor(hdc, RGB(0, 0, 0));
	SetDCBrushColor(hdc, RGB(240, 240, 150));
	Rectangle(hdc, XOFFSET, YOFFSET, XOFFSET + RSIZE * CSIZE + 2, YOFFSET + RSIZE * CSIZE + 2);

	// Draw grid lines
	for (int i = 0; i <= RSIZE; i++)
	{
		MoveToEx(hdc, XOFFSET + i * CSIZE, YOFFSET, 0);
		LineTo(hdc, XOFFSET + i * CSIZE, YOFFSET + RSIZE * CSIZE);
		MoveToEx(hdc, XOFFSET, YOFFSET + i * CSIZE, 0);
		LineTo(hdc, XOFFSET + RSIZE * CSIZE, YOFFSET + i * CSIZE);
	}

	// Draw pieces and hint markers
	for (int r = 0; r < RSIZE; r++)
	{
		for (int c = 0; c < RSIZE; c++)
		{
			int idx = r * RSIZE + c;
			if (board[idx] == 3) continue;
			if (board[idx] == 0)
			{
				// Hint marker: small green rectangle
				SetDCBrushColor(hdc, RGB(100, 180, 100));
				Rectangle(hdc, XOFFSET + CSIZE * c + 24, YOFFSET + CSIZE * r + 24,
					XOFFSET + CSIZE * c + CSIZE - 24, YOFFSET + CSIZE * r + CSIZE - 24);
			}
			else
			{
				// Piece: white or black
				SetDCBrushColor(hdc, (board[idx] == 1) ? RGB(255, 255, 255) : RGB(30, 30, 30));
				Ellipse(hdc, XOFFSET + CSIZE * c + 4, YOFFSET + CSIZE * r + 4,
					XOFFSET + CSIZE * c + CSIZE - 4, YOFFSET + CSIZE * r + CSIZE - 4);
			}
		}
	}

	// Draw status text
	char str[128];
	int len;
	const char *playerstr[3] = { "Wait", "Network", "User" };
	len = sprintf(str, "Player W : %s", playerstr[players[0]]);
	TextOutA(hdc, XOFFSET * 2 + RSIZE * CSIZE, YOFFSET + 20, str, len);
	len = sprintf(str, "Player B : %s", playerstr[players[1]]);
	TextOutA(hdc, XOFFSET * 2 + RSIZE * CSIZE, YOFFSET + 40, str, len);
	len = sprintf(str, "Score  W : %3d", scores[1]);
	TextOutA(hdc, XOFFSET * 2 + RSIZE * CSIZE, YOFFSET + 80, str, len);
	len = sprintf(str, "Score  B : %3d", scores[2]);
	TextOutA(hdc, XOFFSET * 2 + RSIZE * CSIZE, YOFFSET + 100, str, len);
	SelectObject(hdc, oldObj);
}
