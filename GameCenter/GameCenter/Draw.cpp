#define _CRT_SECURE_NO_WARNINGS
#include "Draw.h"
#include "GameCenter.h"
#include <stdio.h>
#include <string.h>
#include <string>
#include <objidl.h>
#include <gdiplus.h>

// Link GDI+ without touching the vcxproj
#pragma comment(lib, "gdiplus.lib")

// Module-private resources
static HDC memDC = NULL;
static HBITMAP memBitmap = NULL;
static Gdiplus::Bitmap *boardImage = NULL;
static ULONG_PTR gdiplusToken = 0;

// Walk from the exe directory upward, looking for resource/board.png.
// Build output paths vary (Debug/x64/..., .. projects root), so try a
// few levels before giving up.
static std::wstring FindBoardImage()
{
	wchar_t exePath[MAX_PATH];
	if (!GetModuleFileNameW(NULL, exePath, MAX_PATH)) return L"";
	std::wstring dir(exePath);
	size_t slash = dir.find_last_of(L"\\/");
	if (slash != std::wstring::npos) dir.resize(slash);

	static const wchar_t *rels[] = {
		L"\\resource\\board.png",
		L"\\..\\resource\\board.png",
		L"\\..\\..\\resource\\board.png",
		L"\\..\\..\\..\\resource\\board.png",
		L"\\..\\..\\..\\..\\resource\\board.png",
	};
	for (const wchar_t *rel : rels)
	{
		std::wstring candidate = dir + rel;
		DWORD attr = GetFileAttributesW(candidate.c_str());
		if (attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY))
			return candidate;
	}
	// Last-ditch: current working directory
	return L"resource\\board.png";
}

bool InitDraw(HWND hWnd)
{
	Gdiplus::GdiplusStartupInput gdiplusInput;
	if (Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusInput, NULL) != Gdiplus::Ok)
		return false;

	std::wstring path = FindBoardImage();
	if (!path.empty())
	{
		boardImage = Gdiplus::Bitmap::FromFile(path.c_str(), FALSE);
		if (boardImage && boardImage->GetLastStatus() != Gdiplus::Ok)
		{
			delete boardImage;
			boardImage = NULL;
		}
	}

	HDC hdc = GetDC(hWnd);
	memBitmap = CreateCompatibleBitmap(hdc, WIDTH, HEIGHT);
	memDC = CreateCompatibleDC(hdc);
	SelectObject(memDC, memBitmap);
	ReleaseDC(hWnd, hdc);
	return true;
}

void ShutdownDraw()
{
	if (memDC) { DeleteDC(memDC); memDC = NULL; }
	if (memBitmap) { DeleteObject(memBitmap); memBitmap = NULL; }
	if (boardImage) { delete boardImage; boardImage = NULL; }
	if (gdiplusToken) { Gdiplus::GdiplusShutdown(gdiplusToken); gdiplusToken = 0; }
}

void OnPaint(HWND hWnd)
{
	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(hWnd, &ps);
	Draw(memDC);
	BitBlt(hdc, 0, 0, WIDTH, HEIGHT, memDC, 0, 0, SRCCOPY);
	EndPaint(hWnd, &ps);
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

	const int boardPx = RSIZE * CSIZE;

	// Board background: scaled board.png if loaded, else the original
	// pale-yellow rectangle.
	if (boardImage)
	{
		Gdiplus::Graphics graphics(hdc);
		graphics.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
		graphics.DrawImage(boardImage,
			XOFFSET, YOFFSET, boardPx, boardPx);
	}
	else
	{
		SetDCPenColor(hdc, RGB(0, 0, 0));
		SetDCBrushColor(hdc, RGB(240, 240, 150));
		Rectangle(hdc, XOFFSET, YOFFSET,
			XOFFSET + boardPx + 2, YOFFSET + boardPx + 2);
	}

	// Draw grid lines on top of the background
	SetDCPenColor(hdc, RGB(0, 0, 0));
	for (int i = 0; i <= RSIZE; i++)
	{
		MoveToEx(hdc, XOFFSET + i * CSIZE, YOFFSET, 0);
		LineTo(hdc, XOFFSET + i * CSIZE, YOFFSET + boardPx);
		MoveToEx(hdc, XOFFSET, YOFFSET + i * CSIZE, 0);
		LineTo(hdc, XOFFSET + boardPx, YOFFSET + i * CSIZE);
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
