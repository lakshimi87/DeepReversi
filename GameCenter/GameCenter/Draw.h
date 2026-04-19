#pragma once

#include "framework.h"

// One-time GDI+ startup + board.png load.  Idempotent; returns false on
// GDI+ init failure (image failure is non-fatal — Draw falls back to
// the original solid-yellow board).
bool InitDraw(HWND hWnd);
void ShutdownDraw();

// WM_PAINT handler and the raw draw-to-memDC routine
void OnPaint(HWND hWnd);
void Draw(HDC hdc);
