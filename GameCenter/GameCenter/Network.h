#pragma once

#include "framework.h"
#include <winsock2.h>

// Sockets owned by Network.cpp
extern SOCKET hListen;
extern SOCKET sockClient[2];

// Bring up / tear down WSA, listening socket, async-select on hWnd
bool InitNetwork(HWND hWnd);
void ShutdownNetwork();

// Protocol helpers: length-prefixed JSON frames
bool SendJson(SOCKET sock, const char *json);
bool RecvJson(SOCKET sock, char *buf, int bufSize);

// Outbound game messages
void SendBoard(int slot);
void SendQuit();

// Async-select event handlers dispatched from WndProc
void OnAccept(HWND hWnd);
void OnClient(HWND hWnd, int issue, int slot);
