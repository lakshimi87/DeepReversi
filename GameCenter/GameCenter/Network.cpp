#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#include "Network.h"
#include "GameCenter.h"
#include <stdio.h>
#include <string.h>
#include <cstdint>
#include <winsock2.h>

// Sockets — one listen socket plus one per assigned client slot
SOCKET hListen = INVALID_SOCKET;
SOCKET sockClient[2] = { INVALID_SOCKET, INVALID_SOCKET };

bool InitNetwork(HWND hWnd)
{
	WSADATA wsa;
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;

	hListen = socket(AF_INET, SOCK_STREAM, 0);
	if (hListen == INVALID_SOCKET) return false;

	SOCKADDR_IN addr;
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	addr.sin_port = htons(8888);
	bind(hListen, (SOCKADDR*)&addr, sizeof(addr));

	listen(hListen, 2);
	WSAAsyncSelect(hListen, hWnd, WM_ACCEPT, FD_ACCEPT);
	return true;
}

void ShutdownNetwork()
{
	if (hListen != INVALID_SOCKET)
	{
		closesocket(hListen);
		hListen = INVALID_SOCKET;
	}
	WSACleanup();
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

// Send quit message with final scores; disconnect any network peers
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
