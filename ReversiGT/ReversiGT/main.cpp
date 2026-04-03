// Platform-specific includes
#ifdef _WIN32
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#include <winsock2.h>
#include <conio.h>
typedef SOCKET SocketHandle;
#define SOCKET_INVALID INVALID_SOCKET
#define socket_close closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/select.h>
#include <termios.h>
typedef int SocketHandle;
#define SOCKET_INVALID (-1)
#define socket_close close
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "limittbl.h"

// --- Constants ---

#define LEVEL_HARD		5
#define USE_SCOREBOARD	1
#define USE_PURE		2

static const int DXY[8] = { -8, -7, 1, 9, 8, 7, -1, -9 };

static const int ScoreBoard[64] =
{
	10,  1,  3,  2,  2,  3,  1, 10,
	 1, -5, -1, -1, -1, -1, -5,  1,
	 3, -1,  0,  0,  0,  0, -1,  3,
	 2, -1,  0,  0,  0,  0, -1,  2,
	 2, -1,  0,  0,  0,  0, -1,  2,
	 3, -1,  0,  0,  0,  0, -1,  3,
	 1, -5, -1, -1, -1, -1, -5,  1,
	10,  1,  3,  2,  2,  3,  1, 10
};

// --- Platform helpers ---

static bool NetworkInit()
{
#ifdef _WIN32
	WSADATA wsa;
	return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
#else
	return true;
#endif
}

static void NetworkCleanup()
{
#ifdef _WIN32
	WSACleanup();
#endif
}

#ifndef _WIN32
static struct termios origTermios;
static bool rawModeSet = false;

static void EnableRawMode()
{
	if (rawModeSet) return;
	tcgetattr(STDIN_FILENO, &origTermios);
	struct termios raw = origTermios;
	raw.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &raw);
	rawModeSet = true;
}

static void RestoreTermMode()
{
	if (!rawModeSet) return;
	tcsetattr(STDIN_FILENO, TCSANOW, &origTermios);
	rawModeSet = false;
}
#endif

static bool CheckQuitKey()
{
#ifdef _WIN32
	return _kbhit() && _getch() == 'q';
#else
	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(STDIN_FILENO, &fds);
	struct timeval tv = { 0, 0 };
	if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0)
	{
		char c;
		if (read(STDIN_FILENO, &c, 1) == 1 && c == 'q') return true;
	}
	return false;
#endif
}

// --- JSON helpers ---

static const char *FindJsonValue(const char *json, const char *key)
{
	char pattern[64];
	snprintf(pattern, sizeof(pattern), "\"%s\":", key);
	const char *p = strstr(json, pattern);
	if (!p) return nullptr;
	return p + strlen(pattern);
}

static char ParseCmd(const char *json)
{
	const char *p = FindJsonValue(json, "cmd");
	if (!p || *p != '"') return 0;
	return p[1];
}

static int ParseInt(const char *json, const char *key)
{
	const char *p = FindJsonValue(json, key);
	if (!p) return 0;
	return atoi(p);
}

static uint64_t ParseUint64(const char *json, const char *key)
{
	const char *p = FindJsonValue(json, key);
	if (!p) return 0;
	return strtoull(p, nullptr, 10);
}

// --- Network helpers ---
// Framing: 4-byte length (network byte order) + JSON string

static bool SendJson(SocketHandle sock, const char *json)
{
	int len = (int)strlen(json);
	uint32_t netLen = htonl((uint32_t)len);
	if (send(sock, (const char *)&netLen, 4, 0) != 4) return false;
	if (send(sock, json, len, 0) != len) return false;
	return true;
}

static bool RecvJson(SocketHandle sock, char *buf, int bufSize)
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

// --- Game board ---

struct GameBoard
{
	char cells[64];		// 0: hint(playable), 1: white, 2: black, 3: empty(not playable)
	int scores[3];		// [empty, white, black]
	int hintCount;

	// Convert protocol bitboards to internal representation
	void FromBitboards(uint64_t white, uint64_t black, uint64_t hint)
	{
		scores[1] = scores[2] = 0;
		hintCount = 0;
		for (int i = 0; i < 64; i++)
		{
			uint64_t bit = 1ULL << i;
			if (white & bit) { cells[i] = 1; scores[1]++; }
			else if (black & bit) { cells[i] = 2; scores[2]++; }
			else if (hint & bit) { cells[i] = 0; hintCount++; }
			else cells[i] = 3;
		}
		scores[0] = 64 - scores[1] - scores[2];
	}

	// Place a piece at pos (pass if pos >= 64).
	// Flips captured pieces and recalculates hints for the next player (anti).
	void Place(int pos, int turn)
	{
		int anti = turn ^ 3;

		if (pos < 64)
		{
			cells[pos] = turn;
			scores[0]--;
			scores[turn]++;

			// Flip opponent pieces in each direction
			for (int dir = 0; dir < 8; dir++)
			{
				int lim = limit[pos][dir];
				if (lim == 0) continue;
				int r = pos + DXY[dir];
				int k;
				for (k = 1; k < lim; k++)
				{
					if (cells[r] != anti) break;
					r += DXY[dir];
				}
				if (cells[r] == turn)
				{
					while (--k > 0)
					{
						r -= DXY[dir];
						cells[r] = turn;
						scores[turn]++;
						scores[anti]--;
					}
				}
			}
		}

		// Recalculate hints for the next player (anti)
		hintCount = 0;
		for (int i = 0; i < 64; i++)
		{
			if (cells[i] == 1 || cells[i] == 2) continue;
			cells[i] = 3;
			for (int dir = 0; dir < 8; dir++)
			{
				int lim = limit[i][dir];
				if (lim < 2) continue;
				int r = i + DXY[dir];
				int k;
				for (k = 1; k < lim; k++)
				{
					if (cells[r] != turn) break;
					r += DXY[dir];
				}
				if (k > 1 && cells[r] == anti)
				{
					cells[i] = 0;
					hintCount++;
					break;
				}
			}
		}
	}
};

// --- AI: minimax with alpha-beta pruning ---

static int Evaluate(const GameBoard &board, int method)
{
	if (method == USE_SCOREBOARD)
	{
		int score = 0;
		for (int i = 0; i < 64; i++)
		{
			if (board.cells[i] == 1) score += ScoreBoard[i];
			else if (board.cells[i] == 2) score -= ScoreBoard[i];
		}
		return score;
	}
	// Pure piece count difference
	return board.scores[1] - board.scores[2];
}

static int Search(const GameBoard &board, int depth, int maxDepth,
	int turn, int alpha, int beta, int method)
{
	if (depth >= maxDepth)
		return Evaluate(board, method);

	// No valid moves: pass or game over
	if (board.hintCount == 0)
	{
		if (!board.scores[0]) return Evaluate(board, method);
		GameBoard passed = board;
		passed.Place(64, turn);
		if (passed.hintCount == 0) return Evaluate(board, method);
		return Search(passed, depth + 1, maxDepth, turn ^ 3, alpha, beta, method);
	}

	int best = (turn == 1) ? -1000 : 1000;

	for (int pos = 0; pos < 64; pos++)
	{
		if (board.cells[pos] != 0) continue;

		GameBoard next = board;
		next.Place(pos, turn);
		int score = Search(next, depth + 1, maxDepth, turn ^ 3, alpha, beta, method);

		if (turn == 1)	// White maximizes
		{
			if (score > best) best = score;
			if (best > alpha) alpha = best;
		}
		else			// Black minimizes
		{
			if (score < best) best = score;
			if (best < beta) beta = best;
		}
		if (alpha >= beta) break;
	}

	return best;
}

static int GetOptimal(const GameBoard &board, int turnColor, int maxDepth, int method)
{
	if (board.hintCount == 0) return -1;

	// Only one valid move: return it immediately
	if (board.hintCount == 1)
	{
		for (int i = 0; i < 64; i++)
			if (board.cells[i] == 0) return i;
	}

	int bestPos = -1;
	int bestScore = (turnColor == 1) ? -1000 : 1000;
	int alpha = -1000, beta = 1000;

	for (int pos = 0; pos < 64; pos++)
	{
		if (board.cells[pos] != 0) continue;

		GameBoard next = board;
		next.Place(pos, turnColor);
		int score = Search(next, 1, maxDepth, turnColor ^ 3, alpha, beta, method);

		if (turnColor == 1)
		{
			if (score > bestScore) { bestScore = score; bestPos = pos; }
			if (bestScore > alpha) alpha = bestScore;
		}
		else
		{
			if (score < bestScore) { bestScore = score; bestPos = pos; }
			if (bestScore < beta) beta = bestScore;
		}
	}

	return bestPos;
}

// --- Player: network + AI integration ---

class Player
{
public:
	Player() : sock(SOCKET_INVALID), turnColor(0) {}
	~Player() { Disconnect(); }

	bool Connect(const char *host, int port)
	{
		sock = socket(AF_INET, SOCK_STREAM, 0);
		if (sock == SOCKET_INVALID) return false;

		struct sockaddr_in addr;
		memset(&addr, 0, sizeof(addr));
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = inet_addr(host);
		addr.sin_port = htons((uint16_t)port);

		if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) != 0)
		{
			Disconnect();
			return false;
		}

		// Receive setup message
		char buf[512];
		if (!RecvJson(sock, buf, sizeof(buf))) { Disconnect(); return false; }
		if (ParseCmd(buf) != 'S') { Disconnect(); return false; }

		turnColor = ParseInt(buf, "player");
		printf("Turn color: %s\n", turnColor == 1 ? "White" : "Black");
		return true;
	}

	void Disconnect()
	{
		if (sock != SOCKET_INVALID)
		{
			socket_close(sock);
			sock = SOCKET_INVALID;
		}
	}

	// Process one turn. Returns true to continue, false on game over or error.
	bool RunTurn()
	{
		char buf[512];
		if (!RecvJson(sock, buf, sizeof(buf))) return false;

		char cmd = ParseCmd(buf);

		if (cmd == 'Q')
		{
			int white = ParseInt(buf, "white");
			int black = ParseInt(buf, "black");
			printf("End of Game - White: %d, Black: %d\n", white, black);
			return false;
		}

		if (cmd != 'T') return false;

		// Parse bitboard state from server
		uint64_t white = ParseUint64(buf, "white");
		uint64_t black = ParseUint64(buf, "black");
		uint64_t hint = ParseUint64(buf, "hint");
		board.FromBitboards(white, black, hint);

		// Select search depth and evaluation method based on game phase
		int depth, method;
		if (board.scores[0] > 50)
			depth = LEVEL_HARD, method = USE_SCOREBOARD;
		else if (board.scores[0] > LEVEL_HARD + 4)
			depth = LEVEL_HARD + 2, method = USE_SCOREBOARD;
		else
			depth = board.scores[0], method = USE_PURE;

		int choice = GetOptimal(board, turnColor, depth, method);
		if (choice < 0) return false;

		// Send move to server
		snprintf(buf, sizeof(buf), "{\"cmd\":\"P\",\"pos\":%d}", choice);
		if (!SendJson(sock, buf)) return false;

		printf("Turn - White: %d, Black: %d, Place: %d\n",
			board.scores[1], board.scores[2], choice);
		return true;
	}

	bool IsOver() const
	{
		return !board.scores[0] || !board.scores[1] || !board.scores[2];
	}

private:
	SocketHandle sock;
	GameBoard board;
	int turnColor;
};

// --- Main ---

static bool PlayGame()
{
	Player player;
	if (!player.Connect("127.0.0.1", 8888)) return false;

	bool continueFlag = true;
	while (!player.IsOver())
	{
		if (CheckQuitKey()) continueFlag = false;
		if (!player.RunTurn()) break;
	}

	return continueFlag;
}

int main()
{
	if (!NetworkInit()) return 1;

#ifndef _WIN32
	EnableRawMode();
	atexit(RestoreTermMode);
#endif

	while (PlayGame());

	NetworkCleanup();
	return 0;
}
