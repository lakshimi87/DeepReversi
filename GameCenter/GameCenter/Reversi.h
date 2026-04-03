#pragma once
#include <cstdint>

#define	RSIZE	8

class Reversi
{
public:
	Reversi();
	// Initialize and start a new game
	void Start();
	int GetTurn() const { return turn; }
	void SetPlayer(int idx, int p) { players[idx] = p; }
	const char *GetBoard() const { return board; }
	const int *GetScores() const { return scores; }
	// Bitboard representations for protocol
	uint64_t GetWhiteBits() const;
	uint64_t GetBlackBits() const;
	uint64_t GetHintBits() const;
	// Place a piece at position p, returns true if game continues
	bool Place(int p);

private:
	char board[RSIZE * RSIZE];
	int scores[3], hint;
	int turn;					// 1: White, 2: Black
	int players[2];
};
