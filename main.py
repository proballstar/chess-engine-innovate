import sys
import time
from typing import List, Tuple, Optional
import random
from pystockfish import Engine

# Constants
EMPTY = 0
WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING = range(1, 7)
BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING = range(7, 13)

# Bitboard constants
FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
RANK_1 = 0x00000000000000FF                                                                                                                                                                                                                                        
RANK_8 = 0xFF00000000000000

# Piece values and position tables (simplified)
PIECE_VALUES = [0, 100, 300, 300, 500, 900, 20000, -100, -300, -300, -500, -900, -20000]
POSITION_TABLES = [
    # Pawn
    [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    # Knight
    [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    # Bishop
    [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    # Rook
    [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ],
    # Queen
    [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    # King
    [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ],
    # Empty (placeholder for index 0)
    [0] * 64
]

class TranspositionTable:
    def __init__(self, size: int = 1000000):
        self.size = size
        self.table = {}

    def store(self, key: int, depth: int, flag: str, value: int, best_move: Optional[Tuple[int, int, int]]):
        self.table[key % self.size] = (key, depth, flag, value, best_move)

    def lookup(self, key: int) -> Optional[Tuple[int, str, int, Optional[Tuple[int, int, int]]]]:
        entry = self.table.get(key % self.size)
        if entry and entry[0] == key:
            return entry[1:]
        return None

# Use the transposition table in the minimax function

class ChessBoard:
    def __init__(self):
        self.pieces = [0] * 13
        self.color = 1 # 0 for white, 1 for black
        self.castling = 15  # 1111 in binary, representing KQkq castling rights
        self.ep = 0  # En passant target square
        self.halfmove = 0  # Halfmove clock for 50-move rule
        self.fullmove = 1  # Fullmove number
        self.history = []  # Move history for threefold repetition check
        self.tt = TranspositionTable()
        self.zobrist_table = self.zobrist_table = [
            [[random.randint(1, 2**64 - 1) for _ in range(12)] for _ in range(64)],
            [
                [random.randint(1, 2**64 - 1) for _ in range(2)],  # side to move
                [random.randint(1, 2**64 - 1) for _ in range(16)],  # castling rights (2^4 = 16 possibilities)
                [random.randint(1, 2**64 - 1) for _ in range(65)]   # en passant file (8 files + no ep = 9 possibilities)
            ]
        ]
        self.zobrist_hash_value = self.calculate_zobrist_hash()
        self.game_phase = self.calculate_game_phase()
        self.init_board()

    def is_square_defended(self, square: int) -> bool:
        original_color = self.color
        self.color ^= 1  # Switch color temporarily
        is_defended = self.is_square_attacked(square)
        self.color = original_color  # Restore original color
        return is_defended

    def least_valuable_attacker(self, square: int, side: bool) -> int:
        attackers = self.attackers_of(square, side)
        for piece_type in [WHITE_PAWN if side else BLACK_PAWN,
                           WHITE_KNIGHT if side else BLACK_KNIGHT,
                           WHITE_BISHOP if side else BLACK_BISHOP,
                           WHITE_ROOK if side else BLACK_ROOK,
                           WHITE_QUEEN if side else BLACK_QUEEN,
                           WHITE_KING if side else BLACK_KING]:
            if  piece_type and self.pieces and self.pieces[piece_type]:
                if any(attacker & self.pieces[piece_type] for attacker in attackers):
                    return piece_type
        return EMPTY

    def is_game_over(self) -> bool:
        return self.is_checkmate() or self.is_stalemate() or self.is_insufficient_material() or self.is_fifty_move_rule()

    def make_null_move(self):
        self.color ^= 1
        self.ep = 0
        self.zobrist_hash_value ^= self.zobrist_table[1][0][0]  # Flip side to move
        self.zobrist_hash_value ^= self.zobrist_table[1][2][self.ep if self.ep else 0]  # Clear en passant

    def undo_null_move(self):
        self.color ^= 1
        self.zobrist_hash_value ^= self.zobrist_table[1][0][0]  # Flip side to move back

    def is_endgame(self) -> bool:
        return self.game_phase <= 24  # Assuming a total game phase of 256

    def is_capture(self, move: Tuple[int, int, int]) -> bool:
        _, to_sq, _ = move
        return self.get_piece(to_sq) != EMPTY or (self.get_piece(to_sq) in [WHITE_PAWN, BLACK_PAWN] and to_sq == self.ep)

    def gives_check(self, move: Tuple[int, int, int]) -> bool:
        self.make_move(move)
        in_check = self.is_check()
        self.undo_move()
        return in_check

    def attackers_of(self, square: int, is_white: bool) -> List[int]:
        attackers = []
        color_offset = 0 if is_white else 6

        # Pawns
        pawn_attacks = [-7, -9] if is_white else [7, 9]
        for attack in pawn_attacks:
            if 0 <= square + attack < 64 and abs((square % 8) - ((square + attack) % 8)) == 1:
                if self.get_piece(square + attack) == WHITE_PAWN + color_offset:
                    attackers.append(WHITE_PAWN + color_offset)

        # Knights
        knight_moves = [-17, -15, -10, -6, 6, 10, 15, 17]
        for move in knight_moves:
            
            if 0 <= square + move < 64 and abs((square % 8) - ((square + move) % 8)) <= 2:
                if self.get_piece(square + move) == WHITE_KNIGHT + color_offset:
                    attackers.append(WHITE_KNIGHT + color_offset)

        # Bishops, Rooks, and Queens
        directions = [
            (-1, -1), (-1, 1), (1, -1), (1, 1),  # Diagonals
            (-1, 0), (1, 0), (0, -1), (0, 1)     # Horizontals and verticals
        ]
        for dx, dy in directions:
            x, y = square % 8, square // 8
            while True:
                x, y = x + dx, y + dy
                if not (0 <= x < 8 and 0 <= y < 8):
                    break
                target_square = y * 8 + x
                piece = self.get_piece(target_square)
                if piece != EMPTY:
                    if piece == WHITE_BISHOP + color_offset and (dx, dy) in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        attackers.append(piece)
                    elif piece == WHITE_ROOK + color_offset and (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        attackers.append(piece)
                    elif piece == WHITE_QUEEN + color_offset:
                        attackers.append(piece)
                    break

        # King
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in king_moves:
            x, y = (square % 8) + dx, (square // 8) + dy
            if 0 <= x < 8 and 0 <= y < 8:
                if self.get_piece(y * 8 + x) == WHITE_KING + color_offset:
                    attackers.append(WHITE_KING + color_offset)

        return attackers

    def promote_pawn(self, square: int, promotion_piece: int):
        pawn = WHITE_PAWN if square // 8 == 7 else BLACK_PAWN
        self.pieces[pawn] &= ~(1 << square)
        self.pieces[promotion_piece] |= 1 << square

    def piece_attacks(self, square: int, piece: int) -> int:
        # Implement attack patterns for each piece type
        # This is a simplified version and may need to be expanded
        attacks = 0
        if piece in [WHITE_KNIGHT, BLACK_KNIGHT]:
            knight_moves = [-17, -15, -10, -6, 6, 10, 15, 17]
            for move in knight_moves:
                if 0 <= square + move < 64 and abs((square % 8) - ((square + move) % 8)) <= 2:
                    attacks |= 1 << (square + move)
        elif piece in [WHITE_BISHOP, BLACK_BISHOP, WHITE_QUEEN, BLACK_QUEEN]:
            for direction in [-9, -7, 7, 9]:
                attacks |= self.ray_attacks(square, direction)
        if piece in [WHITE_ROOK, BLACK_ROOK, WHITE_QUEEN, BLACK_QUEEN]:
            for direction in [-8, -1, 1, 8]:
                attacks |= self.ray_attacks(square, direction)
        elif piece in [WHITE_KING, BLACK_KING]:
            king_moves = [-9, -8, -7, -1, 1, 7, 8, 9]
            for move in king_moves:
                if 0 <= square + move < 64 and abs((square % 8) - ((square + move) % 8)) <= 1:
                    attacks |= 1 << (square + move)
        return attacks

    def ray_attacks(self, square: int, direction: int) -> int:
        attacks = 0
        target = square + direction
        while 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 1:
            attacks |= 1 << target
            if self.get_piece(target) != EMPTY:
                break
            target += direction
        return attacks

    def calculate_game_phase(self) -> int:
        phase = 256
        phase -= bin(self.pieces[WHITE_KNIGHT] | self.pieces[BLACK_KNIGHT]).count('1') * 8
        phase -= bin(self.pieces[WHITE_BISHOP] | self.pieces[BLACK_BISHOP]).count('1') * 8
        phase -= bin(self.pieces[WHITE_ROOK] | self.pieces[BLACK_ROOK]).count('1') * 16
        phase -= bin(self.pieces[WHITE_QUEEN] | self.pieces[BLACK_QUEEN]).count('1') * 32
        return phase

    def zobrist_hash(self) -> int:
        return self.zobrist_hash_value

    def calculate_zobrist_hash(self) -> int:
        h = 0
        for piece in range(1, 13):
            bb = self.pieces[piece]
            while bb:
                square = bb.bit_length() - 1
                h ^= self.zobrist_table[0][square][piece - 1]
                bb &= bb - 1
        if self.color == 1:
            h ^= self.zobrist_table[1][0][0]
        h ^= self.zobrist_table[1][1][self.castling]
        h ^= self.zobrist_table[1][2][self.ep if self.ep else 0]
        return h

    def print_board(self):
        piece_symbols = {
            0: '.',
            1: '♙', 2: '♘', 3: '♗', 4: '♖', 5: '♕', 6: '♔',
            7: '♟', 8: '♞', 9: '♝', 10: '♜', 11: '♛', 12: '♚'
        }

        print('  a b c d e f g h')
        print(' +-----------------+')
        for rank in range(7, -1, -1):
            print(f'{rank+1}|', end=' ')
            for file in range(8):
                square = rank * 8 + file
                piece = self.get_piece(square)
                print(piece_symbols[piece], end=' ')
            print(f'|{rank+1}')
        print(' +-----------------+')
        print('  a b c d e f g h')

        print(f"\nSide to move: {'White' if self.color == 0 else 'Black'}")
        print(f"Fullmove number: {self.fullmove}")
        print(f"Halfmove clock: {self.halfmove}")
        print(f"En passant square: {'-' if self.ep == 0 else chr(self.ep % 8 + 97) + str(self.ep // 8 + 1)}")
        print(f"Castling rights: {'K' if self.castling & 1 else ''}{'Q' if self.castling & 2 else ''}{'k' if self.castling & 4 else ''}{'q' if self.castling & 8 else ''}")
    def init_board(self):
        self.pieces[1] = 0x000000000000FF00  # White pawns
        self.pieces[2] = 0x0000000000000042  # White knights
        self.pieces[3] = 0x0000000000000024  # White bishops
        self.pieces[4] = 0x0000000000000081  # White rooks
        self.pieces[5] = 0x0000000000000008  # White queen
        self.pieces[6] = 0x0000000000000010  # White king
        self.pieces[7] = 0x00FF000000000000  # Black pawns
        self.pieces[8] = 0x4200000000000000  # Black knights
        self.pieces[9] = 0x2400000000000000  # Black bishops
        self.pieces[10] = 0x8100000000000000  # Black rooks
        self.pieces[11] = 0x0800000000000000  # Black queen
        self.pieces[12] = 0x1000000000000000  # Black king
    def get_piece(self, square: int) -> int:
        for i, bb in enumerate(self.pieces):
            if bb & (1 << square):
                return i
        return EMPTY

    def generate_captures(self) -> List[Tuple[int, int, int]]:
        captures = []
        for move in self.generate_moves():
            from_sq, to_sq, promotion = move
            if self.get_piece(to_sq) != EMPTY or (self.get_piece(from_sq) in [WHITE_PAWN, BLACK_PAWN] and to_sq == self.ep):
                captures.append(move)
        return captures

    def make_move(self, move: Tuple[int, int, int]):
        from_sq, to_sq, promotion = move
        piece = self.get_piece(from_sq)
        captured = self.get_piece(to_sq)

        # Store state for undo_move
        state = (self.pieces[:], self.color, self.castling, self.ep, self.halfmove, self.fullmove)
        self.history.append(state)

        # Move piece
        self.pieces[piece] ^= (1 << from_sq) | (1 << to_sq)

        # Handle capture
        if captured:
            self.pieces[captured] &= ~(1 << to_sq)

        # Handle promotion
        if promotion:
            self.pieces[piece] &= ~(1 << to_sq)
            self.pieces[promotion] |= 1 << to_sq

        # Handle castling
        if piece in [WHITE_KING, BLACK_KING] and abs(from_sq - to_sq) == 2:
            if to_sq > from_sq:  # Kingside
                rook_from = from_sq + 3
                rook_to = from_sq + 1
            else:  # Queenside
                rook_from = from_sq - 4
                rook_to = from_sq - 1
            rook = WHITE_ROOK if piece == WHITE_KING else BLACK_ROOK
            self.pieces[rook] ^= (1 << rook_from) | (1 << rook_to)

        # Handle en passant
        if piece in [WHITE_PAWN, BLACK_PAWN] and to_sq == self.ep:
            cap_sq = to_sq + (-8 if self.color else 8)
            captured = self.get_piece(cap_sq)
            self.pieces[captured] &= ~(1 << cap_sq)

        # Update castling rights
        self.update_castling_rights(from_sq, to_sq)

        # Update en passant square
        self.ep = 0
        if piece in [WHITE_PAWN, BLACK_PAWN] and abs(from_sq - to_sq) == 16:
            self.ep = (from_sq + to_sq) // 2

        # Update move counters
        self.halfmove = 0 if piece in [WHITE_PAWN, BLACK_PAWN] or captured else self.halfmove + 1
        self.fullmove += self.color

        # Switch side to move
        self.color ^= 1
        self.zobrist_hash_value = self.calculate_zobrist_hash()

    def undo_move(self):
        if not self.history:
            return
        self.pieces, self.color, self.castling, self.ep, self.halfmove, self.fullmove = self.history.pop()
        self.zobrist_hash_value = self.calculate_zobrist_hash()

    def update_castling_rights(self, from_sq: int, to_sq: int):
        # Remove castling rights if king or rook moves
        if from_sq == 4 or to_sq == 4:  # White king
            self.castling &= 0b1100
        elif from_sq == 60 or to_sq == 60:  # Black king
            self.castling &= 0b0011
        elif from_sq == 0 or to_sq == 0:  # White queenside rook
            self.castling &= 0b1110
        elif from_sq == 7 or to_sq == 7:  # White kingside rook
            self.castling &= 0b1101
        elif from_sq == 56 or to_sq == 56:  # Black queenside rook
            self.castling &= 0b1011
        elif from_sq == 63 or to_sq == 63:  # Black kingside rook
            self.castling &= 0b0111

    def generate_moves(self) -> List[Tuple[int, int, int]]:
        moves = []
        for i in range(64):
            piece = self.get_piece(i)
            if piece and (piece < 7) == (self.color == 0):
                piece_moves = self.generate_piece_moves(i, piece)
                for move in piece_moves:
                    from_sq, to_sq, promotion = move
                    print(f"Generated move: {chr(from_sq%8+97)}{from_sq//8+1}{chr(to_sq%8+97)}{to_sq//8+1}")
                moves.extend(piece_moves)
        legal_moves = [move for move in moves if self.is_legal_move(move)]
        print(f"Generated {len(moves)} moves, {len(legal_moves)} are legal")
        return legal_moves

    def generate_piece_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        if piece in [WHITE_PAWN, BLACK_PAWN]:
            moves.extend(self.generate_pawn_moves(square, piece))
        elif piece in [WHITE_KNIGHT, BLACK_KNIGHT]:
            moves.extend(self.generate_knight_moves(square, piece))
        elif piece in [WHITE_BISHOP, BLACK_BISHOP, WHITE_QUEEN, BLACK_QUEEN]:
            moves.extend(self.generate_diagonal_moves(square, piece))
        if piece in [WHITE_ROOK, BLACK_ROOK, WHITE_QUEEN, BLACK_QUEEN]:
            moves.extend(self.generate_straight_moves(square, piece))
        elif piece in [WHITE_KING, BLACK_KING]:
            moves.extend(self.generate_king_moves(square, piece))
        return moves

    def generate_pawn_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        direction = 8 if piece == WHITE_PAWN else -8
        start_rank = 1 if piece == WHITE_PAWN else 6

        # Single push
        target = square + direction
        if 0 <= target < 64 and not self.get_piece(target):
            if target // 8 in [0, 7]:  # Promotion
                for promotion in [2, 3, 4, 5]:  # Knight, Bishop, Rook, Queen
                    moves.append((square, target, piece + promotion - 1))
            else:
                moves.append((square, target, 0))

            # Double push
            if square // 8 == start_rank:
                target = square + 2 * direction
                if 0 <= target < 64 and not self.get_piece(target):
                    moves.append((square, target, 0))

        # Captures
        for offset in [-1, 1]:
            target = square + direction + offset
            if 0 <= target < 64 and abs((square % 8) - (target % 8)) == 1:
                captured = self.get_piece(target)
                if captured and (captured < 7) != (piece < 7):
                    if target // 8 in [0, 7]:  # Promotion
                        for promotion in [2, 3, 4, 5]:  # Knight, Bishop, Rook, Queen
                            moves.append((square, target, piece + promotion - 1))
                    else:
                        moves.append((square, target, 0))
                elif target == self.ep:  # En passant
                    moves.append((square, target, 0))

        return moves

    def generate_knight_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        for offset in [-17, -15, -10, -6, 6, 10, 15, 17]:
            target = square + offset
            if 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 2:
                captured = self.get_piece(target)
                if not captured or (captured < 7) != (piece < 7):
                    moves.append((square, target, 0))
        return moves

    def generate_diagonal_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        for direction in [-9, -7, 7, 9]:
            target = square + direction
            while 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 1:
                captured = self.get_piece(target)
                if not captured:
                    moves.append((square, target, 0))
                elif (captured < 7) != (piece < 7):
                    moves.append((square, target, 0))
                    break
                else:
                    break
                target += direction
        return moves

    def generate_straight_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        for direction in [-8, -1, 1, 8]:
            target = square + direction
            while 0 <= target < 64 and (direction in [-8, 8] or abs((square % 8) - (target % 8)) <= 1):
                captured = self.get_piece(target)
                if not captured:
                    moves.append((square, target, 0))
                elif (captured < 7) != (piece < 7):
                    moves.append((square, target, 0))
                    break
                else:
                    break
                target += direction
        return moves

    def generate_king_moves(self, square: int, piece: int) -> List[Tuple[int, int, int]]:
        moves = []
        for offset in [-9, -8, -7, -1, 1, 7, 8, 9]:
            target = square + offset
            if 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 1:
                captured = self.get_piece(target)
                if not captured or (captured < 7) != (piece < 7):
                    moves.append((square, target, 0))

        # Castling
        if piece == WHITE_KING and square == 4:
            if self.castling & 1 and not self.get_piece(5) and not self.get_piece(6):
                moves.append((square, 6, 0))
            if self.castling & 2 and not self.get_piece(3) and not self.get_piece(2) and not self.get_piece(1):
                moves.append((square, 2, 0))
        elif piece == BLACK_KING and square == 60:
            if self.castling & 4 and not self.get_piece(61) and not self.get_piece(62):
                moves.append((square, 62, 0))
            if self.castling & 8 and not self.get_piece(59) and not self.get_piece(58) and not self.get_piece(57):
                moves.append((square, 58, 0))

        return moves

    def is_legal_move(self, move: Tuple[int, int, int]) -> bool:
        from_sq, to_sq, promotion = move
        piece = self.get_piece(from_sq)
        print(f"Checking move: {chr(from_sq%8+97)}{from_sq//8+1}{chr(to_sq%8+97)}{to_sq//8+1}")
        print(f"Piece: {piece}")
        self.make_move(move)
        is_legal = not self.is_check()
        self.undo_move()
        print(f"Is legal: {is_legal}")
        return is_legal

    def is_check(self) -> bool:
        king = WHITE_KING if self.color == 0 else BLACK_KING
        king_square = self.pieces[king].bit_length() - 1
        is_in_check = self.is_square_attacked(king_square)
        print(f"King on square {king_square}, is in check: {is_in_check}")
        return is_in_check

    def is_square_attacked(self, square: int) -> bool:
        print(f"Checking if square {square} is attacked")
        pawn = BLACK_PAWN if self.color == 0 else WHITE_PAWN
        pawn_attacks = [-7, -9] if self.color == 0 else [7, 9]
        for attack in pawn_attacks:
            target = square + attack
            if 0 <= target < 64 and abs((square % 8) - (target % 8)) == 1:
                if self.pieces[pawn] & (1 << target):
                    return True

        # Check for knight attacks
        knight = BLACK_KNIGHT if self.color == 0 else WHITE_KNIGHT
        knight_moves = [-17, -15, -10, -6, 6, 10, 15, 17]
        for move in knight_moves:
            target = square + move
            if 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 2:
                if self.pieces[knight] & (1 << target):
                    return True
        # Check for diagonal attacks (bishop and queen)
        bishop = BLACK_BISHOP if self.color == 0 else WHITE_BISHOP
        queen = BLACK_QUEEN if self.color == 0 else WHITE_QUEEN
        for direction in [-9, -7, 7, 9]:
            target = square + direction
            while 0 <= target < 64 and abs((square % 8) - (target % 8)) <= 1:
                if self.pieces[bishop] & (1 << target) or self.pieces[queen] & (1 << target):
                    return True
                if self.get_piece(target) != EMPTY:
                    break
                target += direction

        # Check for straight attacks (rook and queen)
        rook = BLACK_ROOK if self.color == 0 else WHITE_ROOK
        for direction in [-8, -1, 1, 8]:
            target = square + direction
            while 0 <= target < 64 and (direction in [-8, 8] or abs((square % 8) - (target % 8)) <= 1):
                if self.pieces[rook] & (1 << target) or self.pieces[queen] & (1 << target):
                    return True
                if self.get_piece(target) != EMPTY:
                    break
                target += direction

        # Check for king attacks
        king = BLACK_KING if self.color == 0 else WHITE_KING
        king_moves = [-9, -8, -7, -1, 1, 7, 8, 9]
        for move in king_moves:
            if 0 <= square + move < 64 and abs((square % 8) - ((square + move) % 8)) <= 1:
                if self.pieces[king] & (1 << (square + move)):
                    return True

        return False

    def is_checkmate(self) -> bool:
        return self.is_check() and not self.generate_moves()

    def is_stalemate(self) -> bool:
        return not self.is_check() and not self.generate_moves()

    def is_insufficient_material(self) -> bool:
        # King vs. King
        if sum(bin(x).count('1') for x in self.pieces) == 2:
            return True

        # King and Bishop vs. King or King and Knight vs. King
        if sum(bin(x).count('1') for x in self.pieces) == 3:
            if self.pieces[WHITE_BISHOP] or self.pieces[BLACK_BISHOP] or \
               self.pieces[WHITE_KNIGHT] or self.pieces[BLACK_KNIGHT]:
                return True

        # King and Bishop vs. King and Bishop with the bishops on the same color
        if sum(bin(x).count('1') for x in self.pieces) == 4:
            if self.pieces[WHITE_BISHOP] and self.pieces[BLACK_BISHOP]:
                white_bishop_square = self.pieces[WHITE_BISHOP].bit_length() - 1
                black_bishop_square = self.pieces[BLACK_BISHOP].bit_length() - 1
                if (white_bishop_square + black_bishop_square) % 2 == 0:
                    return True

        return False

    def is_threefold_repetition(self) -> bool:
        current_position = self.get_position_key()
        repetition_count = 1
        for past_position in reversed(self.history):
            if past_position[0] == current_position:
                repetition_count += 1
                if repetition_count == 3:
                    return True
        return False

    def is_fifty_move_rule(self) -> bool:
        return self.halfmove >= 100

    def is_draw(self) -> bool:
        return self.is_stalemate() or self.is_insufficient_material() or \
               self.is_threefold_repetition() or self.is_fifty_move_rule()

    def get_position_key(self) -> Tuple:
        return tuple(self.pieces + [self.castling, self.ep])

    def get_fen(self) -> str:
        fen = []
        for rank in range(7, -1, -1):
            empty = 0
            rank_fen = []
            for file in range(8):
                piece = self.get_piece(rank * 8 + file)
                if piece == EMPTY:
                    empty += 1
                else:
                    if empty > 0:
                        rank_fen.append(str(empty))
                        empty = 0
                    rank_fen.append("PNBRQKpnbrqk"[piece - 1])
            if empty > 0:
                rank_fen.append(str(empty))
            fen.append("".join(rank_fen))

        fen = "/".join(fen)
        fen += " w " if self.color == 0 else " b "

        castling = ""
        if self.castling & 1: castling += "K"
        if self.castling & 2: castling += "Q"
        if self.castling & 4: castling += "k"
        if self.castling & 8: castling += "q"
        fen += castling if castling else "-"

        fen += " " + ("-" if self.ep == 0 else chr(self.ep % 8 + 97) + str(self.ep // 8 + 1))
        fen += f" {self.halfmove} {self.fullmove}"

        return fen

    def set_from_fen(self, fen: str):
        parts = fen.split()
        self.pieces = [0] * 13

        # Board position
        for rank, row in enumerate(parts[0].split('/')[::-1]):
            file = 0
            for char in row:
                if char.isdigit():
                    file += int(char)
                else:
                    piece = "PNBRQKpnbrqk".index(char) + 1
                    self.pieces[piece] |= 1 << (rank * 8 + file)
                    file += 1

        # Active color
        self.color = 0 if parts[1] == 'w' else 1

        # Castling availability
        self.castling = 0
        if 'K' in parts[2]: 
            self.castling |= 1
        if 'Q' in parts[2]: 
            self.castling |= 2
        if 'k' in parts[2]: 
            self.castling |= 4
        if 'q' in parts[2]: 
            self.castling |= 8

        # En passant target square
        self.ep = 0 if parts[3] == '-' else (ord(parts[3][0]) - 97) + (int(parts[3][1]) - 1) * 8

        # Halfmove clock and fullmove number
        self.halfmove = int(parts[4])
        self.fullmove = int(parts[5])

        self.history = []

def quiescence(board: ChessBoard, alpha: int, beta: int, depth: int = 0) -> int:
    stand_pat = evaluate_board(board)
    if depth >= 4:  # Limit quiescence search depth
        return stand_pat
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in order_moves(board, board.generate_captures()):
        if static_exchange_evaluation(board, move) < 0:
            continue
        board.make_move(move)
        score = -quiescence(board, -beta, -alpha, depth + 1)
        board.undo_move()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

def evaluate_hanging_pieces(board: ChessBoard) -> int:
    hanging_pieces = 0
    for square in range(64):
        piece = board.get_piece(square)
        if piece != EMPTY:
            if board.is_square_attacked(square) and not board.is_square_defended(square):
                hanging_pieces += 1
    return hanging_pieces


def minimax(board: ChessBoard, depth: int, alpha: int, beta: int, maximizing_player: bool) -> int:
    if depth == 0:
        return quiescence(board, alpha, beta)

    tt_entry = board.tt.lookup(board.zobrist_hash())
    if tt_entry:
        stored_depth, flag, value, best_move = tt_entry
        if stored_depth >= depth:
            if flag == 'EXACT':
                return value
            elif flag == 'LOWERBOUND':
                alpha = max(alpha, value)
            elif flag == 'UPPERBOUND':
                beta = min(beta, value)
            if alpha >= beta:
                return value

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in order_moves(board, board.generate_moves()):
            board.make_move(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.undo_move()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        # Store the result in the transposition table
        if max_eval <= alpha:
            flag = 'UPPERBOUND'
        elif max_eval >= beta:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'
        board.tt.store(board.zobrist_hash(), depth, flag, int(max_eval), best_move)

        return int(max_eval)
    else:
        min_eval = float('inf')
        best_move = None
        for move in order_moves(board, board.generate_moves()):
            board.make_move(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.undo_move()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break

        # Store the result in the transposition table
        if min_eval <= alpha:
            flag = 'UPPERBOUND'
        elif min_eval >= beta:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'
        board.tt.store(board.zobrist_hash(), depth, flag,int(min_eval), best_move)

        return int(min_eval)

def null_move_pruning(board: ChessBoard, depth: int, alpha: int, beta: int, maximizing_player: bool) -> int:
    if depth < 3 or board.is_endgame():
        return minimax(board, depth, alpha, beta, maximizing_player)

    if not maximizing_player:
        board.make_null_move()
        score = -minimax(board, depth - 3, -beta, -beta + 1, True)
        board.undo_null_move()
        if score >= beta:
            return beta

    return minimax(board, depth, alpha, beta, maximizing_player)

def evaluate_piece_safety(board: ChessBoard) -> int:
    unsafe_score = 0
    for square in range(64):
        piece = board.get_piece(square)
        if piece != EMPTY:
            attackers = board.attackers_of(square, piece < 7)
            defenders = board.attackers_of(square, piece >= 7)
            if attackers:
                piece_value = PIECE_VALUES[piece]
                lowest_attacker_value = min(PIECE_VALUES[attacker] for attacker in attackers)
                if lowest_attacker_value < piece_value:
                    if not defenders:
                        unsafe_score += piece_value - lowest_attacker_value
                    else:
                        # Consider exchange sequences
                        exchange_value = static_exchange_evaluation(board, (square, square, 0))
                        if exchange_value < 0:
                            unsafe_score -= exchange_value
    return unsafe_score

def evaluate_board(board: ChessBoard) -> int:
    if board.is_checkmate():
        return -1000000 if board.color == 0 else 1000000
    if board.is_draw():
        return 0

    score = 0
    material_balance = 0
    for i, bb in enumerate(board.pieces):
        piece_value = PIECE_VALUES[i]
        piece_count = bin(bb).count('1')
        material_balance += piece_value * piece_count
        while bb:
            square = bb.bit_length() - 1
            score += piece_value
            score += POSITION_TABLES[i % 7][63 - square if i >= 7 else square]
            bb &= bb - 1

    score += material_balance

    # Evaluate piece safety
    score -= evaluate_piece_safety(board) * 30

    # Other evaluations...
    score += evaluate_pawn_structure(board)
    score += evaluate_king_safety(board)
    score += evaluate_mobility(board)
    score += evaluate_center_control(board)

    return score if board.color == 0 else -score

def evaluate_mobility(board: ChessBoard) -> int:
    white_mobility = len(board.generate_moves())
    board.color ^= 1
    black_mobility = len(board.generate_moves())
    board.color ^= 1
    return (white_mobility - black_mobility) * 10

def evaluate_pawn_structure(board: ChessBoard) -> int:
    score = 0
    white_pawns = board.pieces[WHITE_PAWN]
    black_pawns = board.pieces[BLACK_PAWN]

    # Evaluate doubled pawns
    white_doubled = bin(white_pawns & (white_pawns >> 8)).count('1')
    black_doubled = bin(black_pawns & (black_pawns << 8)).count('1')
    score -= (white_doubled - black_doubled) * 20

    # Evaluate isolated pawns
    white_isolated = bin(white_pawns & ~((white_pawns << 1) | (white_pawns >> 1))).count('1')
    black_isolated = bin(black_pawns & ~((black_pawns << 1) | (black_pawns >> 1))).count('1')
    score -= (white_isolated - black_isolated) * 15

    # Evaluate passed pawns
    white_passed = bin(white_pawns & ~((black_pawns << 8) | (black_pawns << 7) | (black_pawns << 9))).count('1')
    black_passed = bin(black_pawns & ~((white_pawns >> 8) | (white_pawns >> 7) | (white_pawns >> 9))).count('1')
    score += (white_passed - black_passed) * 30

    return score

def evaluate_king_safety(board: ChessBoard) -> int:
    score = 0
    white_king_square = board.pieces[WHITE_KING].bit_length() - 1
    black_king_square = board.pieces[BLACK_KING].bit_length() - 1

    # Evaluate pawn shield
    white_pawn_shield = bin(board.pieces[WHITE_PAWN] & king_shield_mask(white_king_square, True)).count('1')
    black_pawn_shield = bin(board.pieces[BLACK_PAWN] & king_shield_mask(black_king_square, False)).count('1')
    score += (white_pawn_shield - black_pawn_shield) * 10

    # Evaluate king tropism (closeness of enemy pieces to the king)
    white_tropism = sum(manhattan_distance(square, black_king_square) for square in get_piece_squares(board, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN))
    black_tropism = sum(manhattan_distance(square, white_king_square) for square in get_piece_squares(board, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN))
    score += (black_tropism - white_tropism) * 2

    return score

def evaluate_center_control(board: ChessBoard) -> int:
    center_squares = [27, 28, 35, 36]
    score = 0
    for square in center_squares:
        if board.is_square_attacked(square):
            score += 5 if board.color == 0 else -5
    return score

def evaluate_piece_coordination(board: ChessBoard) -> int:
    score = 0
    # Bonus for bishop pair
    if bin(board.pieces[WHITE_BISHOP]).count('1') >= 2:
        score += 50
    if bin(board.pieces[BLACK_BISHOP]).count('1') >= 2:
        score -= 50
    # Penalty for uncoordinated rooks
    if bin(board.pieces[WHITE_ROOK]).count('1') == 2 and not (board.pieces[WHITE_ROOK] & (board.pieces[WHITE_ROOK] - 1)):
        score -= 20
    if bin(board.pieces[BLACK_ROOK]).count('1') == 2 and not (board.pieces[BLACK_ROOK] & (board.pieces[BLACK_ROOK] - 1)):
        score += 20
    return score

# Helper functions
def king_shield_mask(king_square: int, is_white: bool) -> int:
    file = king_square % 8
    rank = king_square // 8
    shield = 0
    for f in range(max(0, file - 1), min(8, file + 2)):
        for r in range(rank + (1 if is_white else -1), rank + (3 if is_white else -3), 1 if is_white else -1):
            if 0 <= r < 8:
                shield |= 1 << (r * 8 + f)
    return shield

def manhattan_distance(sq1: int, sq2: int) -> int:
    return abs((sq1 % 8) - (sq2 % 8)) + abs((sq1 // 8) - (sq2 // 8))

def get_piece_squares(board: ChessBoard, *piece_types) -> List[int]:
    squares = []
    for piece_type in piece_types:
        bb = board.pieces[piece_type]
        while bb:
            squares.append(bb.bit_length() - 1)
            bb &= bb - 1
    return squares

def order_moves(board: ChessBoard, moves: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    def move_score(move):
        from_sq, to_sq, promotion = move
        piece = board.get_piece(from_sq)
        captured = board.get_piece(to_sq)
        score = 0

        # Prioritize captures
        if captured:
            score += 10 * PIECE_VALUES[captured] - PIECE_VALUES[piece]
        elif to_sq == board.ep and piece in [WHITE_PAWN, BLACK_PAWN]:
            score += 10 * PIECE_VALUES[WHITE_PAWN] - PIECE_VALUES[piece]  # En passant capture

        # Prioritize promotions
        if promotion:
            score += PIECE_VALUES[promotion]

        # Penalize moving to attacked squares
        if board.is_square_attacked(to_sq):
            score -= PIECE_VALUES[piece]

        # Bonus for controlling the center
        if to_sq in [27, 28, 35, 36]:
            score += 10

        # Bonus for developing pieces in the opening
        if board.fullmove <= 10 and piece in [WHITE_KNIGHT, WHITE_BISHOP, BLACK_KNIGHT, BLACK_BISHOP]:
            score += 5

        return score

    return sorted(moves, key=move_score, reverse=True)
def iterative_deepening(board: ChessBoard, time_limit: float) -> Optional[Tuple[int, int, int]]:
    start_time = time.time()
    best_move = None
    depth = 1
    max_depth = 75  # Increase this for stronger play
    alpha = -1000000  # Use a large negative integer instead of float('-inf')
    beta = 1000000   # Use a large positive integer instead of float('inf')

    while time.time() - start_time < time_limit and depth <= max_depth:
        score = minimax(board, depth, alpha, beta, board.color == 0)
        current_best_move = get_best_move_from_transposition_table(board)
        if current_best_move:
            best_move = current_best_move

        depth += 1
        if time.time() - start_time >= time_limit * 0.8:
            break

    return best_move
    
class UCIEngine:
    def __init__(self):
        self.board = ChessBoard()

    def uci(self):
        print("id name ComprehensiveChessEngine")
        print("id author Your Name")
        print("uciok")

    def isready(self):
        print("readyok")

    def ucinewgame(self):
        self.board = ChessBoard()

    def position(self, command):
        parts = command.split()
        if parts[1] == "startpos":
            self.board = ChessBoard()
            move_index = 3
        elif parts[1] == "fen":
            fen = " ".join(parts[2:8])
            self.board.set_from_fen(fen)
            move_index = 9
        else:
            return

        if len(parts) > move_index and parts[move_index] == "moves":
            for move in parts[move_index + 1:]:
                from_sq = (ord(move[0]) - ord('a')) + (8 * (ord(move[1]) - ord('1')))
                to_sq = (ord(move[2]) - ord('a')) + (8 * (ord(move[3]) - ord('1')))
                promotion = 0
                if len(move) == 5:
                    promotion = {'q': 5, 'r': 4, 'b': 3, 'n': 2}[move[4]]
                self.board.make_move((from_sq, to_sq, promotion))

    def go(self, command):
        parts = command.split()
        time_limit = 5.0  # Default time limit
        for i in range(0, len(parts), 2):
            if parts[i] == "wtime" and self.board.color == 0:
                time_limit = int(parts[i + 1]) / 1000.0 / 30  # Use 1/30 of remaining time
            elif parts[i] == "btime" and self.board.color == 1:
                time_limit = int(parts[i + 1]) / 1000.0 / 30  # Use 1/30 of remaining time
            elif parts[i] == "movetime":
                time_limit = int(parts[i + 1]) / 1000.0

        best_move = iterative_deepening(self.board, time_limit)
        if best_move:
            from_sq, to_sq, promotion = best_move
            move_str = f"{chr(from_sq % 8 + ord('a'))}{from_sq // 8 + 1}{chr(to_sq % 8 + ord('a'))}{to_sq // 8 + 1}"
            if promotion:
                move_str += "qrnb"[promotion - 2]
            print(f"bestmove {move_str}")
        else:
            print("No legal moves available or search failed")

        print("Board after move:")
        self.board.print_board()

    def quit(self):
        sys.exit()

def principal_variation_search(board: ChessBoard, depth: int, alpha: int, beta: int, maximizing_player: bool) -> int:
    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta)

    moves = order_moves(board, board.generate_moves())
    best_move = None
    initial_alpha = alpha

    for i, move in enumerate(moves):
        board.make_move(move)
        if i == 0:
            score = -principal_variation_search(board, depth - 1, -beta, -alpha, not maximizing_player)
        else:
            score = -principal_variation_search(board, depth - 1, -alpha - 1, -alpha, not maximizing_player)
            if alpha < score < beta:
                score = -principal_variation_search(board, depth - 1, -beta, -score, not maximizing_player)
        board.undo_move()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
            best_move = move

    # Store the best move in the transposition table
    if best_move:
        board.tt.store(board.zobrist_hash(), depth, 'PV' if alpha > initial_alpha else 'ALL', alpha, best_move)

    return alpha

def static_exchange_evaluation(board: ChessBoard, move: Tuple[int, int, int]) -> int:
    from_sq, to_sq, promotion = move
    target_piece = board.get_piece(to_sq)
    attacking_piece = board.get_piece(from_sq)

    if target_piece == EMPTY:
        return 0

    gain = [0] * 32
    depth = 0

    gain[depth] = PIECE_VALUES[target_piece]
    may_xray = (attacking_piece in [WHITE_PAWN, BLACK_PAWN, WHITE_BISHOP, BLACK_BISHOP, WHITE_ROOK, BLACK_ROOK, WHITE_QUEEN, BLACK_QUEEN])

    board.make_move(move)

    while True:
        depth += 1
        if depth >= len(gain):
            break
        # Find the least valuable attacker
        attacker = board.least_valuable_attacker(to_sq, depth % 2 == 0)
        if attacker == EMPTY:
            break

        # Add the new entry to the gain array
        gain[depth] = PIECE_VALUES[attacking_piece] - gain[depth-1]
        if max(-gain[depth-1], gain[depth]) < 0:
            break

        # Make the capture
        board.make_move((attacker, to_sq, 0))

        # If a pawn has reached the back rank, promote it to a queen
        if board.get_piece(to_sq) in [WHITE_PAWN, BLACK_PAWN] and to_sq // 8 in [0, 7]:
            board.promote_pawn(to_sq, WHITE_QUEEN if board.get_piece(to_sq) == WHITE_PAWN else BLACK_QUEEN)

        attacking_piece = board.get_piece(to_sq)
        may_xray = (attacking_piece in [WHITE_PAWN, BLACK_PAWN, WHITE_BISHOP, BLACK_BISHOP, WHITE_ROOK, BLACK_ROOK, WHITE_QUEEN, BLACK_QUEEN])

    # Undo the moves
    for _ in range(depth):
        board.undo_move()

    # Return the final score
    while depth > 0:
        depth -= 1
        gain[depth-1] = -max(-gain[depth-1], gain[depth])

    return gain[0]


def get_best_move_from_transposition_table(board: ChessBoard) -> Optional[Tuple[int, int, int]]:
    tt_entry = board.tt.lookup(board.zobrist_hash())
    if tt_entry:
        depth, flag, value, best_move = tt_entry
        return best_move
    return None

def main():
    engine = UCIEngine()
    engine.uci()
    engine.isready()
    engine.ucinewgame()
    engine.position("position startpos")
    
    while True:
        command = input("Enter your move (e.g., 'e2e4') or 'go' to let the engine move, or 'quit' to exit: ")
        if command == "quit":
            break
        elif command == "go":
            engine.go("go movetime 15000")  # 10 seconds per move
        else:
            # Assume the input is a move
            from_sq = (ord(command[0]) - ord('a')) + (8 * (ord(command[1]) - ord('1')))
            to_sq = (ord(command[2]) - ord('a')) + (8 * (ord(command[3]) - ord('1')))
            promotion = 0
            if len(command) == 5:
                promotion = {'q': 5, 'r': 4, 'b': 3, 'n': 2}[command[4]]
            move = (from_sq, to_sq, promotion)
            engine.board.make_move(move)
            print("Your move:")
            engine.board.print_board()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
