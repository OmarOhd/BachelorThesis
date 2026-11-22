import chess
import numpy as np

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING] # 0..5
PROMO_IDX = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3} # for promotion encoding

def encode_pieces(board: chess.Board):
    #Encode board piece placement as [64, 12] from side-to-move perspective.
    #Channels 0..5 are own pawn..king, 6..11 are opponent pawn..king.

    #1. flip if black to move
    if board.turn == chess.BLACK:
        board = board.mirror()
    #2. create 64x12 new tensor with zeros
    encoding = np.zeros((64, 12), dtype=np.float32)
    #3 for each of the 64 squares look up piece type and set appropriate channel to 1 for own pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_type = piece.piece_type
            if piece.color == chess.WHITE:
                channel = PIECE_ORDER.index(piece_type) # 0..5 for own pieces
            else:
                channel = PIECE_ORDER.index(piece_type) + 6 # 6..11 for opponent pieces
            encoding[square, channel] = 1.0
    return encoding

def encode_position(fens: list[str], halfmove_for_norm=None, *,
                    repetition_flags=None) -> np.ndarray:
    """
    fens: list of 1..8 FEN strings, chronological, current position LAST.
    repetition_flags: matching chronological 0/1 flags, or None for legacy zeros.
    Returns [64, 112] float32.
    """
    enc = np.zeros((64, 112), dtype=np.float32)

    for k, fen in enumerate(reversed(fens)):
        if k >= 8:
            break #safety check, should never happen since dataset should enforce max history of 8
        board = chess.Board(fen)
        piece_enc = encode_pieces(board) # [64, 12]
        start = k * 12
        enc[:, start:start+12] = piece_enc

    #auxiliary channels
    current_board = chess.Board(fens[-1])
    if current_board.turn == chess.BLACK:
        current_board = current_board.mirror()
    
    #96-103: current position first, then earlier positions; broadcast per square.
    #None preserves the input used to train every legacy checkpoint.
    if repetition_flags is not None:
        if len(repetition_flags) != len(fens):
            raise ValueError("Each FEN must have exactly one repetition flag")
        if any(flag not in (0, 1) for flag in repetition_flags):
            raise ValueError("Repetition flags must be binary (0 or 1)")
        for k, flag in enumerate(reversed(repetition_flags[-8:])):
            enc[:, 96 + k] = float(flag)

    #104-107 castling rights
    if current_board.has_kingside_castling_rights(chess.WHITE):
        enc[:, 104] = 1.0
    if current_board.has_queenside_castling_rights(chess.WHITE):
        enc[:, 105] = 1.0
    if current_board.has_kingside_castling_rights(chess.BLACK):
        enc[:, 106] = 1.0
    if current_board.has_queenside_castling_rights(chess.BLACK):
        enc[:, 107] = 1.0

    #108 color, post mirroring its always white
    enc[:, 108] = 1.0 # white to move after mirroring

    #109 no-progress count
    enc[:, 109] = current_board.halfmove_clock / 100.0 # normalize to [0,1]

    #110 en-passant: single square
    if current_board.has_legal_en_passant():
        enc[current_board.ep_square, 110] = 1.0
    
    #111 alls 1s for edge squares
    enc[:, 111] = 1.0
    return enc # [64, 112]

def encode_move(move: chess.Move, board_turn: chess.Color) -> int:
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion

    if board_turn == chess.BLACK:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)
    if promotion is not None:
        promo_idx = PROMO_IDX[promotion] # 0..3
        from_file = chess.square_file(from_square)
        to_file = chess.square_file(to_square)
        return (from_file * 8 + to_file) * 4 + promo_idx + 4096 # promotions are indexed after the 4096 normal moves, and within that ordered by from_file*8+to_file, then promo piece type
    else:
        return from_square * 64 + to_square # normal moves are indexed by from*64 + to, giving 4096 possible moves

def _check(uci, turn, expected):
    got = encode_move(chess.Move.from_uci(uci), turn)
    tag = "white" if turn == chess.WHITE else "black"
    ok = "OK " if got == expected else "FAIL"
    print(f"  [{ok}] {tag} {uci:6} -> {got:5}  (expected {expected})")
    return got == expected

def legal_move_mask(board) -> np.ndarray:
    #zeros (4352, bool); for mv in board.legal_moves: mask[encode_move(mv, board.turn)] = True
    mask = np.zeros(4352, dtype=np.bool_)
    for mv in board.legal_moves:
        idx = encode_move(mv, board.turn)
        mask[idx] = True
    return mask

if __name__ == "__main__":
    print("encode_move checks:")
    results = [
        # 1. mover-frame invariant: symmetric moves collapse
        _check("e2e4", chess.WHITE, 796),
        _check("e7e5", chess.BLACK, 796),
        # 2. straight queen promo
        _check("e7e8q", chess.WHITE, 4240),
        # 3. underpromotion ordering (same as #2, +3 for knight)
        _check("e7e8n", chess.WHITE, 4243),
        # 4. black promo mirrors onto the white frame -> same as #2
        _check("e2e1q", chess.BLACK, 4240),
        # 5. capture-promotion, file changes
        _check("d7c8q", chess.WHITE, 4200),
    ]
    print(f"\n{sum(results)}/{len(results)} passed")

    # 6. range guard: every encoding lands in [0, 4352)
    import itertools
    bad = 0
    for frm, to in itertools.product(range(64), range(64)):
        idx = encode_move(chess.Move(frm, to), chess.WHITE)
        if not (0 <= idx < 4352):
            bad += 1
    # plus a few promotions
    for uci in ["a7a8q", "h7h8r", "a7b8b", "h7g8n"]:
        idx = encode_move(chess.Move.from_uci(uci), chess.WHITE)
        if not (0 <= idx < 4352):
            bad += 1
    print(f"range guard: {bad} out-of-range encodings (expect 0)")

    #legal move masking test
    b = chess.Board()
    m = legal_move_mask(b)
    print(m.sum(), "(expect 20)")          # start position: 20 legal moves
    assert m[encode_move(chess.Move.from_uci("e2e4"), b.turn)]   # a known-legal move is marked

    b.push_uci("e2e4")                      # black to move -> mirror path exercised
    m2 = legal_move_mask(b)
    print(m2.sum(), "(expect 20)")
    assert m2[encode_move(chess.Move.from_uci("e7e5"), b.turn)]
