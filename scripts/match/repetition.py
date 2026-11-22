import chess

def hash_board(board: chess.Board) -> tuple:
    """
    takes the current board position, side to move, castling rights,
    en passant target, when an en passant capture is legal. not halfmove clock
    and fullmove number, as those change the hash and detection of duplicate positions
    becomes impossible and returns a hash of the mentioned parameters
    """

    positions = board.board_fen()
    side_to_move = board.turn
    castling_rights = board.castling_rights
    en_passant_legal = board.has_legal_en_passant()
    en_passant_square = board.ep_square if en_passant_legal else None


    position_key = (positions, side_to_move, castling_rights, en_passant_legal, en_passant_square)
    return position_key

def record_position(board: chess.Board, history: dict) -> None:
    """
    takes the current board position and a dict of previous positions
    and records the current position in the history
    """

    current_position_hash = hash_board(board)
    history[current_position_hash] = history.get(current_position_hash, 0) + 1

def occurence_tracker(board: chess.Board, history: dict) -> int:
    """
    takes the current board position and a dict of previous positions
    and returns the number of times the current position has occurred
    in the history
    """

    current_position_hash = hash_board(board)
    return history.get(current_position_hash, 0)

def check_move(board: chess.Board, move: chess.Move, history: dict) -> int:
    """
    takes the current board position, a move, and a dict of previous positions
    and returns the number of times the potential move's resulting position has occurred in the history
    """
    count = 0
    board.push(move)
    try:
        count = occurence_tracker(board, history)
        count += 1  # Include the current position in the count
    finally:
        board.pop()
    return count

def move_selector(board: chess.Board, legal_moves: list, history: dict, avoid_repetition: bool = True) -> chess.Move | None:
    """
    takes the current board position, a list of ranked legal moves from most desired to least desired, a dict of previous positions,
    and a boolean indicating whether to avoid repetition. Returns a legal move that does not lead to repetition.
    """

    if not legal_moves:
        return None

    if not avoid_repetition:
        return legal_moves[0] # Return the first legal move if repetition avoidance is not required

    non_repeating_moves = []
    for move in legal_moves:
        if check_move(board, move, history) < 3:
            non_repeating_moves.append(move)

    if not non_repeating_moves:
        return legal_moves[0]  # If all moves lead to repetition, return the first legal move

    return non_repeating_moves[0]


#test functions
def test_check_move():
    from unittest.mock import patch

    results = []

    def check(name, condition):
        results.append(bool(condition))
        print(f"{'PASS' if condition else 'FAIL'}: {name}")

    def replay(moves):
        board = chess.Board()
        history = {}
        record_position(board, history)

        for san in moves:
            board.push_san(san)
            record_position(board, history)

        return board, history

    def snapshot(board, history):
        return (
            board.fen(en_passant="fen"),
            list(board.move_stack),
            dict(history),
        )

    cases = [
        ("New position", [], "Nf3", 1),
        ("Second occurrence",
         ["Nf3", "Nf6", "Ng1"],
         "Ng8", 2),
        ("Third occurrence",
         ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"],
         "Ng8", 3),
        ("Non-repeating alternative",
         ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"],
         "e6", 1),
    ]

    for name, opening, candidate, expected in cases:
        board, history = replay(opening)
        move = board.parse_san(candidate)
        before = snapshot(board, history)

        actual = check_move(board, move, history)

        check(f"{name}: count is {expected}",
              actual == expected)

        check(f"{name}: board and history unchanged",
              snapshot(board, history) == before)

        again = check_move(board, move, history)

        check(f"{name}: repeated inspection is stable",
              again == expected and snapshot(board, history) == before)

    # Force the history lookup to fail AFTER the candidate was pushed.
    class FailingHistory(dict):
        def get(self, key, default=None):
            raise RuntimeError("Simulated lookup failure")

    board, history = replay(["Nf3", "Nf6", "Ng1"])
    history = FailingHistory(history)
    move = board.parse_san("Ng8")
    before = snapshot(board, history)
    caught = False

    try:
        check_move(board, move, history)
    except RuntimeError as error:
        caught = str(error) == "Simulated lookup failure"

    check("Lookup failure propagates to caller", caught)

    check("Lookup failure still restores board and history",
          snapshot(board, history) == before)

    # Force push to fail BEFORE it changes the board.
    board, history = replay(["Nf3", "Nf6", "Ng1"])
    move = board.parse_san("Ng8")
    before = snapshot(board, history)
    caught = False

    with patch.object(
        board,
        "push",
        side_effect=RuntimeError("Simulated push failure"),
    ):
        try:
            check_move(board, move, history)
        except RuntimeError as error:
            caught = str(error) == "Simulated push failure"

    check("Push failure propagates to caller", caught)

    check("Failed push does not undo a real game move",
          snapshot(board, history) == before)

    print(f"\n{sum(results)}/{len(results)} check_move checks passed.")

    if not all(results):
        raise AssertionError("Fix the failing checks, then run again.")

def test_move_selector():
    def snapshot(board, history, moves):
        return (
            board.fen(en_passant="fen"),
            list(board.move_stack),
            dict(history),
            list(moves),
        )

    def verify(name, board, history, moves, enabled, expected):
        before = snapshot(board, history, moves)
        actual = move_selector(board, moves, history, enabled)

        assert actual == expected, (
            f"{name}: expected {expected}, got {actual}"
        )
        assert snapshot(board, history, moves) == before, (
            f"{name}: board, history, or candidate list changed"
        )
        print(f"PASS: {name}")

    # Counts include the hypothetical candidate occurrence.
    # These synthetic histories isolate the selection rules.
    cases = [
        ("Empty list, filter on", [], True, None),
        ("Empty list, filter off", [], False, None),
        ("Filter off keeps first move", [3, 1, 1], False, 0),
        ("Acceptable first move wins", [1, 3, 1], True, 0),
        ("Second occurrence is allowed", [3, 2, 1], True, 1),
        ("Skip two repeating candidates", [3, 3, 1], True, 2),
        ("All repeating: use first move", [3, 3, 3], True, 0),
    ]

    for name, counts, enabled, expected_index in cases:
        board = chess.Board()
        history = {}
        record_position(board, history)

        moves = [
            board.parse_san(san)
            for san in ("Nf3", "Nc3", "e4")
        ]
        moves = moves[:len(counts)]

        for move, prospective_count in zip(moves, counts):
            board.push(move)
            try:
                for _ in range(prospective_count - 1):
                    record_position(board, history)
            finally:
                board.pop()

        expected = (
            None if expected_index is None else moves[expected_index]
        )
        verify(name, board, history, moves, enabled, expected)

    # Integration check using an actual legal move sequence.
    board = chess.Board()
    history = {}
    record_position(board, history)

    for san in ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"]:
        board.push_san(san)
        record_position(board, history)

    moves = [
        board.parse_san(san)
        for san in ("Ng8", "e6", "a6")
    ]
    verify(
        "Real history: avoid third occurrence",
        board, history, moves, True, moves[1],
    )

    print("\n8/8 move_selector cases passed.")

def main():
    results = []

    def check(name, condition):
        results.append(bool(condition))
        print(f"{'PASS' if condition else 'FAIL'}: {name}")

    # Recording and querying actual game positions.
    board = chess.Board()
    history = {}

    check("Unseen position returns 0 without changing history",
        occurence_tracker(board, history) == 0 and history == {})

    record_position(board, history)
    check("Initial position is counted once",
        occurence_tracker(board, history) == 1)

    before = history.copy()
    for _ in range(3):
        occurence_tracker(board, history)
    check("Repeated lookups do not change counts", history == before)

    cycle = ["Nf3", "Nf6", "Ng1", "Ng8"]
    for expected_count in (2, 3):
        for san in cycle:
            board.push_san(san)
            record_position(board, history)

        check(f"Starting position reaches count {expected_count}",
            occurence_tracker(board, history) == expected_count)

    check("Initial position and all eight plies are recorded",
        sum(history.values()) == 9)

    check("A fresh game's history starts empty",
        occurence_tracker(chess.Board(), {}) == 0)

    # Position identity.
    original = chess.Board()

    check("Equivalent board copies have identical keys",
        hash_board(original) == hash_board(original.copy()))

    changed = original.copy()
    changed.remove_piece_at(chess.A2)
    check("Piece placement changes the key",
        hash_board(original) != hash_board(changed))

    changed = original.copy()
    changed.turn = chess.BLACK
    check("Side to move changes the key",
        hash_board(original) != hash_board(changed))

    changed = original.copy()
    changed.castling_rights = 0
    check("Castling rights change the key",
        hash_board(original) != hash_board(changed))

    changed = original.copy()
    changed.halfmove_clock = 99
    changed.fullmove_number = 42
    check("Move counters do not change the key",
        hash_board(original) == hash_board(changed))

    # A potential en passant target with no legal capture.
    after_e4 = chess.Board()
    after_e4.push_san("e4")

    without_ep = after_e4.copy()
    without_ep.ep_square = None

    check("Unusable en passant target is ignored",
        hash_board(after_e4) == hash_board(without_ep))

    # A legal en passant capture must distinguish the position.
    ep_d6 = chess.Board("7k/8/8/1pPp4/8/8/8/K7 w - d6 0 10")

    without_ep = ep_d6.copy()
    without_ep.ep_square = None

    check("Legal en passant availability changes the key",
        hash_board(ep_d6) != hash_board(without_ep))

    # Capturing en passant would expose White's king to the rook.
    pinned = chess.Board("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 10")

    without_ep = pinned.copy()
    without_ep.ep_square = None

    check("En passant forbidden by a pin is ignored",
        hash_board(pinned) == hash_board(without_ep))

    # Same pieces and turn, but two different legal target squares.
    ep_b6 = chess.Board("7k/8/8/1pPp4/8/8/8/K7 w - b6 0 10")

    check("Different legal en passant targets produce different keys",
        hash_board(ep_b6) != hash_board(ep_d6))

    print(f"\n{sum(results)}/{len(results)} tests passed.")

    if not all(results):
        raise AssertionError("Fix the failing checks, then run again.")


if __name__ == "__main__":
    main()
    test_check_move()
    test_move_selector()