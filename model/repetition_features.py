import chess


REPETITION_MODES = ("none", "binary")


def validate_repetition_features(mode):
    if mode not in REPETITION_MODES:
        raise ValueError(f"Unknown repetition_features {mode!r}; expected {REPETITION_MODES}")
    return mode


def checkpoint_repetition_features(checkpoint):
    # Checkpoints predating these features were trained with all eight bits zero.
    return validate_repetition_features(checkpoint.get("repetition_features", "none"))


def repetition_flags_for_history(board, fens):
    if not 1 <= len(fens) <= 8:
        raise ValueError("Expected 1..8 chronological FENs")
    if board.root().fen() != chess.STARTING_FEN:
        raise ValueError("Repetition features need the full move stack from the starting position")
    probe = board.copy(stack=True)
    flags = []
    for i in range(len(fens) - 1, -1, -1):
        if probe.fen() != fens[i]:
            raise ValueError("FEN history does not match the board's move stack")
        flags.append(int(probe.is_repetition(2)))
        if i:
            if not probe.move_stack:
                raise ValueError("The board's move stack is shorter than its FEN history")
            probe.pop()
    return flags[::-1]
