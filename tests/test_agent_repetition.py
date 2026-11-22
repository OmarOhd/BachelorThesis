import unittest

import chess
import torch

from model.encoder import encode_move
from scripts.match.common import Agent, game_status
from scripts.match.repetition import check_move, record_position


# Each candidate completes either the second or third occurrence of a position.
# All histories are legal sequences replayed from the standard starting board.
POSITIONS = {
    "black": {
        2: ["Nf3", "Nf6", "Ng1"],
        3: ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"],
        "repeating": "Ng8",
        "alternative": "e6",
    },
    "white": {
        2: ["e4", "Nf6", "Nf3", "Ng8"],
        3: ["e4", "Nf6", "Nf3", "Ng8", "Ng1", "Nf6", "Nf3", "Ng8"],
        "repeating": "Ng1",
        "alternative": "d3",
    },
}


class FixedPolicy(torch.nn.Module):
    """Rank the repeating move first, the alternative second, all others lower."""

    def __init__(self, board, repeating, alternative):
        super().__init__()
        scores = torch.full((4352,), -100.0)
        scores[encode_move(repeating, board.turn)] = 20.0
        scores[encode_move(alternative, board.turn)] = 10.0

        self.register_buffer("policy", scores[:4096].reshape(1, 64, 64))
        self.register_buffer("promotion", scores[4096:].reshape(1, 8, 8, 4))
        self.register_buffer("value", torch.zeros((1, 3)))

    def forward(self, encoded):
        return self.policy, self.promotion, self.value


class AgentRepetitionTests(unittest.TestCase):
    @staticmethod
    def replay(moves):
        board = chess.Board()
        counts = {}
        fens = [board.fen()]
        record_position(board, counts)
        for san in moves:
            board.push_san(san)
            record_position(board, counts)
            fens.append(board.fen())
        return board, fens, counts

    @staticmethod
    def snapshot(board, fens, counts):
        return (
            board.fen(en_passant="fen"),
            list(board.move_stack),
            list(fens),
            dict(counts),
        )

    def assert_selection(self, history_depth, enabled, occurrences, use_alternative):
        for color, position in POSITIONS.items():
            with self.subTest(color=color, history=history_depth, enabled=enabled):
                board, fens, counts = self.replay(position[occurrences])
                self.assertEqual(board.turn, color == "white")
                before = self.snapshot(board, fens, counts)

                repeating = board.parse_san(position["repeating"])
                alternative = board.parse_san(position["alternative"])

                # A possible repetition on the next move must not end play yet.
                self.assertIsNone(game_status(board, counts, max_plies=100))
                self.assertEqual(check_move(board, repeating, counts), occurrences)
                self.assertEqual(check_move(board, alternative, counts), 1)

                # Skip __init__ because it loads a checkpoint. Supply just the
                # fields pick_move needs, while keeping pick_move itself real.
                agent = Agent.__new__(Agent)
                agent.history = history_depth
                agent.device = torch.device("cpu")
                agent.name = f"test-h{history_depth}"
                agent.model = FixedPolicy(board, repeating, alternative).eval()

                selected = agent.pick_move(board, fens, counts, enabled)
                expected = alternative if use_alternative else repeating
                self.assertEqual(selected, expected)

                # Inspecting candidates must leave the actual game untouched.
                self.assertEqual(self.snapshot(board, fens, counts), before)

    def test_h1_enabled_avoids_third_occurrence(self):
        self.assert_selection(
            history_depth=1, enabled=True, occurrences=3, use_alternative=True
        )

    def test_h1_disabled_keeps_top_ranked_move(self):
        self.assert_selection(
            history_depth=1, enabled=False, occurrences=3, use_alternative=False
        )

    def test_h8_enabled_keeps_top_ranked_move(self):
        self.assert_selection(
            history_depth=8, enabled=True, occurrences=3, use_alternative=False
        )

    def test_h8_disabled_keeps_top_ranked_move(self):
        self.assert_selection(
            history_depth=8, enabled=False, occurrences=3, use_alternative=False
        )

    def test_h1_enabled_allows_second_occurrence(self):
        self.assert_selection(
            history_depth=1, enabled=True, occurrences=2, use_alternative=False
        )


if __name__ == "__main__":
    unittest.main()
