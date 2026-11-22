import csv
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import chess
import chess.pgn
import torch

from model.encoder import encode_move
from scripts.match import dump_games, play_match
from scripts.match.common import Agent, play_game


# After 1.Nf3 Nf6 2.Ng1 Ng8 3.Nf3 Nf6 4.Ng1, Black is the only model to move.
# 4...Ng8 repeats the start position for the third time. 4...e6 avoids it.
OPENING = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]
MAX_PLIES = len(OPENING) + 1
# (filter enabled, expected h1 move as Black, expected Black interventions)
CASES = [(True, "e7e6", 1), (False, "f6g8", 0)]


def fixed_agent(history):
    """Use the real pick_move with a CPU policy ranking ...Ng8 above ...e6."""
    policy = torch.zeros((1, 64, 64))
    flat = policy.view(-1)
    flat[encode_move(chess.Move.from_uci("f6g8"), chess.BLACK)] = 2.0
    flat[encode_move(chess.Move.from_uci("e7e6"), chess.BLACK)] = 1.0

    agent = Agent.__new__(Agent)  # Bypass checkpoint loading.
    agent.name = f"h{history}"
    agent.history = history
    agent.device = torch.device("cpu")
    agent.model = Mock(return_value=(policy, torch.zeros(1, 8, 8, 4), torch.zeros(1, 3)))
    return agent


def run_cli(module, enabled, *arguments):
    flag = "--avoid-repetition" if enabled else "--no-avoid-repetition"
    argv = [module.__name__, "--max-plies", str(MAX_PLIES), flag, *arguments]
    with (
        patch.object(module, "Agent", side_effect=[fixed_agent(8), fixed_agent(1)]),
        patch.object(module, "collect_test_openings", return_value=[("knights", OPENING)]),
        patch.object(module, "pick_device", return_value=torch.device("cpu")),
        patch.object(sys, "argv", argv),
        redirect_stdout(io.StringIO()),
    ):
        module.main()


class RepetitionInterventionTests(unittest.TestCase):
    def setUp(self):
        # play_match writes match_results.csv in the current directory.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(directory.name)

    def test_play_game_counts_only_the_black_override(self):
        # Reuse agents so a stale flag from the ON game fails the OFF case.
        white, black = fixed_agent(8), fixed_agent(1)
        for enabled, expected_move, expected_count in CASES:
            with self.subTest(enabled=enabled):
                board, _, _, counts = play_game(white, black, OPENING, MAX_PLIES, enabled)
                self.assertEqual([m.uci() for m in board.move_stack], [*OPENING, expected_move])
                self.assertEqual(board.is_repetition(3), not enabled)
                self.assertEqual(counts, {chess.WHITE: 0, chess.BLACK: expected_count})

    def test_csv_counts_by_color(self):
        white_col = "white_repetition_interventions"
        black_col = "black_repetition_interventions"
        for enabled, _, expected_count in CASES:
            with self.subTest(enabled=enabled):
                run_cli(play_match, enabled, "--ckpt-a", "h8.pt", "--ckpt-b", "h1.pt",
                        "--openings", "1")
                with Path("match_results.csv").open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames[-2:], [white_col, black_col])
                    rows = [(r["white"], r["black"], r[white_col], r[black_col]) for r in reader]
                self.assertEqual(rows, [
                    ("h8", "h1", "0", str(expected_count)),
                    ("h1", "h8", "0", "0"),
                ])

    def test_pgn_counts_by_color(self):
        for enabled, expected_move, expected_count in CASES:
            with self.subTest(enabled=enabled):
                with patch.object(dump_games, "PAIRINGS", [("knights", "h8.pt", "h1.pt")]):
                    run_cli(dump_games, enabled, "--games", "2", "--out", "pgn")

                for filename in ("knights.pgn", "all_games.pgn"):
                    with self.subTest(filename=filename):
                        with (Path("pgn") / filename).open(encoding="utf-8") as stream:
                            games = list(iter(lambda: chess.pgn.read_game(stream), None))
                        rows = [
                            (g.headers["White"], g.headers["Black"],
                             g.headers["WhiteRepetitionInterventions"],
                             g.headers["BlackRepetitionInterventions"], g.end().move.uci())
                            for g in games
                        ]
                        self.assertEqual(rows, [
                            ("h8", "h1", "0", str(expected_count), expected_move),
                            ("h1", "h8", "0", "0", "f6g8"),
                        ])


if __name__ == "__main__":
    unittest.main()
