import unittest

import chess
import chess.engine
import chess.pgn

from scripts.analysis.analyze_games import analyse_game


class SavedScores:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def analyse(self, board, limit):
        score = self.scores[self.calls]
        self.calls += 1
        if isinstance(score, int):
            score = chess.engine.Cp(score)
        return {"score": chess.engine.PovScore(score, chess.WHITE)}


def game_from_moves(moves, *, fen=chess.STARTING_FEN, termination="threefold", result="1/2-1/2"):
    board = chess.Board(fen)
    for move in moves:
        board.push_uci(move)
    game = chess.pgn.Game.from_board(board)
    game.headers.update(White="white", Black="black", Result=result, Termination=termination)
    return game


class TerminalDrawTests(unittest.TestCase):
    def check_final_draw(self, game, before):
        moves = list(game.mainline_moves())
        mover = not game.end().board().turn
        sign = 1 if mover == chess.WHITE else -1
        scores = [23] * (len(moves) + 1)
        scores[-2:] = [sign * before, sign * -300]
        engine = SavedScores(scores)
        rows = analyse_game(game, engine, chess.engine.Limit(depth=1))
        self.assertEqual(engine.calls, len(moves) + 1)
        self.assertEqual(rows[-1]["eval_before"], before)
        self.assertEqual(rows[-1]["eval_after"], 0)
        self.assertEqual(rows[-1]["cp_loss"], max(0, before))
        # All earlier evaluations and losses must retain the supplied engine values.
        board = game.board()
        for i, move in enumerate(moves[:-1]):
            pov = 1 if board.turn == chess.WHITE else -1
            self.assertEqual(rows[i]["eval_before"], pov * scores[i])
            self.assertEqual(rows[i]["eval_after"], pov * scores[i + 1])
            self.assertEqual(rows[i]["cp_loss"], max(0, pov * (scores[i] - scores[i + 1])))
            board.push(move)

    def test_actual_third_occurrence_for_both_colours(self):
        cases = [
            (chess.STARTING_FEN, ["g1f3", "g8f6", "f3g1", "f6g8"] * 2),
            (chess.STARTING_FEN.replace(" w ", " b "), ["g8f6", "g1f3", "f6g8", "f3g1"] * 2),
        ]
        for fen, moves in cases:
            for before in (-100, 50):
                with self.subTest(fen=fen, before=before):
                    game = game_from_moves(moves, fen=fen)
                    self.assertTrue(game.end().board().is_repetition(3))
                    self.check_final_draw(game, before)

    def test_second_occurrence_is_not_terminal(self):
        game = game_from_moves(["g1f3", "g8f6", "f3g1", "f6g8"])
        self.assertFalse(game.end().board().is_repetition(3))
        rows = analyse_game(game, SavedScores([40] * 5), None)
        self.assertEqual(rows[-1]["eval_after"], -40)

    def test_prospective_repetition_is_not_terminal(self):
        game = game_from_moves(["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"])
        board = game.end().board()
        self.assertTrue(board.can_claim_threefold_repetition())
        self.assertFalse(board.is_repetition(3))
        rows = analyse_game(game, SavedScores([40] * 8), None)
        self.assertEqual(rows[-1]["eval_after"], 40)

    def test_actual_fifty_move_draw(self):
        game = game_from_moves(["g1f3"], fen=chess.STARTING_FEN.replace(" 0 1", " 99 1"), termination="fifty-move")
        self.assertTrue(game.end().board().is_fifty_moves())
        self.check_final_draw(game, 50)

    def test_prospective_fifty_move_claim_is_not_terminal(self):
        game = game_from_moves(["g1f3"], fen=chess.STARTING_FEN.replace(" 0 1", " 98 1"), termination="fifty-move")
        board = game.end().board()
        self.assertTrue(board.can_claim_fifty_moves())
        self.assertFalse(board.is_fifty_moves())
        rows = analyse_game(game, SavedScores([40, -300]), None)
        self.assertEqual(rows[-1]["eval_after"], -300)
        self.assertEqual(rows[-1]["cp_loss"], 340)

    def test_insufficient_material(self):
        game = game_from_moves(["a1b2"], fen="7k/8/8/8/8/8/1b6/K7 w - - 0 1", termination="INSUFFICIENT_MATERIAL")
        self.assertTrue(game.end().board().is_insufficient_material())
        self.check_final_draw(game, -100)

    def test_stalemate(self):
        game = game_from_moves(["g5g6"], fen="7k/5K2/8/6Q1/8/8/8/8 w - - 0 1", termination="STALEMATE")
        self.assertTrue(game.end().board().is_stalemate())
        self.check_final_draw(game, 50)

    def test_checkmate_signs_stay_correct(self):
        white = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1")
        for board, move, winner in [(white, "g6g7", chess.WHITE), (white.mirror(), "g3g2", chess.BLACK)]:
            with self.subTest(winner=winner):
                game = game_from_moves([move], fen=board.fen(), termination="CHECKMATE", result="1-0" if winner else "0-1")
                self.assertTrue(game.end().board().is_checkmate())
                scores = [chess.engine.Mate(1), chess.engine.MateGiven] if winner else [chess.engine.Mate(-1), chess.engine.Mate(0)]
                rows = analyse_game(game, SavedScores(scores), None)
                self.assertEqual(rows[-1]["eval_after"], 1000)
                self.assertEqual(rows[-1]["cp_loss"], 0)

    def test_nonterminal_draw_headers_keep_engine_score(self):
        for termination in ("normal", "ply-cap", ""):
            with self.subTest(termination=termination):
                game = game_from_moves(["e2e4"], termination=termination)
                rows = analyse_game(game, SavedScores([40, -300]), None)
                self.assertEqual(rows[-1]["eval_after"], -300)

    def test_empty_game(self):
        game = game_from_moves([])
        self.assertEqual(analyse_game(game, SavedScores([0]), None), [])


if __name__ == "__main__":
    unittest.main()
