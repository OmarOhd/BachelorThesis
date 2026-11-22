import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import chess
import chess.pgn
import numpy as np
import torch

from data_pipeline.build_repetition_flags import build_repetition_flags
from data_pipeline.dataset import TCECPlies
from data_pipeline.repetition_cache import cache_paths, sha256_file
from model.encoder import encode_position, encode_move
from model.repetition_features import checkpoint_repetition_features, repetition_flags_for_history
from model import eval_model, train_model
from scripts.match.common import Agent, play_game
from scripts.match import dump_games, play_match
from scripts.match.repetition import record_position


CYCLE = ['g1f3', 'g8f6', 'f3g1', 'f6g8']
LONG_CYCLE = ['g1f3', 'g8f6', 'b1c3', 'b8c6', 'f3g1', 'f6g8', 'c3b1', 'c6b8']


def replay(moves):
    board = chess.Board()
    fens, counts = [board.fen()], {}
    record_position(board, counts)
    for uci in moves:
        board.push_uci(uci)
        fens.append(board.fen())
        record_position(board, counts)
    return board, fens, counts


def write_fixture(directory, games=None):
    games = games if games is not None else [CYCLE * 2 + ['e2e4'], ['d2d4', 'd7d5']]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'plies.jsonl'
    offsets = []
    with source.open('wb') as stream:
        for game_index, moves in enumerate(games):
            board = chess.Board()
            for ply, uci in enumerate(moves):
                record = {'game_id': f'game-{game_index}', 'ply': ply, 'fen': board.fen(),
                          'move': uci, 'result': 1}
                offsets.append(stream.tell())
                stream.write((json.dumps(record) + '\n').encode())
                board.push_uci(uci)
    np.save(directory / 'plies.offsets.npy', np.array(offsets, dtype=np.int64))
    for split in ('train', 'val', 'test'):
        np.save(directory / f'{split}_indices.npy', np.arange(len(offsets)))
    return source


class CapturePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('policy', torch.zeros(1, 64, 64))
        self.register_buffer('promotion', torch.zeros(1, 8, 8, 4))
        self.register_buffer('value', torch.zeros(1, 3))
        self.last_input = None

    def forward(self, inputs):
        self.last_input = inputs.detach().cpu().clone()
        size = inputs.shape[0]
        return (self.policy.expand(size, -1, -1),
                self.promotion.expand(size, -1, -1, -1), self.value.expand(size, -1))


class RepetitionFeatureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = write_fixture(self.directory)

    def dataset(self, **kwargs):
        return TCECPlies(plies_path=self.source,
                         offsets_path=self.directory / 'plies.offsets.npy', **kwargs)

    def build(self):
        return build_repetition_flags(self.source, progress_every=0)

    def test_first_second_third_occurrences_and_game_reset(self):
        stats = self.build()
        flags = np.load(cache_paths(self.source)[0])
        self.assertEqual(flags.tolist(), [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
        self.assertEqual(stats['repeated_positions'], 5)
        self.assertEqual(stats['n_games'], 2)

    def test_repetition_older_than_eight_input_positions(self):
        self.source = write_fixture(self.directory, [LONG_CYCLE + ['e2e4']])
        self.build()
        board, fens, _ = replay(LONG_CYCLE)
        flags = repetition_flags_for_history(board, fens[-8:])
        self.assertEqual(flags, [0] * 7 + [1])
        encoded = self.dataset(history=8, repetition_features='binary')[8]['board'].numpy()
        self.assertTrue(np.all(encoded[:, 96] == 1))
        self.assertTrue(np.all(encoded[:, 97:104] == 0))

    def test_historical_flag_is_not_retroactively_changed(self):
        board, fens, _ = replay(CYCLE)
        before = board.fen(), list(board.move_stack)
        self.assertEqual(repetition_flags_for_history(board, fens), [0, 0, 0, 0, 1])
        self.assertEqual((board.fen(), list(board.move_stack)), before)

    def test_missing_history_is_zero_and_only_flag_channels_change(self):
        board, fens, _ = replay(CYCLE)
        legacy = encode_position(fens)
        enabled = encode_position(fens, repetition_flags=repetition_flags_for_history(board, fens))
        np.testing.assert_array_equal(enabled[:, :96], legacy[:, :96])
        np.testing.assert_array_equal(enabled[:, 104:], legacy[:, 104:])
        self.assertEqual(enabled.shape, (64, 112))
        self.assertEqual(enabled.dtype, np.float32)
        self.assertTrue(np.all(enabled[:, 96] == 1))
        self.assertTrue(np.all(enabled[:, 97:104] == 0))

    def test_castling_rights_are_part_of_repetition_identity(self):
        moves = ['h2h3', 'h7h6'] + ['h1h2', 'h8h7', 'h2h1', 'h7h8'] * 2 + ['e2e4']
        self.source = write_fixture(self.directory, [moves])
        self.build()
        flags = np.load(cache_paths(self.source)[0])
        self.assertEqual([int(flags[i]) for i in (2, 6, 10)], [0, 0, 1])

    def test_irrelevant_ep_and_move_clocks_do_not_hide_repetition(self):
        moves = ['e2e4', 'g8f6', 'g1f3', 'f6g8', 'f3g1', 'e7e5']
        self.source = write_fixture(self.directory, [moves])
        self.build()
        flags = np.load(cache_paths(self.source)[0])
        self.assertEqual([int(flags[i]) for i in (1, 5)], [0, 1])

    def test_default_h1_has_no_flags_and_explicit_h1_binary_has_only_current_flag(self):
        self.build()
        legacy = self.dataset(history=1)[4]['board'].numpy()
        explicit = self.dataset(history=1, repetition_features='binary')[4]['board'].numpy()
        self.assertTrue(np.all(legacy[:, 96:104] == 0))
        self.assertTrue(np.all(explicit[:, 96] == 1))
        self.assertTrue(np.all(explicit[:, 12:96] == 0))
        self.assertTrue(np.all(explicit[:, 97:104] == 0))

    def test_default_dataset_needs_no_cache(self):
        self.assertTrue(torch.all(self.dataset()[4]['board'][:, 96:104] == 0))
        with self.assertRaisesRegex(FileNotFoundError, 'build_repetition_flags'):
            self.dataset(repetition_features='binary')

    def test_flag_cache_build_does_not_modify_dataset_or_splits(self):
        paths = [self.source, *self.directory.glob('*.npy')]
        before = {path: sha256_file(path) for path in paths}
        self.build()
        self.assertEqual({path: sha256_file(path) for path in paths}, before)

    def test_rewritten_or_reordered_dataset_rejects_stale_cache(self):
        self.build()
        # Same records, but different bytes: even this must invalidate the pairing.
        self.source.write_bytes(self.source.read_bytes().replace(b'"result": 1', b'"result":  1'))
        with self.assertRaisesRegex(ValueError, 'different plies.jsonl'):
            self.dataset(repetition_features='binary')

    def test_corrupted_flags_are_rejected(self):
        self.build()
        flags_path, _ = cache_paths(self.source)
        flags_path.write_bytes(flags_path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.dataset(repetition_features='binary')

    def test_missing_prefix_or_gaps_are_rejected_without_partial_outputs(self):
        lines = self.source.read_bytes().splitlines(keepends=True)
        self.source.write_bytes(b''.join(lines[1:]))
        with self.assertRaisesRegex(ValueError, 'expected ply 0'):
            self.build()
        self.assertFalse(cache_paths(self.source)[0].exists())

    def test_wrong_fen_is_rejected(self):
        rows = [json.loads(line) for line in self.source.read_text().splitlines()]
        rows[1]['fen'] = chess.STARTING_FEN
        self.source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'FEN disagrees'):
            self.build()

    def test_isolated_fen_and_mismatched_history_are_rejected(self):
        board, fens, _ = replay(CYCLE)
        with self.assertRaisesRegex(ValueError, 'full move stack'):
            repetition_flags_for_history(chess.Board(board.fen()), [board.fen()])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            repetition_flags_for_history(board, [chess.STARTING_FEN])

    def test_invalid_encoder_flags_and_checkpoint_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            encode_position([chess.STARTING_FEN], repetition_flags=[])
        with self.assertRaisesRegex(ValueError, 'binary'):
            encode_position([chess.STARTING_FEN], repetition_flags=[2])
        self.assertEqual(checkpoint_repetition_features({}), 'none')
        with self.assertRaises(ValueError):
            checkpoint_repetition_features({'repetition_features': 'typo'})

    def test_dataset_and_real_agent_encode_identically_for_both_colours(self):
        self.build()
        ds = self.dataset(history=8, repetition_features='binary')
        for ply in (0, 4, 5, 8):
            with self.subTest(ply=ply):
                board, fens, counts = replay((CYCLE * 2)[:ply])
                agent = Agent.__new__(Agent)
                agent.history, agent.repetition_features = 8, 'binary'
                agent.device = torch.device('cpu')
                agent.model = CapturePolicy()
                agent.pick_move(board, fens, counts, avoid_repetition=False)
                torch.testing.assert_close(agent.model.last_input[0], ds[ply]['board'], rtol=0, atol=0)

    def test_checkpoint_metadata_drives_agent_without_changing_old_checkpoints(self):
        board, fens, counts = replay(CYCLE)
        for mode in (None, 'none', 'binary'):
            with self.subTest(mode=mode):
                model = CapturePolicy()
                checkpoint = {'history': 8, 'step': 1, 'model_state_dict': model.state_dict()}
                if mode is not None:
                    checkpoint['repetition_features'] = mode
                path = self.directory / 'model.pt'
                torch.save(checkpoint, path)
                with patch('scripts.match.common.cf_6M_2024', return_value=model), redirect_stdout(io.StringIO()):
                    agent = Agent(path, torch.device('cpu'))
                agent.pick_move(board, fens, counts, False)
                self.assertEqual(agent.repetition_features, mode or 'none')
                self.assertTrue(torch.all(model.last_input[0, :, 96] == int(mode == 'binary')))

    def test_opening_history_is_available_during_play_game(self):
        white = Agent.__new__(Agent)
        white.name, white.history, white.repetition_features = 'h8-rep', 8, 'binary'
        white.device, white.model = torch.device('cpu'), CapturePolicy()
        black = Agent.__new__(Agent)
        black.name, black.history = 'h8', 8
        black.device, black.model = torch.device('cpu'), CapturePolicy()
        board, _, _, counts = play_game(white, black, CYCLE, 5, avoid_repetition=False)
        self.assertEqual(board.ply(), 5)
        self.assertTrue(torch.all(white.model.last_input[0, :, 96] == 1))
        self.assertEqual(counts, {chess.WHITE: 0, chess.BLACK: 0})

    def test_features_do_not_add_an_external_h8_move_filter(self):
        board, fens, counts = replay(CYCLE + CYCLE[:3])
        agent = Agent.__new__(Agent)
        agent.history, agent.repetition_features = 8, 'binary'
        agent.device, agent.model = torch.device('cpu'), CapturePolicy()
        repeat = board.parse_san('Ng8')
        agent.model.policy.view(-1)[encode_move(repeat, board.turn)] = 10
        self.assertEqual(agent.pick_move(board, fens, counts, True), repeat)
        self.assertFalse(agent.last_repetition_intervention)

    def test_eval_reads_checkpoint_mode_and_exports_it(self):
        self.build()
        model = CapturePolicy()
        checkpoint = self.directory / 'eval.pt'
        torch.save({'history': 8, 'repetition_features': 'binary',
                    'model_state_dict': model.state_dict()}, checkpoint)
        output = self.directory / 'metrics.csv'
        argv = ['eval_model', '--ckpt', str(checkpoint), '--split', 'test', '--by-game', str(output)]
        with (patch.object(eval_model, 'TCECPlies', side_effect=self.dataset),
              patch.object(eval_model, 'PROCESSED_DIR', self.directory),
              patch.object(eval_model, 'cf_6M_2024', return_value=model),
              patch('sys.argv', argv), patch.object(eval_model, 'pick_device', return_value=torch.device('cpu')),
              redirect_stdout(io.StringIO())):
            eval_model.main()
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(sum(int(row['n']) for row in rows), 11)
        self.assertEqual({row['repetition_features'] for row in rows}, {'binary'})
        self.assertTrue(torch.all(model.last_input[4, :, 96] == 1))

    def test_eval_reuses_dataset_without_changing_validation_samples(self):
        self.build()
        ds = self.dataset(history=8, repetition_features='binary', return_legal_mask=True)
        with patch.object(eval_model, 'PROCESSED_DIR', self.directory):
            reused = eval_model.build_eval_loader(history=8, n=7, repetition_features='binary', dataset=ds)
            with patch.object(eval_model, 'TCECPlies', side_effect=self.dataset):
                separate = eval_model.build_eval_loader(history=8, n=7, repetition_features='binary')
        self.assertIs(reused.dataset.dataset, ds)
        np.testing.assert_array_equal(reused.dataset.indices, separate.dataset.indices)
        reused_batch, separate_batch = next(iter(reused)), next(iter(separate))
        for key in ('board', 'policy_target', 'value_target', 'legal_mask'):
            torch.testing.assert_close(reused_batch[key], separate_batch[key], rtol=0, atol=0)

    def test_eval_rejects_incompatible_shared_dataset(self):
        with patch.object(eval_model, 'PROCESSED_DIR', self.directory):
            with self.assertRaisesRegex(ValueError, 'input settings'):
                eval_model.build_eval_loader(history=8, dataset=self.dataset(history=1))
            with self.assertRaisesRegex(ValueError, 'required evaluation fields'):
                eval_model.build_eval_loader(history=8, dataset=self.dataset(history=8))

    def run_match_cli(self, module, extra_args):
        agents = []
        for mode in ('binary', 'none'):
            agent = Agent.__new__(Agent)
            agent.name, agent.history, agent.repetition_features = f'h8-{mode}', 8, mode
            agent.device, agent.model = torch.device('cpu'), CapturePolicy()
            agents.append(agent)
        argv = ['match', '--ckpt-a', 'rep.pt', '--ckpt-b', 'control.pt',
                '--max-plies', '5', '--no-avoid-repetition', *extra_args]
        with (patch.object(module, 'Agent', side_effect=agents),
              patch.object(module, 'collect_test_openings', return_value=[('cycle', CYCLE)]),
              patch.object(module, 'pick_device', return_value=torch.device('cpu')),
              patch('sys.argv', argv), redirect_stdout(io.StringIO())):
            module.main()

    def test_custom_pairing_pgn_records_features_by_colour(self):
        output = self.directory / 'pgn'
        self.run_match_cli(dump_games, ['--games', '2', '--out', str(output)])
        with (output / 'custom_pairing.pgn').open() as stream:
            games = list(iter(lambda: chess.pgn.read_game(stream), None))
        self.assertEqual([(g.headers['WhiteRepetitionFeatures'], g.headers['BlackRepetitionFeatures'])
                          for g in games], [('binary', 'none'), ('none', 'binary')])

    def test_match_csv_records_features_by_colour(self):
        old_cwd = Path.cwd()
        try:
            os.chdir(self.directory)
            self.run_match_cli(play_match, ['--openings', '1'])
            with Path('match_results.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(r['white_repetition_features'], r['black_repetition_features'])
                              for r in rows], [('binary', 'none'), ('none', 'binary')])
        finally:
            os.chdir(old_cwd)


class RepetitionTrainingTests(unittest.TestCase):
    def test_small_smoke_options_preserve_full_run_defaults(self):
        defaults = train_model.parse_args([])
        self.assertEqual(defaults.batch_size, 2048)
        smoke = train_model.parse_args(['--batch-size', '16', '--log-every', '1',
                                        '--trace-first-step', '--device', 'cpu', '--steps', '3'])
        self.assertEqual((smoke.batch_size, smoke.log_every, smoke.steps), (16, 1, 3))
        self.assertTrue(smoke.trace_first_step)
        self.assertEqual(smoke.device, 'cpu')

    def test_resume_batch_order_matches_uninterrupted_order(self):
        sampler = train_model.SeededBatchSampler(11, 3, 17)
        original = list(sampler)
        sampler.set_step(2)
        self.assertEqual(list(sampler), original[2:])
        sampler.set_step(4)
        following = list(sampler)
        self.assertNotEqual(following, original)
        sampler.set_step(5)
        self.assertEqual(list(sampler), following[1:])

    def test_learning_rate_horizon_is_independent_of_stop_step(self):
        args = train_model.parse_args(['--steps', '60000', '--lr-horizon', '100000'])
        self.assertEqual((args.steps, args.lr_horizon), (60000, 100000))
        self.assertEqual(train_model.lr_for_step(60000, args.lr_horizon), train_model.LR)
        self.assertEqual(train_model.lr_for_step(80000, args.lr_horizon), train_model.MIN_LR1)
        self.assertEqual(train_model.lr_for_step(90000, args.lr_horizon), train_model.MIN_LR2)

    def test_resume_rejects_changed_feature_mode_or_schedule(self):
        for key in ('repetition_features', 'lr_horizon', 'seed', 'plies_sha256', 'batch_size'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Cannot resume'):
                train_model.validate_resume({key: 'old'}, {key: 'new'})

    def test_real_model_training_resume_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'processed'
            source = write_fixture(data)
            build_repetition_flags(source, progress_every=0)
            def dataset(**kwargs):
                return TCECPlies(plies_path=source, offsets_path=data / 'plies.offsets.npy', **kwargs)
            old_cwd, old_threads = Path.cwd(), torch.get_num_threads()
            captured = io.StringIO()
            try:
                os.chdir(root)
                torch.set_num_threads(1)
                with (patch.object(train_model, 'TCECPlies', side_effect=dataset),
                      patch.object(eval_model, 'TCECPlies', side_effect=dataset) as evaluation_factory,
                      patch.object(eval_model, 'PROCESSED_DIR', data),
                      patch.object(train_model, 'TRAIN_IDX_PATH', data / 'train_indices.npy'),
                      patch.object(train_model, 'EVAL_INTERVAL', 2),
                      patch.object(train_model, 'CHECKPOINT_INTERVAL', 2),
                      patch.object(train_model, 'pick_device', return_value=torch.device('cpu')),
                      redirect_stdout(captured)):
                    shared = ['--history', '8', '--repetition-features', 'binary', '--seed', '17',
                              '--lr-horizon', '10', '--workers', '0', '--batch-size', '2',
                              '--log-every', '1', '--trace-first-step']
                    train_model.main(shared + ['--run-name', 'full', '--steps', '4'])
                    train_model.main(shared + ['--run-name', 'resumed', '--steps', '2'])
                    train_model.main(shared + ['--run-name', 'resumed', '--steps', '4', '--resume'])
                    with self.assertRaisesRegex(SystemExit, 'not empty'):
                        train_model.main(shared + ['--run-name', 'full', '--steps', '4'])
                    evaluation_factory.assert_not_called()
                full = torch.load(root / 'checkpoints/full/model_step_4.pt', map_location='cpu', weights_only=False)
                resumed = torch.load(root / 'checkpoints/resumed/model_step_4.pt', map_location='cpu', weights_only=False)
                self.assertEqual(full['step'], 4)
                self.assertEqual(full['batch_size'], 2)
                self.assertEqual(resumed['repetition_features'], 'binary')
                for key, value in full['model_state_dict'].items():
                    torch.testing.assert_close(value, resumed['model_state_dict'][key], rtol=0, atol=0)
                self.assertFalse((root / 'checkpoints/full/model_step_5.pt').exists())
                for message in ('batch loaded; transferring tensors', 'forward pass complete',
                                'backward pass complete', 'optimizer update complete', 'step 1: loss='):
                    self.assertIn(message, captured.getvalue())
            finally:
                os.chdir(old_cwd)
                torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
