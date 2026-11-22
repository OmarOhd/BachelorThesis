import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OPENINGS_SHA256 = '91dc4f1442acd17c7e3a23cd7cfbd457beec3e131dd9ecbc6306685e5f2efccc'
FIELDS = ('configuration', 'game', 'opening_game_id', 'white', 'black',
          'result', 'plies', 'termination', 'white_post_opening_moves',
          'black_post_opening_moves', 'white_interventions', 'black_interventions',
          'h1_filter_enabled')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def opening_digest(openings):
    items = sorted((str(gid), list(moves)) for gid, moves in openings)
    return hashlib.sha256(json.dumps(items, separators=(',', ':')).encode()).hexdigest()


def resolve_baseline(explicit, run, step, project, drive):
    if explicit is not None:
        return explicit.expanduser().resolve()
    relative = Path(run) / f'model_step_{step}.pt'
    roots = (project / 'checkpoints', drive / 'BachelorThesis/baseline_checkpoints',
             drive / 'BachelorThesis/checkpoints',
             drive / 'BachelorThesis/colab_runs/checkpoints', drive / 'checkpoints')
    return next((p / relative for p in roots if (p / relative).is_file()), roots[1] / relative)


def summarize(rows):
    if not rows:
        return {'games': 0, 'terminations': {}, 'threefold_games': 0,
                'threefold_rate_percent': None, 'players': {}}
    terminations = Counter(row['termination'] for row in rows)
    players = {}
    for row in rows:
        for colour in ('white', 'black'):
            name = row[colour]
            record = players.setdefault(name, dict(wins=0, draws=0, losses=0,
                                                   post_opening_moves=0, interventions=0,
                                                   games_with_intervention=0))
            result = row['result']
            if result == '1/2-1/2':
                record['draws'] += 1
            elif (colour == 'white' and result == '1-0') or (colour == 'black' and result == '0-1'):
                record['wins'] += 1
            elif result in ('1-0', '0-1'):
                record['losses'] += 1
            else:
                raise ValueError(f'Unfinished/unknown result: {result}')
            record['post_opening_moves'] += row[colour + '_post_opening_moves']
            interventions = row[colour + '_interventions']
            record['interventions'] += interventions
            record['games_with_intervention'] += int(interventions > 0)
    for record in players.values():
        n = record['wins'] + record['draws'] + record['losses']
        record['games'] = n
        record['score_percent'] = 100 * (record['wins'] + record['draws'] / 2) / n
        record['intervention_rate_percent'] = (100 * record['interventions'] / record['post_opening_moves']
                                               if record['post_opening_moves'] else None)
    threefold = terminations.get('threefold', 0)
    return dict(games=len(rows), terminations=dict(terminations), threefold_games=threefold,
                threefold_rate_percent=100 * threefold / len(rows), players=players)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=ROOT)
    ap.add_argument('--data-root', type=Path, default=Path('/content/data'))
    ap.add_argument('--drive-root', type=Path, default=Path('/content/drive/MyDrive'))
    ap.add_argument('--h1', type=Path, help='Original h1_masked model_step_60000.pt')
    ap.add_argument('--h8', type=Path, help='Original h8_masked model_step_30000.pt')
    ap.add_argument('--rep', type=Path, help='h8+rep model_step_90000.pt')
    args = ap.parse_args()
    project, data, drive = (p.expanduser().resolve() for p in (args.project, args.data_root, args.drive_root))
    paths = {
        'h8+rep@90k': args.rep or drive / 'h8_rep/checkpoints/h8_rep/model_step_90000.pt',
        'h1@60k': resolve_baseline(args.h1, 'h1_masked', 60000, project, drive),
        'h8@30k': resolve_baseline(args.h8, 'h8_masked', 30000, project, drive),
    }
    needed = [*paths.values(), project / 'scripts/match/common.py',
              project / 'scripts/match/dump_games.py', data / 'tcec/processed/plies.jsonl']
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise SystemExit('Missing files:\n' + '\n'.join(missing) +
                         '\nCheck the paths or pass --h1, --h8 and --rep.')
    os.environ['DATA_ROOT'] = str(data)
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
    sys.path.insert(0, str(project))
    import torch
    import chess
    from scripts.match import common
    from scripts.match.dump_games import board_to_pgn

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = common.pick_device()
    print(f'Device: {device}', flush=True)
    print('Loading the 50 test openings...', flush=True)
    openings = common.collect_test_openings(50)
    if len(openings) != 50 or opening_digest(openings) != OPENINGS_SHA256:
        raise SystemExit('The openings differ from those of the original matches.')
    if any(len(moves) != 8 for _, moves in openings):
        raise SystemExit('Expected eight opening plies.')
    expected = {'h8+rep@90k': (8, 'binary', 90000), 'h1@60k': (1, 'none', 60000), 'h8@30k': (8, 'none', 30000)}
    agents = {}
    for name, path in paths.items():
        agent = common.Agent(path, device)
        if (agent.history, agent.repetition_features, int(agent.step)) != expected[name]:
            raise SystemExit(f'Unexpected checkpoint metadata for {name}: {path}')
        agents[name] = agent
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = drive / 'h8_rep/matches' / ('h8_rep_90k_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    configs = [('rep_vs_h8', 'h8@30k', False), ('rep_vs_h1_filter_off', 'h1@60k', False),
               ('rep_vs_h1_filter_on', 'h1@60k', True)]
    manifest = dict(checkpoints={name: dict(path=str(path), sha256=digest(path),
                                          pgn_name=agents[name].name) for name, path in paths.items()},
                    source_sha256={rel: digest(project / rel) for rel in
                                   ('model/main.py', 'model/encoder.py', 'model/repetition_features.py',
                                    'scripts/match/common.py', 'scripts/match/repetition.py', 'scripts/match/dump_games.py')},
                    torch_version=torch.__version__, device=str(device),
                    gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                    seed=0, opening_plies=8, max_plies=300, games_per_configuration=100,
                    openings_sha256=OPENINGS_SHA256, configurations=configs,
                    inference='FP32; AMP and TF32 disabled',
                    termination_note='Threefold is an outcome of both players; h8+rep has no external filter.')
    save_json(out / 'manifest.json', manifest)
    save_json(out / 'openings.json', [(str(g), m) for g, m in openings])
    print(f'Output: {out}', flush=True)
    summaries = {}
    for config, opponent, filtering in configs:
        rows = []
        print(f'\n{config}: h8+rep@90k vs {opponent}, h1 filter {filtering}', flush=True)
        with (out / (config + '.pgn')).open('w', encoding='utf-8') as pgn, \
             (out / (config + '.csv')).open('w', newline='', encoding='utf-8') as csv_fh:
            writer = csv.DictWriter(csv_fh, fieldnames=FIELDS)
            writer.writeheader()
            for gid, moves in openings:
                for white_name, black_name in (('h8+rep@90k', opponent), (opponent, 'h8+rep@90k')):
                    white, black = agents[white_name], agents[black_name]
                    board, result, reason, interventions = common.play_game(white, black, moves, 300, filtering)
                    n = len(rows) + 1
                    game = board_to_pgn(board, white, black, result, reason, 8,
                                        config, n, gid, filtering, interventions)
                    pgn.write(str(game) + '\n\n')
                    pgn.flush()
                    white_moves = (board.ply() + 1) // 2 - 4
                    black_moves = board.ply() // 2 - 4
                    row = dict(configuration=config, game=n, opening_game_id=str(gid),
                               white=white_name, black=black_name, result=result, plies=board.ply(),
                               termination=reason, white_post_opening_moves=white_moves,
                               black_post_opening_moves=black_moves, white_interventions=interventions[chess.WHITE],
                               black_interventions=interventions[chess.BLACK], h1_filter_enabled=filtering)
                    writer.writerow(row)
                    csv_fh.flush()
                    rows.append(row)
                    summaries[config] = summarize(rows)
                    save_json(out / 'summary.json', summaries)
                    print(f'{config} game {n:3d}/100: {white_name} (W) vs {black_name} (B) '
                          f'{result}; {board.ply()} plies; {reason}; '
                          f'interventions W/B={interventions[chess.WHITE]}/{interventions[chess.BLACK]}', flush=True)
        s = summaries[config]
        rep = s['players']['h8+rep@90k']
        print(f"h8+rep: +{rep['wins']} ={rep['draws']} -{rep['losses']}; "
              f"score={rep['score_percent']:.1f}%; threefold={s['threefold_games']}/100", flush=True)
    save_json(out / 'status.json', dict(status='completed', completed_games=300))
    print(f'\nDone. Results in {out}', flush=True)


if __name__ == '__main__':
    main()
