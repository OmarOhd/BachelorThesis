import argparse
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request

try:
    chess_version = importlib.metadata.version('chess')
except importlib.metadata.PackageNotFoundError:
    chess_version = None
if chess_version != '1.11.2':
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'chess==1.11.2'], check=True)
    importlib.invalidate_caches()
import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parents[2]
INACCURACY, MISTAKE, BLUNDER, CP_CAP, MATE_CP = 50, 100, 300, 1000, 1000
DEPTH, THREADS, HASH_MB = 16, 4, 512
MATCH_NAME = 'h8_rep_90k_20261114T164524699673Z'
CONFIGS = ('rep_vs_h8', 'rep_vs_h1_filter_off', 'rep_vs_h1_filter_on')
ARCHIVES = {
    'mac_arm64': ('stockfish-macos-m1-apple-silicon.tar',
                  '4d77c4aa3ad9bd1ea8111f2ac5a4620fe7ebf998d6893bf828d49ccd579c8cb0'),
    'mac_x86_64': ('stockfish-macos-x86-64.tar',
                   'e7d7a2bca13915419d41ac6cb8cedb123dd2ba1c39a22c574df7a2aa3f526592'),
    'avx2': ('stockfish-ubuntu-x86-64-avx2.tar',
             '536c0c2c0cf06450df0bfb5e876ef0d3119950703a8f143627f990c7b5417964'),
    'generic': ('stockfish-ubuntu-x86-64.tar',
                '5c6f38b02a4da5f3ffe763f27da6c3e743eebefd92b50cb3661623b96696adff'),
}
FIELDS = ('pairing', 'game', 'ply', 'in_book', 'mover', 'player', 'move',
          'eval_before', 'eval_after', 'cp_loss', 'result', 'opening_game_id',
          'avoid_repetition', 'white', 'black', 'termination')


def score_to_cp(score, pov_color):
    """PovScore -> centipawns from pov_color's perspective, clamped."""
    s = score.pov(pov_color)
    if s.is_mate():
        return MATE_CP if s > chess.engine.Cp(0) else -MATE_CP
    return max(-CP_CAP, min(CP_CAP, s.score()))


def analyse_game(game, engine, limit):
    """Yield one dict per ply: who moved, eval before/after (mover's POV), cp loss."""
    board = game.board()
    opening_plies = int(game.headers.get("OpeningPlies", 0))
    rows = []
    # eval of the starting position (mover = white to move at ply 0)
    info = engine.analyse(board, limit)
    prev_eval_white = score_to_cp(info["score"], chess.WHITE)

    for ply, move in enumerate(game.mainline_moves()):
        mover = board.turn
        eval_before = prev_eval_white if mover == chess.WHITE else -prev_eval_white
        board.push(move)
        info = engine.analyse(board, limit)
        eval_after_white = score_to_cp(info["score"], chess.WHITE)
        eval_after = eval_after_white if mover == chess.WHITE else -eval_after_white
        # centipawn loss: how much the position worsened for the player who moved
        cp_loss = max(0, eval_before - eval_after)
        rows.append({
            "ply": ply,
            "in_book": ply < opening_plies,
            "mover": "white" if mover == chess.WHITE else "black",
            "player": game.headers["White"] if mover == chess.WHITE else game.headers["Black"],
            "move": move.uci(),
            "eval_before": eval_before,
            "eval_after": eval_after,
            "cp_loss": cp_loss,
        })
        prev_eval_white = eval_after_white

    # The recorded match may already have ended in a draw even when an
    # engine search from the final board returns a nonzero continuation score.
    if rows and game.headers.get("Result") == "1/2-1/2":
        outcome = board.outcome(claim_draw=False)
        drawn = outcome is not None and outcome.winner is None
        termination = game.headers.get("Termination")
        drawn = drawn or (termination == "threefold" and board.is_repetition(3))
        drawn = drawn or (termination == "fifty-move" and board.is_fifty_moves())
        if drawn:
            rows[-1]["eval_after"] = 0
            rows[-1]["cp_loss"] = max(0, rows[-1]["eval_before"])
    return rows


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def install_stockfish(cache):
    if platform.system() == 'Darwin' and platform.machine() in ('arm64', 'aarch64'):
        key = 'mac_arm64'
    elif platform.system() == 'Darwin' and platform.machine() == 'x86_64':
        key = 'mac_x86_64'
    elif platform.system() == 'Linux' and platform.machine() in ('x86_64', 'AMD64'):
        cpuinfo = Path('/proc/cpuinfo').read_text().lower()
        key = 'avx2' if 'avx2' in cpuinfo.split() else 'generic'
    else:
        raise RuntimeError('Supply --stockfish /path/to/stockfish18 on this platform.')
    name, expected = ARCHIVES[key]
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / name
    if not archive.is_file() or digest(archive) != expected:
        print('Downloading official Stockfish 18...', flush=True)
        url = 'https://github.com/official-stockfish/Stockfish/releases/download/sf_18/' + name
        partial = archive.with_suffix('.download')
        with urllib.request.urlopen(url, timeout=60) as src, partial.open('wb') as dest:
            shutil.copyfileobj(src, dest)
        if digest(partial) != expected:
            raise RuntimeError('Downloaded Stockfish archive checksum does not match the official release.')
        partial.replace(archive)
    executable_name = name.removesuffix('.tar')
    executable = cache / executable_name
    with tarfile.open(archive) as bundle:
        candidates = [m for m in bundle.getmembers() if m.isfile() and Path(m.name).name == executable_name]
        if len(candidates) != 1:
            raise RuntimeError('Expected exactly one Stockfish executable in the verified archive.')
        with bundle.extractfile(candidates[0]) as src, executable.open('wb') as dest:
            shutil.copyfileobj(src, dest)
    executable.chmod(0o755)
    return executable


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_games(match_dir):
    openings = dict(json.loads((match_dir / 'openings.json').read_text()))
    games = []
    for config in CONFIGS:
        seen = set()
        with (match_dir / (config + '.pgn')).open(encoding='utf-8') as stream:
            number = 0
            while (game := chess.pgn.read_game(stream)) is not None:
                number += 1
                moves = list(game.mainline_moves())
                gid = game.headers['OpeningGameId']
                key = (gid, game.headers['White'])
                if game.errors or key in seen or number > 100:
                    raise ValueError('PGN parsing error, duplicate colour assignment or extra game: ' + config)
                seen.add(key)
                if len(moves) < 8 or int(game.headers['OpeningPlies']) != 8:
                    raise ValueError('Expected eight opening plies.')
                if [m.uci() for m in moves[:8]] != openings[gid]:
                    raise ValueError('Opening differs from the saved match manifest.')
                if game.headers['Result'] not in ('1-0', '0-1', '1/2-1/2'):
                    raise ValueError('Unfinished game in ' + config)
                games.append((config, number, game))
        if number != 100 or len({gid for gid, _ in seen}) != 50:
            raise ValueError('Expected 100 games from 50 opening IDs in ' + config)
    return games


def evaluated_rows(config, number, game, engine):
    rows = analyse_game(game, engine, chess.engine.Limit(depth=DEPTH))
    for row in rows:
        row.update(pairing=config, game=number, result=game.headers['Result'],
                   opening_game_id=game.headers['OpeningGameId'],
                   avoid_repetition=game.headers['AvoidRepetition'],
                   white=game.headers['White'], black=game.headers['Black'],
                   termination=game.headers['Termination'])
    return rows


def cached_or_analyse(path, config, number, game, engine, session):
    signature = hashlib.sha256(str(game).encode()).hexdigest()
    if path.is_file():
        data = json.loads(path.read_text())
        if data['game_sha256'] != signature or len(data['rows']) != len(list(game.mainline_moves())):
            raise ValueError('Saved game analysis does not match its PGN: ' + str(path))
        return data['rows'], True
    rows = evaluated_rows(config, number, game, engine)
    save_json(path, dict(game_sha256=signature, session=session, rows=rows))
    return rows, False


def import_completed(source, out, protocol, games):
    source = source.expanduser().resolve()
    if source == out.resolve():
        raise ValueError('Import source must differ from the output directory.')
    original = json.loads((source / 'protocol.json').read_text())
    keys = ('engine_name', 'depth', 'threads', 'hash_mb', 'chess_version',
            'cp_cap', 'mate_cp', 'thresholds', 'opening_plies', 'pgn_sha256',
            'scoring_sha256', 'total_games', 'engine_search_state')
    for key in keys:
        if original.get(key) != protocol.get(key):
            raise ValueError(f'Colab analysis setting differs: {key}')
    known = {f'{config}_{number:03d}.json': (config, number, game)
             for config, number, game in games}
    files = sorted((source / 'per_game').glob('*.json'))
    prepared = []
    for path in files:
        if path.name not in known:
            raise ValueError('Unrecognized cached game: ' + path.name)
        config, number, game = known[path.name]
        data = json.loads(path.read_text())
        signature = hashlib.sha256(str(game).encode()).hexdigest()
        moves = list(game.mainline_moves())
        if data['game_sha256'] != signature or len(data['rows']) != len(moves):
            raise ValueError('Colab cache does not match its PGN: ' + path.name)
        for ply, (row, move) in enumerate(zip(data['rows'], moves)):
            colour = 'white' if ply % 2 == 0 else 'black'
            expected = dict(pairing=config, game=number, ply=ply, in_book=ply < 8,
                            mover=colour, player=game.headers[colour.capitalize()], move=move.uci(),
                            result=game.headers['Result'], opening_game_id=game.headers['OpeningGameId'],
                            white=game.headers['White'], black=game.headers['Black'],
                            avoid_repetition=game.headers['AvoidRepetition'], termination=game.headers['Termination'])
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError('Colab cache row metadata differs: ' + path.name)
            if not (-CP_CAP <= row['eval_before'] <= CP_CAP and -CP_CAP <= row['eval_after'] <= CP_CAP):
                raise ValueError('Colab evaluation outside scoring caps: ' + path.name)
            if row['cp_loss'] != max(0, row['eval_before'] - row['eval_after']):
                raise ValueError('Colab centipawn-loss formula differs: ' + path.name)
        prepared.append((path, data, signature))
    manifest_path = out / 'imported_games.json'
    expected_manifest = dict(
        source_protocol_sha256=digest(source / 'protocol.json'),
        source_engine_sha256=original['engine_sha256'],
        new_engine_sha256=protocol['engine_sha256'],
        note='Imported games retain the original Colab evaluations. New games use the Mac engine build. Both use Stockfish 18 with identical analysis settings.',
        games={path.name: dict(sha256=digest(path), game_sha256=signature,
                               original_session=data['session']) for path, data, signature in prepared})
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != expected_manifest:
        raise ValueError('Imported-game snapshot differs from the previous run.')
    for path, _, _ in prepared:
        target = out / 'per_game' / path.name
        if target.is_file() and digest(target) != digest(path):
            raise ValueError('Existing analysis differs from import: ' + str(target))
    save_json(manifest_path, expected_manifest)
    for path, _, _ in prepared:
        target = out / 'per_game' / path.name
        if not target.exists():
            temporary = target.with_suffix('.json.tmp')
            shutil.copyfile(path, temporary)
            temporary.replace(target)
    print(f'Imported {len(prepared)} games analysed on Colab', flush=True)
    return len(prepared)


def export_results(out, all_rows):
    positions = out / 'positions.csv'
    with positions.with_suffix('.csv.tmp').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    positions.with_suffix('.csv.tmp').replace(positions)
    groups = defaultdict(list)
    for row in all_rows:
        if not row['in_book']:
            groups[(row['pairing'], row['player'])].append(row['cp_loss'])
    summaries = []
    for (pairing, player), losses in sorted(groups.items()):
        blunders = sum(x >= BLUNDER for x in losses)
        summaries.append(dict(pairing=pairing, player=player, moves=len(losses),
                              acpl=statistics.fmean(losses), median_cp_loss=statistics.median(losses),
                              inaccuracies=sum(INACCURACY <= x < MISTAKE for x in losses),
                              mistakes=sum(MISTAKE <= x < BLUNDER for x in losses),
                              blunders=blunders, blunder_rate_pct=100 * blunders / len(losses)))
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    save_json(out / 'summary.json', dict(positions_in_csv=len(all_rows),
              post_opening_moves=sum(s['moves'] for s in summaries), players=summaries,
              confidence_intervals='Not calculated here; use paired opening-level bootstrap for thesis comparisons.'))
    return summaries


def run(args, out):
    match_dir = args.match_dir.expanduser().resolve()
    games = load_games(match_dir)
    executable = args.stockfish.expanduser().resolve() if args.stockfish else install_stockfish(args.engine_cache)
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    cache = out / 'per_game'
    cache.mkdir(exist_ok=True)
    print(f'Source: {match_dir}\nOutput: {out}\nSettings: Stockfish 18, depth 16, threads 4, hash 512 MB.', flush=True)
    with chess.engine.SimpleEngine.popen_uci(str(executable), timeout=120) as engine:
        if engine.id.get('name') != 'Stockfish 18':
            raise RuntimeError('Expected exactly Stockfish 18; got ' + str(engine.id))
        engine.configure({'Threads': THREADS, 'Hash': HASH_MB})
        protocol = dict(engine_name=engine.id['name'], engine_sha256=digest(executable),
                        depth=DEPTH, threads=THREADS, hash_mb=HASH_MB,
                        chess_version=chess.__version__, cp_cap=CP_CAP, mate_cp=MATE_CP,
                        thresholds=[INACCURACY, MISTAKE, BLUNDER], opening_plies=8,
                        pgn_sha256={config: digest(match_dir / (config + '.pgn')) for config in CONFIGS},
                        scoring_sha256=hashlib.sha256((inspect.getsource(score_to_cp) + inspect.getsource(analyse_game)).encode()).hexdigest(),
                        total_games=len(games), engine_search_state='Preserved within a process; fresh on resume. Completed evaluations are retained.',
                        imported_evaluations_manifest='imported_games.json',
                        engine_sha256_scope='New evaluations only; imported evaluation origins are listed separately.')
        protocol_file = out / 'protocol.json'
        if protocol_file.is_file():
            if json.loads(protocol_file.read_text()) != protocol:
                raise RuntimeError('Analysis settings, engine or PGNs differ from the saved run. Use a new --out directory.')
        else:
            save_json(protocol_file, protocol)
        if args.import_analysis:
            import_completed(args.import_analysis, out, protocol, games)
        all_rows, reused, start = [], 0, time.monotonic()
        for done, (config, number, game) in enumerate(games, start=1):
            game_file = cache / f'{config}_{number:03d}.json'
            rows, was_cached = cached_or_analyse(game_file, config, number, game, engine, session)
            reused += int(was_cached)
            all_rows.extend(rows)
            save_json(out / 'progress.json', dict(status='running', completed_games=done,
                      total_games=len(games), reused_games=reused, session=session,
                      last_configuration=config, last_game=number, elapsed_seconds=time.monotonic() - start))
            print(f'[{done:3d}/{len(games)}] {config} game {number:3d}: {len(rows)} plies; '
                  f'{"reused saved analysis" if was_cached else "analysed and saved"}', flush=True)
    summaries = export_results(out, all_rows)
    save_json(out / 'progress.json', dict(status='completed', completed_games=len(games),
              total_games=len(games), reused_games=reused, positions=len(all_rows), session=session,
              elapsed_seconds=time.monotonic() - start))
    for row in summaries:
        print(f"{row['pairing']} | {row['player']} | moves={row['moves']} "
              f"ACPL={row['acpl']:.2f} blunders={row['blunders']} ({row['blunder_rate_pct']:.2f}%)", flush=True)
    print(f'\nDone. {len(games)} games, {len(all_rows)} positions. Results in {out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--match-dir', type=Path, default=ROOT / 'results/matches/h8_rep')
    ap.add_argument('--stockfish', type=Path, help='Optional existing Stockfish 18 executable')
    ap.add_argument('--engine-cache', type=Path, default=Path.home() / '.cache/stockfish18')
    ap.add_argument('--import-analysis', type=Path, help='Earlier analysis folder whose games are reused')
    ap.add_argument('--out', type=Path, default=Path('analysis_h8_rep_d16'))
    args = ap.parse_args()
    if not args.match_dir.is_dir():
        ap.error(f'Match folder not found: {args.match_dir}')
    out = args.out or args.match_dir / 'stockfish_d16'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'console.log').open('a', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args, out)
            except BaseException:
                import traceback
                traceback.print_exc()
                print('Finished games are saved. Rerun the same command to continue.', flush=True)
                raise


if __name__ == '__main__':
    main()
