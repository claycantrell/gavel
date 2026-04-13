#!/usr/bin/env python3
import sys, os, importlib, random
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/Users/a12066/Documents/GitHub/ludax/src')
import ludax.ludii_transpiler as lt; importlib.reload(lt)
from ludax import LudaxEnvironment
import jax
from collections import Counter

errors = Counter()
ok = 0
total = 0
game_files = []
for root, dirs, files in os.walk('ludii_data/games/expanded'):
    for f in files:
        if f.endswith('.lud'):
            game_files.append(os.path.join(root, f))

random.seed(42)
random.shuffle(game_files)

for path in game_files[:100]:
    total += 1
    name = os.path.basename(path)
    with open(path) as fh:
        lud = fh.read().strip()
    t = lt.LudiiTranspiler()
    ldx = t.transpile(lud)
    if not ldx:
        errors['transpile_fail'] += 1
        print(f'  TRANSPILE FAIL: {name}', flush=True)
        continue
    try:
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("JAX compilation timeout")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout per game
        env = LudaxEnvironment(game_str=ldx.replace('\n', ' '))
        state = env.init(jax.random.PRNGKey(0))
        signal.alarm(0)
        if state.legal_action_mask.sum() > 0:
            ok += 1
            print(f'  OK: {name}', flush=True)
        else:
            errors['no_legal_moves'] += 1
            print(f'  NO MOVES: {name}', flush=True)
    except Exception as e:
        err = str(e)
        if 'No terminal matches' in err:
            errors['ludax_grammar'] += 1
        elif 'pop from empty' in err:
            errors['piece_rendering'] += 1
        elif 'switch branches' in err:
            errors['phase_mismatch'] += 1
        else:
            errors[str(e).split('\n')[0][:40]] += 1
        signal.alarm(0)
        print(f'  FAIL: {name} — {str(e).split(chr(10))[0][:50]}', flush=True)

print(f'\n{ok}/{total} end-to-end ({ok/total*100:.0f}%)', flush=True)
for err, c in errors.most_common():
    print(f'  {c:3d}  {err}', flush=True)
