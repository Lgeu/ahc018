import sys
from math import log
from pathlib import Path
from datetime import datetime
from threading import Thread
from subprocess import Popen, PIPE
from concurrent.futures import ThreadPoolExecutor, as_completed

N = 1000

scores = [0] * N


def read_stream(name, in_file, out_file):
    for line in in_file:
        print(f"[{name}] {line.strip()}", file=out_file)
        try:
            scores[name] = log(int(line.strip().split()[-1]))
        except:
            pass


def run(cmd, name, timeout=None):
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    stdout_thread = Thread(target=read_stream, args=(name, proc.stdout, sys.stdout))
    stderr_thread = Thread(target=read_stream, args=(name, proc.stderr, sys.stderr))
    stdout_thread.start()
    stderr_thread.start()
    try:
        proc.wait(timeout=timeout)
    except TimeoutError:
        pass
    return proc


out_dir = Path("out") / datetime.now().isoformat()
out_dir.mkdir()

MULTITHREAD = False
TIME_LIMIT = 10.0

if MULTITHREAD:
    with ThreadPoolExecutor(1) as executor:
        futures = []
        for i in range(N):
            out_file = out_dir / f"{i:04d}.txt"
            cmd = f"./tools/target/release/tester python main.py < ./tools/in/{i:04d}.txt > {out_file}"
            futures.append(executor.submit(run, cmd, i, TIME_LIMIT))
        as_completed(futures)
else:
    for i in range(N):
        out_file = out_dir / f"{i:04d}.txt"
        # cmd = f"./tools/target/release/tester python main.py < ./tools/in/{i:04d}.txt > {out_file}"
        cmd = f"./tools/target/release/tester python submission.py < ./tools/in/{i:04d}.txt > {out_file}"
        run(cmd, i, TIME_LIMIT)

print(f"Mean Score = {sum(scores) / len(scores)}")
