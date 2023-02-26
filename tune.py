import optuna

params = dict(
    BASE_P_COEF=(1.5, 6.0),
    RECOVERY_P_COEF=(1.5, 6.0),
    STOP_SIGMA=(-0.5, 1.5),
    TEMPORAL_PREDICTION_COEF=(0.0, 3.0),
    MU_START=(500.0, 2500.0),
    MU_END=(1500.0, 4000.0),
    SIGMA=(500.0, 2000.0),
    NOISE_BASE_P_RATIO=(0.5, 2.0),
    SIGMA_RBF=(5.0, 20.0),
    N_COLS=(10, 25),
    LEFT_TAIL_COEF=(0.5, 4.0),
    PRIORITY_COEF_STURDINESS_STD=(-1.0, 1.0),
    PRIORITY_COEF_SYSTEM_SIZE=(0.5, 4.0),
    PRIORITY_COEF_SYSTEM_SIZE_K=(0.5, 10.0),
    COEF_INITIAL_P_STD=(0.5, 4.0),
    MIN_STEINER_MU=(1000.0, 3000.0),
)

# for name in params.keys():
#     print(f'{name} = params["{name}"]')

import sys
from math import log
from pathlib import Path
from datetime import datetime
from threading import Thread
from subprocess import Popen, PIPE
from concurrent.futures import ThreadPoolExecutor, as_completed

N = 100


def read_stream(name, in_file, out_file, scores=None):
    for line in in_file:
        print(f"[{name}] {line.strip()}", file=out_file)
        try:
            scores[name] = log(int(line.strip().split()[-1]))
        except:
            pass


def run(cmd, name, scores):
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    stdout_thread = Thread(target=read_stream, args=(name, proc.stdout, sys.stdout))
    stderr_thread = Thread(
        target=read_stream, args=(name, proc.stderr, sys.stderr, scores)
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        proc.wait(timeout=10.0)
    except:
        proc.kill()
    return proc


def objective(trial):
    scores = [100.0] * N
    options = []
    for name, (mi, ma) in params.items():
        if isinstance(mi, int):
            p = trial.suggest_int(name, mi, ma)
        else:
            p = trial.suggest_float(name, mi, ma)
        options.append(f"{name}={p}")
    options = " ".join(options)

    with ThreadPoolExecutor(4) as executor:
        futures = []
        for i in range(N):
            cmd = f"./tools/target/release/tester python main.py {options} < ./tools/in/{i:04d}.txt > /dev/null"
            futures.append(executor.submit(run, cmd, i, scores))
        as_completed(futures)

    mean_score = sum(scores) / len(scores)
    print(f"Mean Score = {mean_score}")
    return mean_score


def callback(study, trial):
    if study.best_value == trial.value:
        print(f"Updated! {study.best_value}")
        with open(f"{trial.value:.5f}.txt", "w") as f:
            f.write(f"{study.best_trial}")


storage_path = f"test_study.db"
storage = f"sqlite:///{storage_path}"
study_name = "study"
study = optuna.create_study(storage=storage, load_if_exists=True, study_name=study_name)
study.optimize(objective, n_trials=10, timeout=86400, callbacks=[callback])
