"""Run a queue of `1016_oetjen_d3_decipher.py` jobs from a frozen copy of HEAD.

    .venv/bin/python "Real Data/Oetjen Bone Marrow/1016_oetjen_d3_queue.py" \
        <job file> <N parallel> <tag> <scratch dir>

The job file has one line of driver args per job (blank lines and `#` lines skipped), e.g.
`--preset model5 --seed 0 --lambda-c 1 --scan`.
The helper does `git worktree add --no-checkout --detach <scratch>/runtree_<tag> HEAD` of the main
checkout (this file's repo) and checks out only `RUNTREE_PATHS` (`decipher_models3/`, this
folder's `*.py` and the BMMC `1008_bmmc_build_shared.py` the NUT builder loads, which is all the
drivers import; the full tree is 4.5 GB and does not fit on the VM's disk). It runs every job N at a time with 1 thread each as

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=<runtree> \
        <repo>/.venv/bin/python "<runtree>/Real Data/Oetjen Bone Marrow/1016_oetjen_d3_decipher.py" \
        <args> --src <abs src> --out-dir <abs out-dir>

so code edits in the main checkout during the queue do not reach running jobs. `--src` and
`--out-dir` resolve against the main checkout: a job's own value is taken relative to the repo
root, else the driver's defaults (`1016_oetjen5/oetjen5_shared.h5ad` and the `run_names` out-dir)
are taken under the main checkout's `Real Data/Oetjen Bone Marrow/`. The h5ad is gitignored, so
it only exists there; the `.sha256` beside it is read from there too. Each job logs to
`<out-dir>/logs/seed<k>[_<λ tag>].log`. The worktree is removed at the end, also on failure or
SIGTERM (running jobs are terminated first). Exit code 1 if any job failed.

`--driver <repo-relative path>` runs another driver with the same interface (`build_parser`,
`lambda_overrides`, `run_names`, `SRC`, `HERE`), e.g.
`"Real Data/Spatial NUT Carcinoma Data/1018_nut_d3_decipher.py"`; its folder's `*.py` join the
runtree, and its `SRC` and out-dirs resolve under that folder of the main checkout. Default: the
Oetjen d3 driver.
"""

import argparse
import functools
import importlib.util
import os
import shlex
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
SCRIPT_REL = Path("Real Data") / "Oetjen Bone Marrow" / "1016_oetjen_d3_decipher.py"
RUNTREE_PATHS = [
    "decipher_models3",
    "Real Data/Oetjen Bone Marrow/*.py",
    # loaded at import by the NUT builder, which the NUT drivers import
    "Real Data/Healthy Human Bone Marrow Mononuclear Cells/1008_bmmc_build_shared.py",
]


@functools.cache
def load_driver(repo_rel: Path) -> ModuleType:
    """The driver module at `repo_rel` in the main checkout (loaded once per path)."""
    spec = importlib.util.spec_from_file_location(f"queue_driver_{repo_rel.stem}", REPO / repo_rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtree_paths(driver_rel: Path = SCRIPT_REL) -> list[str]:
    """`RUNTREE_PATHS` plus the driver folder's `*.py`, without duplicates (git pathspecs)."""
    return list(dict.fromkeys([*RUNTREE_PATHS, f"{driver_rel.parent.as_posix()}/*.py"]))


def read_jobs(text: str) -> list[list[str]]:
    """Driver arg lists, one per non-blank, non-`#` line."""
    lines = [ln.strip() for ln in text.splitlines()]
    return [shlex.split(ln) for ln in lines if ln and not ln.startswith("#")]


def job_command(
    job: list[str], runtree: Path, repo: Path = REPO, driver_rel: Path = SCRIPT_REL
) -> tuple[list[str], Path]:
    """The command running `job` with the driver `driver_rel` from `runtree`, src and out-dir in
    `repo`, and its log path."""
    d3 = load_driver(driver_rel)
    args = d3.build_parser().parse_args(job)
    lambdas = d3.lambda_overrides(args)
    root = repo / driver_rel.parent
    names = d3.run_names(
        args.dataset,
        args.preset,
        args.seed,
        lambdas,
        args.scan,
        args.final,
        root=root,
        tagged_final=args.tagged_final,
    )
    src = repo / args.src if args.src != d3.SRC else root / d3.SRC.relative_to(d3.HERE)
    out = repo / args.out_dir if args.out_dir is not None else names["out_dir"]
    tag = names["lambda_tag"]
    log = out / "logs" / (f"seed{args.seed}_{tag}.log" if tag else f"seed{args.seed}.log")
    # argparse keeps the last value of a repeated option, so these override the job's own
    cmd = [str(repo / ".venv" / "bin" / "python"), str(runtree / driver_rel), *job]
    return [*cmd, "--src", str(src), "--out-dir", str(out)], log


def run_queue(commands: list[tuple[list[str], Path]], n_parallel: int, runtree: Path) -> list[int]:
    """Run each (cmd, log) N at a time; return the exit codes (-1 = not started)."""
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "PYTHONPATH": str(runtree),
    }
    procs: list[subprocess.Popen] = []
    stop = threading.Event()

    def run(job: tuple[list[str], Path]) -> int:
        cmd, log = job
        if stop.is_set():
            return -1
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"start {shlex.join(cmd[2:])} -> {log}", flush=True)
        with log.open("w") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
            procs.append(proc)
            code = proc.wait()
        print(f"exit {code}: {log}", flush=True)
        return code

    pool = ThreadPoolExecutor(n_parallel)
    try:
        return list(pool.map(run, commands))
    finally:
        stop.set()
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        pool.shutdown(wait=True, cancel_futures=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jobs", type=Path, help="one line of driver args per job")
    parser.add_argument("n_parallel", type=int)
    parser.add_argument("tag", help="names the worktree runtree_<tag>")
    parser.add_argument("scratch", type=Path, help="folder for the worktree")
    parser.add_argument(
        "--driver",
        type=Path,
        default=SCRIPT_REL,
        help=f"repo-relative driver script (default: {SCRIPT_REL})",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    runtree = args.scratch.resolve() / f"runtree_{args.tag}"
    load_driver(args.driver)
    jobs = read_jobs(args.jobs.read_text())
    commands = [job_command(job, runtree, driver_rel=args.driver) for job in jobs]
    if not commands:
        raise ValueError(f"no jobs in {args.jobs}")
    git = ["git", "-C", str(REPO), "worktree"]
    subprocess.run([*git, "add", "--no-checkout", "--detach", str(runtree), "HEAD"], check=True)
    try:
        checkout = [
            "git",
            "-C",
            str(runtree),
            "checkout",
            "HEAD",
            "--",
            *runtree_paths(args.driver),
        ]
        subprocess.run(checkout, check=True)
        codes = run_queue(commands, args.n_parallel, runtree)
    finally:
        subprocess.run([*git, "remove", "--force", str(runtree)], check=True)
    failed = [log for (_, log), code in zip(commands, codes) if code != 0]
    print(f"{len(codes) - len(failed)}/{len(codes)} jobs ok; failed: {[str(f) for f in failed]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
