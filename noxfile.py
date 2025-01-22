# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>
# type: ignore

import shlex
import shutil
from pathlib import Path

# noinspection PyPackageRequirements
import nox


BYPRODUCTS = [
    "*.egg-info",
    "src/*.egg-info",
    ".coverage*",
    "html*",
    ".*cache",
    "__pycache__",
    ".*compiled",
    "*.log",
    "*.tmp",
    "*.jafit.png",
    "*.jafit.tab",
]

nox.options.error_on_external_run = True


@nox.session(python=False)
def clean(session: nox.Session) -> None:
    for w in BYPRODUCTS:
        for f in Path.cwd().glob(w):
            try:
                session.log(f"Removing: {f}")
                if f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    f.unlink(missing_ok=True)
            except Exception as ex:
                session.error(f"Failed to remove {f}: {ex}")


@nox.session(reuse_venv=True)
def mypy(session: nox.Session) -> None:
    session.install("-e", ".")
    session.install("mypy ~= 1.14")
    session.run("mypy", ".")


@nox.session(reuse_venv=True)
def black(session: nox.Session) -> None:
    session.install("black ~= 24.10")
    session.run("black", "--check", ".")


@nox.session(reuse_venv=True)
def test(session: nox.Session) -> None:
    session.install("-e", ".")
    session.install("coverage ~= 7.6")

    # Run the tool with coverage
    def run(args: str) -> None:
        session.run("coverage", "run", "-m", "jafit", *shlex.split(args))

    run("model=ve c_r=0.1       M_s=1.6e6       a=560           k_p=1200         alpha=0.0007")
    run("model=po c_r=0.1       M_s=1.6e6       a=560           k_p=1200         alpha=0.0007")
    run("model=sz c_r=0.1       M_s=1.6e6       a=560           k_p=1200         alpha=0.0007")
    run("model=ve c_r=0.2107788 M_s=1306755.22  a=108.694943    k_p=177.625645   alpha=0.000294224757 H_amp_max=1111")
    run("model=or c_r=0.885     M_s=1080000     a=1107718.3824  k_p=702271.17275 alpha=3.168")
    run("model=po c_r=0.956886  M_s=2956870.912 a=025069.875361 k_p=019498.2     alpha=0.18122")

    run("model=venk effort=20 plot_failed=1 fast=1 ref='data/B(H).AlNiCo_5.tab'")
    run("model=venk effort=20 plot_failed=1 fast=1 ref='data/B(H).LNG37.ansys.tab'")
    run("model=venk effort=20 plot_failed=1 fast=1 ref='data/Altair_Flux_HystereticExample.csv'")

    # Run pytest with coverage
    session.install("pytest ~= 8.3")
    session.run("coverage", "run", "-m", "pytest", env={"NUMBA_DISABLE_JIT": "1"})

    # Generate coverage report
    session.run("coverage", "combine")
    session.run("coverage", "report", "--fail-under=25")
    if session.interactive:
        session.run("coverage", "html")
        report_file = Path.cwd().resolve() / "htmlcov" / "index.html"
        session.log(f"OPEN IN WEB BROWSER: file://{report_file}")
