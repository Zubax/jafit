# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>
# type: ignore

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
    "*.png",
]

nox.options.error_on_external_run = True


@nox.session(python=False)
def clean(session):
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
def test(session):
    session.install("-e", ".")
    session.install("pytest ~= 8.3", "mypy ~= 1.14", "coverage ~= 7.6")

    session.run("coverage", "run", "-m", "jafit", "data/bh-lng37-ansys.tab", "effort=100")
    session.run("coverage", "run", "-m", "jafit", "c_r=0.1", "M_s=1e6", "a=560", "k_p=1200", "alpha=0.0007")

    session.run("coverage", "run", "-m", "pytest", env={"NUMBA_DISABLE_JIT": "1"})

    session.run("coverage", "combine")
    session.run("coverage", "report", "--fail-under=25")
    if session.interactive:
        session.run("coverage", "html")
        report_file = Path.cwd().resolve() / "htmlcov" / "index.html"
        session.log(f"OPEN IN WEB BROWSER: file://{report_file}")

    session.install("mypy ~= 1.14")
    session.run("mypy", ".")


@nox.session(reuse_venv=True)
def black(session):
    session.install("black ~= 24.10")
    session.run("black", "--check", ".")
