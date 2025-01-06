# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>
# type: ignore

import shutil
from pathlib import Path
# noinspection PyPackageRequirements
import nox


nox.options.error_on_external_run = True


@nox.session(python=False)
def clean(session):
    for w in [
        "*.egg-info",
        ".coverage*",
        "html*",
        ".*cache",
        "__pycache__",
        ".*compiled",
        "*.log",
        "*.tmp",
    ]:
        for f in Path.cwd().glob(w):
            session.log(f"Removing: {f}")
            if f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink(missing_ok=True)


@nox.session(reuse_venv=True)
def test(session):
    session.install("-e", ".")
    session.install("-r", "requirements.txt")
    session.install("pytest ~= 8.3", "mypy ~= 1.14", "coverage ~= 7.6")

    # PyTest
    session.run("coverage", "run", "-m", "pytest")

    # Coverage report
    session.run("coverage", "report", "--fail-under=25")
    if session.interactive:
        session.run("coverage", "html")
        report_file = Path.cwd().resolve() / "htmlcov" / "index.html"
        session.log(f"OPEN IN WEB BROWSER: file://{report_file}")

    # Static analysis
    session.run("mypy", ".")


@nox.session(reuse_venv=True)
def black(session):
    session.install("black ~= 24.10")
    session.run("black", "--check", ".")
