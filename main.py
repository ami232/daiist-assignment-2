"""
Single entrypoint for Assignment 2. Run:

    uv run python main.py train    # runs train.py, or train.ipynb if you used a notebook
    uv run python main.py app      # runs app.py, or app.ipynb if you used a notebook

Whichever format you used (script or notebook) for a stage, this looks for
`<stage>.py` first, then `<stage>.ipynb`, and runs it the same way every time:
a notebook is first converted to a plain script (`jupyter nbconvert --to
script`) and then executed exactly like a .py file would be. This matters for
`app`: your Gradio app calls `demo.launch()`, which blocks and opens a
browser tab. Running it as a real script (not inside a Jupyter kernel) is
what makes that work whether you wrote app.py or app.ipynb.

Do not rename this file or its two subcommands. Grading runs exactly these
two commands against your submission.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_entrypoint(name: str) -> None:
    py_path = ROOT / f"{name}.py"
    nb_path = ROOT / f"{name}.ipynb"

    if py_path.exists() and nb_path.exists():
        raise SystemExit(
            f"Found both {py_path.name} and {nb_path.name}. main.py won't guess which "
            f"one you want. Delete whichever one you're not using, then try again."
        )

    if py_path.exists():
        subprocess.run([sys.executable, str(py_path)], cwd=ROOT, check=True)
        return

    if nb_path.exists():
        with tempfile.NamedTemporaryFile(
            suffix=".py", dir=ROOT, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with tmp_path.open("w") as tmp_file:
                subprocess.run(
                    ["jupyter", "nbconvert", "--to", "script", "--stdout", str(nb_path)],
                    cwd=ROOT, check=True, stdout=tmp_file,
                )
            subprocess.run([sys.executable, str(tmp_path)], cwd=ROOT, check=True)
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    raise FileNotFoundError(
        f"Expected {py_path.name} or {nb_path.name} in {ROOT}. Create one of them."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "app"])
    args = parser.parse_args()
    run_entrypoint(args.stage)
