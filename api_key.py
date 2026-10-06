"""
Assignment 2: find and verify your Roboflow API key.

Run it on its own to check your setup before you write any training code:

    uv run python api_key.py

It looks for your key in two places, in this order:

1. The ROBOFLOW_API_KEY environment variable, if you set one.
2. A .env file in this folder. Copy .env.example to .env and paste your
   key in. This is the easy route: you set it once and forget about it.

.env is in .gitignore and must stay there. Your key is a credential: it is
not yours to leak, and a key committed to a public fork is a key you have to go
and revoke.

Your training code should get the key from here rather than reading the
environment directly, so that both routes work:

    from api_key import load_api_key
    rf = Roboflow(api_key=load_api_key())
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
VAR = "ROBOFLOW_API_KEY"
VERIFY_URL = "https://api.roboflow.com/?api_key={key}"


def read_env_file(path: Path = ENV_FILE) -> str | None:
    """Pull ROBOFLOW_API_KEY out of a .env file, if there is one.

    Deliberately a tiny parser rather than a dependency: KEY=value per line,
    blank lines and # comments ignored, surrounding quotes stripped.
    """
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator and name.strip() == VAR:
            return value.strip().strip('"').strip("'") or None
    return None


def resolve_api_key() -> tuple[str | None, str]:
    """Return (key, where it came from), without changing anything."""
    from_env = (os.environ.get(VAR) or "").strip()
    if from_env:
        return from_env, f"the {VAR} environment variable"
    from_file = read_env_file()
    if from_file:
        return from_file, ENV_FILE.name
    return None, "nowhere"


def load_api_key(required: bool = True) -> str | None:
    """Return the API key from the environment, else from .env.

    Also puts it into os.environ so libraries that look it up themselves
    (the roboflow SDK does) find it too.
    """
    key, _ = resolve_api_key()
    if key:
        os.environ[VAR] = key
        return key
    if required:
        raise RuntimeError(
            f"No {VAR} found.\n"
            f"  Copy .env.example to .env and paste your key in, or set the\n"
            f"  {VAR} environment variable. Get your key from the Roboflow\n"
            f"  dashboard under Account > Roboflow Keys."
        )
    return None


def masked(key: str) -> str:
    return f"{key[:2]}{'*' * max(len(key) - 4, 4)}{key[-2:]}" if len(key) > 4 else "****"


def verify(key: str, timeout: int = 20) -> tuple[bool, str]:
    """Ask Roboflow whether this key actually works."""
    try:
        with urllib.request.urlopen(VERIFY_URL.format(key=key), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8"))["error"]["message"]
        except Exception:
            detail = f"HTTP {error.code}"
        return False, detail
    except urllib.error.URLError as error:
        return False, f"could not reach api.roboflow.com ({error.reason}). Are you online?"
    except Exception as error:  # noqa: BLE001 - surface anything else plainly
        return False, f"{type(error).__name__}: {error}"

    workspace = payload.get("workspace")
    if isinstance(workspace, dict):
        workspace = workspace.get("name") or workspace.get("url")
    return True, str(workspace) if workspace else "key accepted"


def main() -> int:
    key, source = resolve_api_key()
    if not key:
        print(
            f"No API key found.\n\n"
            f"  Copy .env.example to .env and paste your key in:\n\n"
            f"      cp .env.example .env\n\n"
            f"  Get your key from the Roboflow dashboard, under\n"
            f"  Account > Roboflow Keys.",
            file=sys.stderr,
        )
        return 1

    print(f"Found a key in {source}: {masked(key)}")
    print("Checking it against the Roboflow API ...")
    ok, detail = verify(key)
    if not ok:
        print(f"\nThat key did not work: {detail}", file=sys.stderr)
        return 1
    print(f"Key works. Workspace: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
