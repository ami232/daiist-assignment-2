import sys
from pathlib import Path
from roboflow import Roboflow

env = dict(l.strip().split("=", 1) for l in Path(".env").read_text().splitlines()
           if "=" in l and not l.startswith("#"))
workspace, project, version = sys.argv[1], sys.argv[2], int(sys.argv[3])
rf = Roboflow(api_key=env["ROBOFLOW_API_KEY"])
rf.workspace(workspace).project(project).version(version).download(
    "yolov11", location=str(Path.home() / "datasets" / f"{project}-v{version}"))