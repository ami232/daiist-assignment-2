import collections
import tempfile
from pathlib import Path

import yaml
from roboflow import Roboflow

from api_key import load_api_key

CANDIDATES = [
    ("research-zdvjv", "pcb_defect_detection-3ecqi"),
    ("pcb-defect-detection-emmts", "pcb-defect-detection-datasets-whwcy"),
    ("pcb-qbyda", "solder_detection-64b0a"),
    ("pcb-defect-detection-emmts", "pcb-deffect-detection-solder-sthr7"),
]

rf = Roboflow(api_key=load_api_key())

for ws, proj in CANDIDATES:
    print(f"\n=== {ws}/{proj} ===")
    try:
        versions = rf.workspace(ws).project(proj).versions()
    except Exception as e:
        print("  No se pudo abrir:", e)
        continue
    if not versions:
        print("  Sin versiones -> descartado")
        continue

    print("  Versiones:", [getattr(v, "version", "?") for v in versions])
    latest = max(versions, key=lambda v: int(getattr(v, "version", 0)))
    print("  Descargando la mas reciente:", getattr(latest, "version", "?"))

    with tempfile.TemporaryDirectory() as tmp:
        try:
            ds = latest.download("yolov8", location=tmp)
        except Exception as e:
            print("  Error al descargar:", e)
            continue
        root = Path(ds.location)
        print("  Descargado en:", root)
        yamls = list(root.rglob("data.yaml"))
        if not yamls:
            print("  data.yaml no encontrado. Contenido descargado:")
            for p in list(root.rglob("*"))[:15]:
                print("     ", p)
            continue
        root = yamls[0].parent
        names = yaml.safe_load(yamls[0].read_text())["names"]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]

        for split in ("train", "valid", "test"):
            label_dir = root / split / "labels"
            if not label_dir.is_dir():
                print(f"  {split}: no existe")
                continue
            counts = collections.Counter()
            areas = collections.defaultdict(list)
            files = list(label_dir.glob("*.txt"))
            for f in files:
                for line in f.read_text().splitlines():
                    p = line.split()
                    if len(p) >= 5:
                        c = int(p[0])
                        counts[c] += 1
                        areas[c].append(float(p[3]) * float(p[4]))
            print(f"  {split}: {len(files)} imagenes")
            for c in sorted(counts):
                mean_area = 100 * sum(areas[c]) / len(areas[c])
                print(f"     {names[c]:<22} {counts[c]:>5} instancias | area media {mean_area:.2f}% de la imagen")