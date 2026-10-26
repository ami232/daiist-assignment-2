"""Costes, barrido de umbrales y carga de revisión humana.

Los números del bloque COSTES son PROVISIONALES: sirven para probar el
dashboard y NO son tus cifras. Sustitúyelos por los tuyos (con su
justificación en REPORT.md) y pon PROVISIONAL = False.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"

# ===================== COSTES (EDITA AQUÍ) =====================
PROVISIONAL = False
# Escenario: placas de control industrial, línea de ~5.000 placas/día.
C_FN = 150.0    # EUR: defectuosa que llega al cliente (sustitución, reparación, devolución, garantía, reputación)
C_FP = 12.0     # EUR: buena rechazada (reinspección y, a veces, descarte o daño al retrabajar)
C_TP = 8.0      # EUR: retrabajo en fábrica de una defectuosa bien detectada
C_TN = 0.0      # EUR: buena que pasa
MIN_PER_REVIEW = 1.0         # minutos por revisión (defectos muy pequeños)
EUR_PER_MIN = 0.5            # EUR/min de un revisor (~30 EUR/h con cargas)
C_REVIEW = MIN_PER_REVIEW * EUR_PER_MIN   # EUR por revisión; se asume que la persona acierta
BASE_RATE = 0.03             # fracción REAL de placas defectuosas en producción
BOARDS_PER_DAY = 5000        # placas diarias en la línea
MIN_PER_PERSON_DAY = 420     # minutos útiles de revisión por persona y día
# ================================================================


def load_scores() -> pd.DataFrame:
    """Puntuación por placa: confianza máxima de cualquier detección."""
    return pd.read_csv(OUT / "scores_finetuned.csv")


def outcome_rates(df, t_low, t_high):
    """Reparto (pasa, revisión, falla) entre placas defectuosas y buenas.

    score < t_low -> pasa | t_low <= score < t_high -> revisión | >= t_high -> falla
    """
    def split(s):
        if len(s) == 0:
            return (np.nan, np.nan, np.nan)
        passed = float((s < t_low).mean())
        failed = float((s >= t_high).mean())
        return (passed, 1.0 - passed - failed, failed)

    d = df.loc[df.has_defect == 1, "score"].to_numpy()
    g = df.loc[df.has_defect == 0, "score"].to_numpy()
    return split(d), split(g)


def cost_per_1000(df, t_low, t_high):
    (dp, dr, dfail), (gp, gr, gfail) = outcome_rates(df, t_low, t_high)
    pi = BASE_RATE
    cost_defective = dp * C_FN + dr * (C_REVIEW + C_TP) + dfail * C_TP
    cost_good = gp * C_TN + gr * (C_REVIEW + C_TN) + gfail * C_FP
    return 1000.0 * (pi * cost_defective + (1 - pi) * cost_good)


def review_share(df, t_low, t_high):
    (_, dr, _), (_, gr, _) = outcome_rates(df, t_low, t_high)
    return BASE_RATE * dr + (1 - BASE_RATE) * gr


def review_load(df, t_low, t_high):
    share = review_share(df, t_low, t_high)
    per_day = BOARDS_PER_DAY * share
    people = per_day * MIN_PER_REVIEW / MIN_PER_PERSON_DAY
    return share, per_day, people


def confusion_per_1000(df, t_low, t_high):
    """Matriz esperada por cada 1000 placas, con la tasa base asumida."""
    d, g = outcome_rates(df, t_low, t_high)
    pi = BASE_RATE
    rows = {
        "Defectuosa (real)": [1000 * pi * x for x in d],
        "Buena (real)": [1000 * (1 - pi) * x for x in g],
    }
    return pd.DataFrame(rows, index=["Pasa", "Revisión", "Falla"]).T.round(1)


def sweep(df):
    """Coste por 1000 placas con un único umbral (sin revisión humana)."""
    ts = np.append(np.linspace(0, 1, 101), 1.0001)
    return ts, np.array([cost_per_1000(df, t, t) for t in ts])


def best_single(df):
    ts, costs = sweep(df)
    i = int(np.argmin(costs))
    return float(ts[i]), float(costs[i])


def best_pair(df, step=0.02):
    best = (0.0, 0.0, float("inf"))
    grid = np.round(np.arange(0, 1.0001, step), 4)
    for lo in grid:
        for hi in grid[grid >= lo]:
            c = cost_per_1000(df, lo, hi)
            if c < best[2]:
                best = (float(lo), float(hi), float(c))
    return best


def policy_comparison(df):
    """Coste por 1000 placas de cada política (paso 7 del enunciado)."""
    pi = BASE_RATE
    t1, c1 = best_single(df)
    lo, hi, c2 = best_pair(df)
    return pd.DataFrame({
        "Política": [
            "No hacer nada (todo pasa)",
            "Una persona revisa todo",
            f"Modelo solo (umbral {t1:.2f})",
            f"Modelo + revisión humana ({lo:.2f}-{hi:.2f})",
        ],
        "Coste por 1000 placas (EUR)": [
            round(1000 * (pi * C_FN + (1 - pi) * C_TN), 1),
            round(1000 * (C_REVIEW + pi * C_TP + (1 - pi) * C_TN), 1),
            round(c1, 1),
            round(c2, 1),
        ],
    })