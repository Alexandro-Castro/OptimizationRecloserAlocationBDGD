from __future__ import annotations

import numpy as np
import pandas as pd

from .io_bdgd import norm01


def preparar_candidatos(
    df_nos: pd.DataFrame,
    no_raiz: str,
    min_ucs_jus: int = 1,
    min_dist_raiz: float = 1.0,
    compactar_candidatos: bool = True,
    pesos_beneficio: dict[str, float] | None = None,
) -> pd.DataFrame:
    if pesos_beneficio is None:
        pesos_beneficio = {
            "DIC": 0.45,
            "FIC": 0.25,
            "UC": 0.20,
            "TRONCO": 0.10,
            "DIST": 0.00,
        }

    cand = df_nos.copy()
    cand = cand[cand["PAC"] != no_raiz].copy()
    cand = cand[cand["UCs_JUS"] >= min_ucs_jus].copy()
    cand = cand[cand["DIST_RAIZ"] >= min_dist_raiz].copy()

    if compactar_candidatos:
        mask_relevante = (
            (cand["INICIO_RAMAL"] == 1)
            | (cand["BIFURCACAO"] == 1)
            | (cand["CARGA_LOCAL"] == 1)
            | (cand["FIM_RAMAL"] == 1)
        )
        cand = cand[mask_relevante].copy()

    cand["UCs_JUS_N"] = norm01(cand["UCs_JUS"])
    cand["DIC_JUS_N"] = norm01(cand["DIC_JUS"])
    cand["FIC_JUS_N"] = norm01(cand["FIC_JUS"])
    cand["DIST_RAIZ_N"] = norm01(cand["DIST_RAIZ"])

    cand["BENEFICIO"] = (
        pesos_beneficio["DIC"] * cand["DIC_JUS_N"]
        + pesos_beneficio["FIC"] * cand["FIC_JUS_N"]
        + pesos_beneficio["UC"] * cand["UCs_JUS_N"]
        + pesos_beneficio["TRONCO"] * cand["TRONCO_AUTO"]
        + pesos_beneficio["DIST"] * cand["DIST_RAIZ_N"]
    )

    cand = cand[cand["BENEFICIO"] > 0].copy()
    cand = cand.sort_values("BENEFICIO", ascending=False).reset_index(drop=True)
    cand["ID_CAND"] = np.arange(len(cand))
    return cand
