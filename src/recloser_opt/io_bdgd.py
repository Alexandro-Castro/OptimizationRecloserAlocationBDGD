from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dados_entrada"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "saida_otimizacao"
DEFAULT_CONNECTED_NETWORK_DIR = PROJECT_ROOT / "outputs" / "redes_conectadas"
DEFAULT_SOLUTIONS_DIR = PROJECT_ROOT / "outputs" / "solucoes"

FASES_NUM = {
    "ABCN": 3,
    "ABC": 3,
    "ABN": 2,
    "BCN": 2,
    "CAN": 2,
    "AB": 2,
    "BC": 2,
    "CA": 2,
    "AN": 1,
    "BN": 1,
    "CN": 1,
    "A": 1,
    "B": 1,
    "C": 1,
}

TIPO_CHAVE = {
    "34": "Seccionadora unipolar de subestacao",
    "32": "RL",
    "27": "3OP",
    "33": "Seccionadora tripolar de subestacao",
    "22": "CH",
    "29": "DJ",
    "23": "Chave fusivel abertura com carga com aterramento",
    "36": "Seccionalizador monofasico",
    "49": "Chave Tipo Tandem",
    "47": "Seccionadora com lamina de terra",
}


def resolver_entrada(input_dir: str | Path | None = None) -> Path:
    return Path(input_dir) if input_dir is not None else DEFAULT_INPUT_DIR


def resolver_saida(output_dir: str | Path | None = None) -> Path:
    return Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR


def resolver_saida_redes_conectadas(output_dir: str | Path | None = None) -> Path:
    if output_dir is None:
        return DEFAULT_CONNECTED_NETWORK_DIR

    pasta = Path(output_dir)
    if pasta.name == "redes_conectadas":
        return pasta
    return pasta / "redes_conectadas"


def resolver_saida_solucoes(output_dir: str | Path | None = None) -> Path:
    if output_dir is None:
        return DEFAULT_SOLUTIONS_DIR

    pasta = Path(output_dir)
    if pasta.name == "solucoes":
        return pasta
    return pasta / "solucoes"


def caminho_csv(nome_arquivo: str | Path, input_dir: str | Path | None = None) -> Path:
    arq = Path(nome_arquivo)
    if arq.is_absolute():
        return arq
    return resolver_entrada(input_dir) / arq


def detectar_separador(arq: Path) -> str:
    with arq.open("r", encoding="utf-8-sig", errors="ignore") as f:
        primeira_linha = f.readline()
    return ";" if primeira_linha.count(";") > primeira_linha.count(",") else ","


def ler_csv(
    nome_arquivo: str | Path,
    input_dir: str | Path | None = None,
    sep: str | None = None,
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Le CSV preservando identificadores como texto."""
    arq = caminho_csv(nome_arquivo, input_dir)
    if not arq.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {arq}")

    separador = sep if sep is not None else detectar_separador(arq)
    return pd.read_csv(
        arq,
        sep=separador,
        dtype=str,
        usecols=usecols,
        low_memory=False,
    )


def normaliza_id(valor: object) -> str | float:
    """Normaliza identificadores sem converter para inteiro."""
    if pd.isna(valor):
        return np.nan

    s = str(valor).strip().strip('"')
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def to_num(serie: pd.Series, default: float = 0.0) -> pd.Series:
    return (
        serie.astype(str)
        .str.replace(",", ".", regex=False)
        .replace(["nan", "None", ""], np.nan)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(default)
    )


def normaliza_pacs(df: pd.DataFrame, cols: Iterable[str] = ("PAC_1", "PAC_2")) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(normaliza_id)
    return df


def norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    maximo = s.max()
    if maximo <= 0:
        return pd.Series(0.0, index=s.index)
    return s / maximo


def colunas_dic_fic(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    colunas_dic = [f"DIC_{i:02d}" for i in range(1, 13) if f"DIC_{i:02d}" in df.columns]
    colunas_fic = [f"FIC_{i:02d}" for i in range(1, 13) if f"FIC_{i:02d}" in df.columns]
    return colunas_dic, colunas_fic


def ler_alm(alimentador: str, input_dir: str | Path | None = None) -> tuple[str, str, str]:
    al = ler_csv("CTMT.csv", input_dir=input_dir, usecols=["COD_ID", "NOME", "PAC_INI"])
    al["NOME"] = al["NOME"].map(normaliza_id)

    al = al[al["NOME"] == normaliza_id(alimentador)].copy()
    if al.empty:
        raise ValueError(f"Alimentador {alimentador} nao encontrado em CTMT.csv")

    no_raiz = normaliza_id(al["PAC_INI"].iloc[0])
    cod_al = normaliza_id(al["NOME"].iloc[0])
    ctmt = normaliza_id(al["COD_ID"].iloc[0])
    return str(no_raiz), str(cod_al), str(ctmt)


def ler_linhas(alimentador: str, input_dir: str | Path | None = None) -> pd.DataFrame:
    cod_mt = ler_csv(
        "SSDMT.csv",
        input_dir=input_dir,
        usecols=["CT_COD_OP", "PAC_1", "PAC_2", "FAS_CON", "COMP"],
    )
    cod_mt["CT_COD_OP"] = cod_mt["CT_COD_OP"].map(normaliza_id)
    cod_mt = cod_mt[cod_mt["CT_COD_OP"] == normaliza_id(alimentador)].copy()

    cod_mt = normaliza_pacs(cod_mt, ["PAC_1", "PAC_2"])
    cod_mt["NUM_FASES"] = cod_mt["FAS_CON"].map(FASES_NUM)
    cod_mt["COMP"] = to_num(cod_mt["COMP"])

    cod_mt = cod_mt[["PAC_1", "PAC_2", "NUM_FASES", "COMP"]].copy()
    cod_mt["TIPO"] = "CP"
    return cod_mt


def ler_chaves(ctmt: str, input_dir: str | Path | None = None) -> pd.DataFrame:
    sec_mt = ler_csv(
        "UNSEMT.csv",
        input_dir=input_dir,
        usecols=["PAC_1", "PAC_2", "FAS_CON", "TIP_UNID", "P_N_OPE", "CTMT"],
    )
    sec_mt["CTMT"] = sec_mt["CTMT"].map(normaliza_id)
    sec_mt = sec_mt[sec_mt["CTMT"] == normaliza_id(ctmt)].copy()
    sec_mt = sec_mt[sec_mt["P_N_OPE"] == "F"].copy()

    sec_mt = normaliza_pacs(sec_mt, ["PAC_1", "PAC_2"])
    sec_mt["NUM_FASES"] = sec_mt["FAS_CON"].map(FASES_NUM)
    sec_mt["TIPO"] = sec_mt["TIP_UNID"].map(lambda x: TIPO_CHAVE.get(normaliza_id(x), normaliza_id(x)))

    sec_mt = sec_mt[["PAC_1", "PAC_2", "NUM_FASES", "TIPO"]].copy()
    sec_mt["COMP"] = 0.0
    return sec_mt


def ler_reguladores(ctmt: str, input_dir: str | Path | None = None) -> pd.DataFrame:
    arq = caminho_csv("UNREMT.csv", input_dir=input_dir)
    colunas = ["PAC_1", "PAC_2", "NUM_FASES", "TIPO", "COMP"]
    if not arq.exists():
        return pd.DataFrame(columns=colunas)

    reg_mt = ler_csv(
        "UNREMT.csv",
        input_dir=input_dir,
        usecols=["PAC_1", "PAC_2", "FAS_CON", "CTMT"],
    )
    reg_mt["CTMT"] = reg_mt["CTMT"].map(normaliza_id)
    reg_mt = reg_mt[reg_mt["CTMT"] == normaliza_id(ctmt)].copy()

    if reg_mt.empty:
        return pd.DataFrame(columns=colunas)

    reg_mt = normaliza_pacs(reg_mt, ["PAC_1", "PAC_2"])
    reg_mt["NUM_FASES"] = reg_mt["FAS_CON"].map(FASES_NUM)

    reg_mt = reg_mt[["PAC_1", "PAC_2", "NUM_FASES"]].copy()
    reg_mt["TIPO"] = "RT"
    reg_mt["COMP"] = 0.0
    return reg_mt


def ler_cargas_mt(ctmt: str, input_dir: str | Path | None = None) -> pd.DataFrame:
    base_cols = ["PAC", "CTMT", "COD_ID"]
    dic_fic = [f"DIC_{i:02d}" for i in range(1, 13)] + [f"FIC_{i:02d}" for i in range(1, 13)]
    ucmt = ler_csv("UCMT.csv", input_dir=input_dir, usecols=lambda c: c in set(base_cols + dic_fic))
    ucmt["CTMT"] = ucmt["CTMT"].map(normaliza_id)
    ucmt = ucmt[ucmt["CTMT"] == normaliza_id(ctmt)].copy()

    if ucmt.empty:
        return pd.DataFrame(columns=["PAC", "UCs", "DIC", "FIC", "GRUPO"])

    ucmt["PAC"] = ucmt["PAC"].map(normaliza_id)
    colunas_dic, colunas_fic = colunas_dic_fic(ucmt)

    for col in colunas_dic + colunas_fic:
        ucmt[col] = to_num(ucmt[col])

    ucmt["DIC_MEDIO"] = ucmt[colunas_dic].mean(axis=1) if colunas_dic else 0.0
    ucmt["FIC_MEDIO"] = ucmt[colunas_fic].mean(axis=1) if colunas_fic else 0.0

    df_final = (
        ucmt.groupby("PAC", as_index=False)
        .agg(
            UCs=("COD_ID", "count"),
            DIC=("DIC_MEDIO", "sum"),
            FIC=("FIC_MEDIO", "sum"),
        )
    )
    df_final["GRUPO"] = "A"
    return df_final


def ler_cargas_bt(ctmt: str, input_dir: str | Path | None = None) -> pd.DataFrame:
    base_cols = ["UNI_TR_MT", "CTMT", "COD_ID"]
    dic_fic = [f"DIC_{i:02d}" for i in range(1, 13)] + [f"FIC_{i:02d}" for i in range(1, 13)]
    ucbt = ler_csv("UCBT.csv", input_dir=input_dir, usecols=lambda c: c in set(base_cols + dic_fic))
    ucbt["CTMT"] = ucbt["CTMT"].map(normaliza_id)
    ucbt = ucbt[ucbt["CTMT"] == normaliza_id(ctmt)].copy()

    if ucbt.empty:
        return pd.DataFrame(columns=["PAC", "UCs", "DIC", "FIC", "GRUPO"])

    colunas_dic, colunas_fic = colunas_dic_fic(ucbt)
    for col in colunas_dic + colunas_fic:
        ucbt[col] = to_num(ucbt[col])

    ucbt["DIC_MEDIO"] = ucbt[colunas_dic].mean(axis=1) if colunas_dic else 0.0
    ucbt["FIC_MEDIO"] = ucbt[colunas_fic].mean(axis=1) if colunas_fic else 0.0
    ucbt["UNI_TR_MT"] = ucbt["UNI_TR_MT"].map(normaliza_id)

    ucbt_g = (
        ucbt.groupby("UNI_TR_MT", as_index=False)
        .agg(
            UCs=("COD_ID", "count"),
            DIC=("DIC_MEDIO", "sum"),
            FIC=("FIC_MEDIO", "sum"),
        )
    )

    trafos = ler_csv("UNTRMT.csv", input_dir=input_dir, usecols=["COD_ID", "PAC_1", "CTMT"])
    trafos["CTMT"] = trafos["CTMT"].map(normaliza_id)
    trafos = trafos[trafos["CTMT"] == normaliza_id(ctmt)].copy()

    if trafos.empty:
        return pd.DataFrame(columns=["PAC", "UCs", "DIC", "FIC", "GRUPO"])

    trafos["COD_ID"] = trafos["COD_ID"].map(normaliza_id)
    trafos["PAC_1"] = trafos["PAC_1"].map(normaliza_id)
    trafos = trafos[["COD_ID", "PAC_1"]].copy()

    merge = pd.merge(trafos, ucbt_g, left_on="COD_ID", right_on="UNI_TR_MT", how="inner")
    if merge.empty:
        return pd.DataFrame(columns=["PAC", "UCs", "DIC", "FIC", "GRUPO"])

    merge["PAC"] = merge["PAC_1"]
    df_final = (
        merge.groupby("PAC", as_index=False)
        .agg(
            UCs=("UCs", "sum"),
            DIC=("DIC", "sum"),
            FIC=("FIC", "sum"),
        )
    )
    df_final["GRUPO"] = "B"
    return df_final


def agrupa_cargas(ctmt: str, input_dir: str | Path | None = None) -> pd.DataFrame:
    ucmt = ler_cargas_mt(ctmt, input_dir=input_dir)
    ucbt = ler_cargas_bt(ctmt, input_dir=input_dir)
    df_concat = pd.concat([ucmt, ucbt], ignore_index=True)

    if df_concat.empty:
        return pd.DataFrame(columns=["PAC", "UCs_A", "UCs_B", "DIC", "FIC"])

    df_concat["PAC"] = df_concat["PAC"].map(normaliza_id)
    df_concat["UCs_A"] = np.where(df_concat["GRUPO"] == "A", df_concat["UCs"], 0)
    df_concat["UCs_B"] = np.where(df_concat["GRUPO"] == "B", df_concat["UCs"], 0)

    return (
        df_concat.groupby("PAC", as_index=False)
        .agg(
            UCs_A=("UCs_A", "sum"),
            UCs_B=("UCs_B", "sum"),
            DIC=("DIC", "sum"),
            FIC=("FIC", "sum"),
        )
    )
