from __future__ import annotations

import pandas as pd
from pathlib import Path

from recloser_opt.graph_builder import filtrar_componente_da_raiz, montar_rede_conectada, orientar_rede_radial


def test_filtra_componente_da_raiz_descarta_ilha() -> None:
    arestas = pd.DataFrame(
        [
            {"PAC_1": "R", "PAC_2": "A", "NUM_FASES": 3, "COMP": 1.0, "TIPO": "CP"},
            {"PAC_1": "A", "PAC_2": "B", "NUM_FASES": 3, "COMP": 1.0, "TIPO": "CP"},
            {"PAC_1": "X", "PAC_2": "Y", "NUM_FASES": 3, "COMP": 1.0, "TIPO": "CP"},
        ]
    )

    arestas_conn, G_conn, resumo = filtrar_componente_da_raiz(arestas, "R")

    assert set(G_conn.nodes) == {"R", "A", "B"}
    assert len(arestas_conn) == 2
    assert resumo["nos_descartados"] == 2


def test_orienta_rede_radial_a_partir_da_raiz() -> None:
    arestas = pd.DataFrame(
        [
            {"PAC_1": "B", "PAC_2": "A", "NUM_FASES": 3, "COMP": 1.0, "TIPO": "CP"},
            {"PAC_1": "R", "PAC_2": "A", "NUM_FASES": 3, "COMP": 1.0, "TIPO": "CP"},
        ]
    )
    _, G_conn, _ = filtrar_componente_da_raiz(arestas, "R")

    T = orientar_rede_radial(G_conn, "R")

    assert list(T.predecessors("R")) == []
    assert list(T.predecessors("A")) == ["R"]
    assert list(T.predecessors("B")) == ["A"]


def _write_csv(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_monta_rede_conectada_com_csvs_ficticios(tmp_path: Path) -> None:
    input_dir = tmp_path / "dados_entrada"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()

    _write_csv(
        input_dir / "CTMT.csv",
        """
COD_ID,NOME,PAC_INI
CT01,0001,001R
CT02,0002,999R
""",
    )
    _write_csv(
        input_dir / "SSDMT.csv",
        """
CT_COD_OP,PAC_1,PAC_2,FAS_CON,COMP
0001,001R,001A,ABC,10
0001,001A,001B,ABC,20
0001,001B,001C,ABC,30
0001,999X,999Y,ABC,5
0001,001C,001C,ABC,1
0001,,001Z,ABC,1
0002,999R,999A,ABC,1
""",
    )
    _write_csv(
        input_dir / "UNSEMT.csv",
        """
PAC_1,PAC_2,FAS_CON,TIP_UNID,P_N_OPE,CTMT
001C,001D,ABC,32,F,CT01
001D,001E,ABC,32,A,CT01
999R,999B,ABC,32,F,CT02
""",
    )

    resultado = montar_rede_conectada(
        "0001",
        input_dir=input_dir,
        output_dir=output_dir,
        salvar_csv=True,
    )

    arestas = resultado["arestas_conectadas"]
    diagnostico = resultado["diagnostico"]
    caminho_saida = output_dir / "redes_conectadas" / "0001_arestas_conectadas.csv"

    assert resultado["pac_ini"] == "001R"
    assert resultado["nome"] == "0001"
    assert resultado["cod_id"] == "CT01"
    assert set(arestas["PAC_1"]).union(set(arestas["PAC_2"])) == {"001R", "001A", "001B", "001C", "001D"}
    assert len(arestas) == 4
    assert diagnostico["nos_total"] == 7
    assert diagnostico["arestas_total"] == 5
    assert diagnostico["nos_conectados_raiz"] == 5
    assert diagnostico["arestas_conectadas_raiz"] == 4
    assert diagnostico["nos_descartados"] == 2
    assert diagnostico["eh_arvore"] is True
    assert diagnostico["ciclos_estimados"] == 0
    assert caminho_saida.exists()
