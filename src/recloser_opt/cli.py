from __future__ import annotations

import argparse
from pathlib import Path

from .candidates import preparar_candidatos
from .downstream import calcular_metricas_nos, marcar_tronco_automatico
from .graph_builder import filtrar_componente_da_raiz, montar_arestas_rede, orientar_rede_radial
from .io_bdgd import agrupa_cargas, ler_alm
from .metaheuristics import otimizar_religadores_ga
from .reports import imprimir_resumo_conectividade, imprimir_solucao, salvar_resultados


def otimizar_alimentador(
    alimentador: str,
    n_religadores: int = 3,
    salvar_csv: bool = True,
    min_ucs_jus: int = 1,
    min_dist_raiz: float = 1.0,
    compactar_candidatos: bool = True,
    alpha_penalidade: float = 1.0,
    d0: float = 1000.0,
    min_dist_serie: float = 500.0,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    pop_size: int = 120,
    geracoes: int = 250,
    taxa_mutacao: float = 0.25,
    elite: int = 8,
    seed: int = 42,
) -> dict[str, object]:
    no_raiz, _, ctmt = ler_alm(alimentador, input_dir=input_dir)

    print(f"\nProcessando alimentador {alimentador}")
    print(f"CTMT: {ctmt}")
    print(f"No raiz: {no_raiz}")

    arestas, no_raiz, cod_al, ctmt = montar_arestas_rede(alimentador, input_dir=input_dir)
    arestas_conn, G_conn, resumo = filtrar_componente_da_raiz(arestas, no_raiz)
    imprimir_resumo_conectividade(resumo)

    if not resumo["eh_arvore"]:
        print(
            "\nAtencao: a rede conectada a raiz possui ciclos. "
            "Sera usada uma arvore de menores caminhos a partir da subestacao."
        )

    T = orientar_rede_radial(G_conn, no_raiz)
    cargas = agrupa_cargas(ctmt, input_dir=input_dir)
    df_nos = calcular_metricas_nos(T, no_raiz, cargas)
    df_nos = marcar_tronco_automatico(T, no_raiz, df_nos)

    candidatos = preparar_candidatos(
        df_nos,
        no_raiz=no_raiz,
        min_ucs_jus=min_ucs_jus,
        min_dist_raiz=min_dist_raiz,
        compactar_candidatos=compactar_candidatos,
    )

    print(f"\nQuantidade de nos conectados: {len(df_nos)}")
    print(f"Quantidade de candidatos: {len(candidatos)}")

    solucao, info = otimizar_religadores_ga(
        candidatos,
        n_religadores=n_religadores,
        alpha_penalidade=alpha_penalidade,
        d0=d0,
        min_dist_serie=min_dist_serie,
        pop_size=pop_size,
        geracoes=geracoes,
        taxa_mutacao=taxa_mutacao,
        elite=elite,
        seed=seed,
    )
    imprimir_solucao(solucao, info)

    pasta_saida = None
    if salvar_csv:
        pasta_saida = salvar_resultados(
            alimentador,
            arestas_conn,
            df_nos,
            candidatos,
            solucao,
            info,
            output_dir=output_dir,
        )
        print(f"\nArquivos salvos em: {pasta_saida}")

    return {
        "arestas_conectadas": arestas_conn,
        "grafo_conectado": G_conn,
        "arvore_orientada": T,
        "nos": df_nos,
        "candidatos": candidatos,
        "solucao": solucao,
        "info": info,
        "resumo_conectividade": resumo,
        "cod_al": cod_al,
        "ctmt": ctmt,
        "pasta_saida": pasta_saida,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Otimiza alocacao de religadores para um alimentador BDGD.")
    parser.add_argument("alimentador", nargs="?", default="056001")
    parser.add_argument("--n-religadores", type=int, default=3)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--geracoes", type=int, default=250)
    parser.add_argument("--pop-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sem-salvar", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    otimizar_alimentador(
        alimentador=args.alimentador,
        n_religadores=args.n_religadores,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        salvar_csv=not args.sem_salvar,
        geracoes=args.geracoes,
        pop_size=args.pop_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
