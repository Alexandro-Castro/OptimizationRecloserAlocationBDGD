from __future__ import annotations

from html import escape
import pandas as pd
from pathlib import Path

from .io_bdgd import CSV_DECIMAL_SEPARATOR, CSV_SEPARATOR, resolver_saida, resolver_saida_solucoes


def salvar_csv_saida(df: pd.DataFrame, caminho: Path) -> None:
    df.to_csv(
        caminho,
        sep=CSV_SEPARATOR,
        decimal=CSV_DECIMAL_SEPARATOR,
        index=False,
    )


def _formata_numero_grafico(valor: float) -> str:
    texto = f"{valor:.2f}".rstrip("0").rstrip(".")
    return texto.replace(".", ",")


def _coordenadas_linha(
    xs: list[float],
    ys: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    plot_x: float,
    plot_y: float,
    plot_w: float,
    plot_h: float,
) -> list[tuple[float, float]]:
    x_intervalo = x_max - x_min
    y_intervalo = y_max - y_min

    pontos = []
    for x, y in zip(xs, ys):
        px = plot_x + (plot_w / 2 if x_intervalo == 0 else ((x - x_min) / x_intervalo) * plot_w)
        py = plot_y + (plot_h / 2 if y_intervalo == 0 else plot_h - ((y - y_min) / y_intervalo) * plot_h)
        pontos.append((px, py))
    return pontos


def _polyline(pontos: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pontos)


def salvar_curva_convergencia(
    alimentador: str,
    historico: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    """Salva curva de convergencia do GA em SVG, sem depender de bibliotecas graficas."""
    if historico.empty:
        raise ValueError("Historico vazio; nao e possivel gerar curva de convergencia.")

    colunas_obrigatorias = {"geracao", "melhor_objetivo"}
    faltantes = colunas_obrigatorias - set(historico.columns)
    if faltantes:
        raise ValueError(f"Historico sem colunas obrigatorias: {sorted(faltantes)}")

    dados = historico.copy()
    dados["geracao"] = pd.to_numeric(dados["geracao"], errors="coerce")
    dados["melhor_objetivo"] = pd.to_numeric(dados["melhor_objetivo"], errors="coerce")

    series = [("Melhor objetivo", "melhor_objetivo", "#1f77b4")]
    if "media_objetivo" in dados.columns:
        dados["media_objetivo"] = pd.to_numeric(dados["media_objetivo"], errors="coerce")
        if dados["media_objetivo"].notna().any():
            series.append(("Media da populacao", "media_objetivo", "#d95f02"))

    dados = dados.dropna(subset=["geracao", "melhor_objetivo"]).sort_values("geracao")
    if dados.empty:
        raise ValueError("Historico sem valores numericos validos para gerar curva de convergencia.")

    xs = dados["geracao"].astype(float).tolist()
    valores_y = []
    for _, coluna, _ in series:
        valores_y.extend(dados[coluna].dropna().astype(float).tolist())

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(valores_y)
    y_max = max(valores_y)

    if y_min == y_max:
        folga = max(abs(y_min) * 0.05, 1.0)
        y_min -= folga
        y_max += folga
    else:
        folga = (y_max - y_min) * 0.08
        y_min -= folga
        y_max += folga

    largura = 900
    altura = 520
    margem_esq = 78
    margem_dir = 34
    margem_sup = 52
    margem_inf = 76
    plot_x = margem_esq
    plot_y = margem_sup
    plot_w = largura - margem_esq - margem_dir
    plot_h = altura - margem_sup - margem_inf

    elementos = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222} .grade{stroke:#e5e7eb;stroke-width:1} .eixo{stroke:#374151;stroke-width:1.4}</style>',
        f'<text x="{largura / 2:.1f}" y="28" text-anchor="middle" font-size="19" font-weight="700">Curva de convergencia - Alimentador {escape(str(alimentador))}</text>',
    ]

    for i in range(6):
        frac = i / 5
        y = plot_y + plot_h - frac * plot_h
        valor = y_min + frac * (y_max - y_min)
        elementos.append(f'<line class="grade" x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_w:.1f}" y2="{y:.1f}"/>')
        elementos.append(
            f'<text x="{plot_x - 12:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{_formata_numero_grafico(valor)}</text>'
        )

    for i in range(6):
        frac = i / 5
        x = plot_x + frac * plot_w
        valor = x_min + frac * (x_max - x_min)
        elementos.append(f'<line class="grade" x1="{x:.1f}" y1="{plot_y:.1f}" x2="{x:.1f}" y2="{plot_y + plot_h:.1f}"/>')
        elementos.append(
            f'<text x="{x:.1f}" y="{plot_y + plot_h + 22:.1f}" text-anchor="middle" font-size="12">{_formata_numero_grafico(valor)}</text>'
        )

    elementos.extend(
        [
            f'<line class="eixo" x1="{plot_x:.1f}" y1="{plot_y + plot_h:.1f}" x2="{plot_x + plot_w:.1f}" y2="{plot_y + plot_h:.1f}"/>',
            f'<line class="eixo" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_h:.1f}"/>',
            f'<text x="{plot_x + plot_w / 2:.1f}" y="{altura - 20:.1f}" text-anchor="middle" font-size="14">Geracao</text>',
            f'<text x="22" y="{plot_y + plot_h / 2:.1f}" text-anchor="middle" font-size="14" transform="rotate(-90 22 {plot_y + plot_h / 2:.1f})">Objetivo</text>',
        ]
    )

    legenda_x = plot_x + plot_w - 190
    legenda_y = plot_y + 18
    linhas_plotadas = 0
    for rotulo, coluna, cor in series:
        dados_serie = dados.dropna(subset=[coluna])
        xs_serie = dados_serie["geracao"].astype(float).tolist()
        ys = dados_serie[coluna].astype(float).tolist()
        if not ys:
            continue

        pontos = _coordenadas_linha(xs_serie, ys, x_min, x_max, y_min, y_max, plot_x, plot_y, plot_w, plot_h)
        elementos.append(
            f'<polyline points="{_polyline(pontos)}" fill="none" stroke="{cor}" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if pontos:
            x0, y0 = pontos[0]
            x1, y1 = pontos[-1]
            elementos.append(f'<circle cx="{x0:.2f}" cy="{y0:.2f}" r="3.5" fill="{cor}"/>')
            elementos.append(f'<circle cx="{x1:.2f}" cy="{y1:.2f}" r="3.5" fill="{cor}"/>')

        y_leg = legenda_y + linhas_plotadas * 22
        elementos.append(f'<line x1="{legenda_x:.1f}" y1="{y_leg:.1f}" x2="{legenda_x + 26:.1f}" y2="{y_leg:.1f}" stroke="{cor}" stroke-width="2.8"/>')
        elementos.append(f'<text x="{legenda_x + 34:.1f}" y="{y_leg + 4:.1f}" font-size="12">{escape(rotulo)}</text>')
        linhas_plotadas += 1

    elementos.append("</svg>")

    pasta_saida = resolver_saida_solucoes(output_dir)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho = pasta_saida / f"{alimentador}_curva_convergencia.svg"
    caminho.write_text("\n".join(elementos), encoding="utf-8")
    return caminho


def salvar_resultados(
    alimentador: str,
    arestas_conectadas: pd.DataFrame,
    nos_metricas: pd.DataFrame,
    candidatos: pd.DataFrame,
    solucao: pd.DataFrame,
    info: dict[str, object],
    output_dir: str | Path | None = None,
) -> Path:
    pasta_saida = resolver_saida(output_dir)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    salvar_csv_saida(arestas_conectadas, pasta_saida / f"{alimentador}_arestas_conectadas.csv")
    salvar_csv_saida(nos_metricas, pasta_saida / f"{alimentador}_nos_metricas.csv")
    salvar_csv_saida(candidatos, pasta_saida / f"{alimentador}_candidatos.csv")
    salvar_csv_saida(solucao, pasta_saida / f"{alimentador}_solucao_religadores.csv")

    historico = info.get("historico")
    if isinstance(historico, pd.DataFrame):
        salvar_csv_saida(historico, pasta_saida / f"{alimentador}_historico_ga.csv")

    pares_redundantes = info.get("pares_redundantes")
    if isinstance(pares_redundantes, pd.DataFrame):
        salvar_csv_saida(pares_redundantes, pasta_saida / f"{alimentador}_pares_redundantes.csv")

    return pasta_saida


def salvar_solucao_ga(
    alimentador: str,
    solucao: pd.DataFrame,
    historico: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    pasta_saida = resolver_saida_solucoes(output_dir)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    salvar_csv_saida(solucao, pasta_saida / f"{alimentador}_solucao_ga.csv")
    salvar_csv_saida(historico, pasta_saida / f"{alimentador}_historico_ga.csv")
    salvar_curva_convergencia(alimentador, historico, output_dir=pasta_saida)
    return pasta_saida


def imprimir_resumo_conectividade(resumo: dict[str, object]) -> None:
    print("\nResumo da conectividade:")
    for k, v in resumo.items():
        print(f"  {k}: {v}")


def imprimir_solucao(solucao: pd.DataFrame, info: dict[str, object]) -> None:
    colunas = [
        "PAC",
        "PAI",
        "DIST_RAIZ",
        "UCs_A_JUS",
        "UCs_B_JUS",
        "UCs_JUS",
        "DIC_JUS",
        "FIC_JUS",
        "BENEFICIO",
        "TRONCO_AUTO",
        "INICIO_RAMAL",
        "BIFURCACAO",
        "FIM_RAMAL",
    ]
    colunas_existentes = [col for col in colunas if col in solucao.columns]

    print("\nSolucao encontrada:")
    print(solucao[colunas_existentes])
    print("\nResumo da otimizacao:")
    print(f"  Objetivo: {float(info['objetivo']):.6f}")
    print(f"  Beneficio total: {float(info['beneficio_total']):.6f}")
    print(f"  Penalidade total: {float(info['penalidade_total']):.6f}")
