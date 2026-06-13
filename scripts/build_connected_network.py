from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recloser_opt.graph_builder import montar_rede_conectada  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monta a rede conectada de um alimentador BDGD.")
    parser.add_argument("alimentador")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sem-salvar", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resultado = montar_rede_conectada(
        args.alimentador,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        salvar_csv=not args.sem_salvar,
    )

    print(f"Alimentador: {resultado['nome']}")
    print(f"COD_ID: {resultado['cod_id']}")
    print(f"PAC_INI: {resultado['pac_ini']}")
    print("Diagnostico:")
    for chave, valor in resultado["diagnostico"].items():
        print(f"  {chave}: {valor}")

    if resultado["caminho_saida"] is not None:
        print(f"Arquivo salvo em: {resultado['caminho_saida']}")


if __name__ == "__main__":
    main()

