from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.recloser_opt.cli import otimizar_alimentador  # noqa: E402


def main() -> None:
    otimizar_alimentador(
        alimentador="056001",
        n_religadores=5,
        geracoes=250,
        pop_size=120,
        seed=41,
        salvar_csv=True,
    )


if __name__ == "__main__":
    main()
