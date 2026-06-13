"""Ferramentas para otimizacao de religadores em redes BDGD."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.recloser_opt.cli import otimizar_alimentador

__all__ = ["otimizar_alimentador"]

