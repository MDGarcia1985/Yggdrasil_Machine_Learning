"""
main.py

Purpose:
CLI entrypoint for the Yggdrasil circuit classification MVP.

Runs the same training flow as ``python -m src.train``: load
``Data/raw/circuits_sample.csv``, preprocess, train a RandomForest on
``next_component_type``, and write artifacts under ``Models/``.
"""

from __future__ import annotations

from src.train import main


if __name__ == "__main__":
    main()
