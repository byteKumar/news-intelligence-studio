"""Standalone training entry point for Team 116 News Intelligence Studio.

Run this file from the project root to rebuild saved artifacts after placing
the raw datasets in ``data/raw``. The actual training logic lives in
``src.train_pipeline`` so the code can also be reused from the notebook and UI.
"""

from src.train_pipeline import main


if __name__ == '__main__':
    main()
