"""Presentation helpers for synthetic local and cohort SHAP explanations."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display


class HaybarnSession:
    """Keep JupySQL statements on one native session so SQL variables persist."""

    def __init__(self, connection: Any) -> None:
        """Wrap the native connection without changing SQL or its results."""
        self._connection = connection

    def cursor(self) -> Any:
        """Reuse the native session instead of creating a new variable scope."""
        return self._connection

    def commit(self) -> None:
        """Delegate the DBAPI commit operation."""
        self._connection.commit()

    def close(self) -> None:
        """Close the native connection and its worker processes."""
        self._connection.close()


def theme() -> None:
    """Apply readable shared chart settings."""
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    plt.rcParams.update(
        {"figure.dpi": 130, "text.parse_math": False, "axes.spines.top": False, "axes.spines.right": False}
    )


def prose(text: str) -> None:
    """Display result-derived Markdown."""
    display(Markdown(text))


def finish(fig: plt.Figure, title: str, subtitle: str, filename: str) -> None:
    """Add source context and save the chart for review."""
    fig.suptitle(title, x=0.07, y=0.98, ha="left", fontsize=16, weight="bold", color="#253d31")
    fig.text(0.07, 0.925, subtitle, fontsize=10.5, color="#52675a")
    fig.text(
        0.07,
        0.025,
        "Source: synthetic SQL grid · sklearn v0.3.0 + explain v0.4.0 · real workers through Haybarn / VGI\n"
        "RandomForestRegressor: 64 trees, depth 6, seed 7 · SHAP model attributions, not causal effects",
        fontsize=9,
        color="#52675a",
    )
    directory = Path("validation/charts")
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / filename, dpi=150)
    plt.show()
