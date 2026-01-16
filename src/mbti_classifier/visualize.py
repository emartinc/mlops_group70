import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_data_distribution(
    processed_data_path: Path = Path("data/processed/train.csv"), output_path: Path = Path("reports/figures")
):
    """
    Generates a plot showing the balance of classes for each MBTI axis.
    """
    if not processed_data_path.exists():
        logger.error(f"File not found: {processed_data_path}")
        return

    # Load data
    df = pd.read_csv(processed_data_path)
    logger.info(f"Loading data from {processed_data_path}...")

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up the plot style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Distribution of MBTI Personality Axes", fontsize=16)

    # Define the 4 axes and their labels
    plot_settings = [
        (axes[0, 0], "is_E", ["Introvert (I)", "Extrovert (E)"], "Blues"),
        (axes[0, 1], "is_S", ["Intuitive (N)", "Sensing (S)"], "Greens"),
        (axes[0, 2], "is_T", ["Feeling (F)", "Thinking (T)"], "Oranges"),
        (axes[0, 3], "is_J", ["Perceiving (P)", "Judging (J)"], "Purples"),
    ]

    # Generate bar charts
    for ax, col, labels, color in plot_settings:
        # Count the values (0 vs 1)
        counts = df[col].value_counts().sort_index()

        # Create bar plot
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=color)

        # Formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_title(f"{col} Distribution")
        ax.set_ylabel("Count")

        # Add numbers on top of bars
        for i, v in enumerate(counts.values):
            ax.text(i, v + 50, str(v), ha="center", fontweight="bold")

    plt.tight_layout()

    # Save the plot
    save_file = output_path / "class_distribution.png"
    plt.savefig(save_file)
    logger.info(f"✅ Chart saved successfully at: {save_file}")


if __name__ == "__main__":
    typer.run(plot_data_distribution)
