# Add the parent directory to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from modules.utils.aggregator import load_folders
from modules.utils.hydraqol import run_decorator

# Registering the config path with Hydra
@hydra.main(config_path="../config", config_name="report", version_base="1.3")
@run_decorator
def main(cfg: DictConfig) -> None:
    """
    Main function for aggregating and reporting results from multiple training runs.
    Creates both tabular and visual representations of the results.

    Args:
        cfg (DictConfig): Configuration object containing all parameters and sub-configurations.
            Structure and default values of cfg are as follows:
            ```
            defaults:
              - _self_
              - path: relative
            
            base_dir: ${path.base_path_models}  # Base directory for loading results
            max_pool: 8  # Maximum number of parallel processes
            
            # Output path name
            name: report_${now:%Y-%m-%d_%H-%M-%S}
            
            # Hydraqol parameters
            save_dir: ${path.base_path}/reports/${name}
            mode: base
            retry:
              max_retries: 3
              delay: 5
            ```

    Returns:
        None: This function does not return any value.

    Examples:
        To run reporting with the default configuration:
        ```bash
        $ python runs/report.py
        ```

        To specify a different base directory:
        ```bash
        $ python runs/report.py base_dir="path/to/results"
        ```

        To change the number of parallel processes:
        ```bash
        $ python runs/report.py max_pool=4
        ```
    """

    ##############################
    # Preliminaries
    ##############################

    # Initialize logger
    logger = logging.getLogger("report")

    # Create output directory
    output_dir = Path(cfg.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ##############################
    # Load and Process Results
    ##############################

    # Load results from the specified directory
    results = load_folders(Path(cfg.base_dir), max_pool=cfg.max_pool)
    df = pd.json_normalize(results)

    ##############################
    # Create Aggregated Table
    ##############################

    # Define column mappings for metrics and aggregation
    metrics_columns = {
        **{c:c.replace("result.","") for c in df.columns if ("result." in c)}, 
        **{"run_info.total_time_seconds": "time"}
    }
    agg_columns = {
        "config.data.name": "dataset",
        "config.model.name": "model"
    }

    # Combine column mappings
    cols_renaming = {
        **metrics_columns,
        **agg_columns,
    }

    # Get lists of columns
    metrics_columns_list = list(metrics_columns.values())
    agg_columns_list = list(agg_columns.values())

    # Rename and select columns
    dft = df.rename(columns=cols_renaming)
    dft = dft[metrics_columns_list + agg_columns_list]

    # Calculate mean and standard deviation
    dft_mean = dft.groupby(agg_columns_list)[metrics_columns_list].mean().reset_index()
    dft_std = dft.groupby(agg_columns_list)[metrics_columns_list].std().reset_index().fillna(0.001)
    dft_std[dft_mean.isna()] = np.nan

    # Merge mean and std results
    dft_merged = pd.merge(dft_mean, dft_std, on=agg_columns_list, suffixes=('_mean', '_std'))

    # Create final formatted table
    dft_txt = dft_merged.copy()
    for col in metrics_columns_list:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        dft_txt[col] = dft_merged.apply(
            lambda x: f"{x[mean_col]:.3f} ± {x[std_col]:.3f}" if not pd.isna(x[mean_col]) else "N/A",
            axis=1
        )
        dft_txt.drop(columns=[mean_col, std_col], inplace=True)

    # Save the aggregated table
    table_path = output_dir / "results_table.csv"
    dft_txt.to_csv(table_path, index=False)
    dft_txt.to_latex(table_path.with_suffix(".tex"), index=False)
    logger.info(f"Results table saved to: {table_path}")

    ##############################
    # Create Visualization
    ##############################

    # Set the style
    sns.set_palette("husl")

    # Create figure with subplots
    n_metrics = len(metrics_columns_list)
    n_cols = min(3, n_metrics)  # Maximum 3 plots per row
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Create boxplots for each metric
    for idx, metric in enumerate(metrics_columns_list):
        ax = axes[idx]
        # Create boxplot
        sns.boxplot(data=dft, x='model', y=metric, hue='model', ax=ax)
        ax.set_title(f'{metric}')
        ax.set_ylabel(metric)
        # Rotate x-axis labels if they're too long
        ax.tick_params(axis='x', rotation=45)

    # Remove any empty subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout and save the plot
    plt.tight_layout()
    plot_path = output_dir / "results_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.with_suffix(".pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Results plot saved to: {plot_path}")

    # Save the configuration
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Configuration saved to: {config_path}")

if __name__ == '__main__':
    main() 