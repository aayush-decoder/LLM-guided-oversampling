"""Small entry point for the reusable, leakage-safe data preparation layer.

The notebook is the primary educational deliverable. This script provides a
structured smoke-test and writes the partition/interpolation artifacts before
the optional Hugging Face training stage.
"""

from pathlib import Path
import pandas as pd

from imbllm_pipeline import (
    ExperimentConfig,
    build_minority_partitions,
    interpolate_minority,
    load_and_split_credit_card,
)


def main():
    config = ExperimentConfig()
    config.results_dir.mkdir(exist_ok=True)
    frame, x_train, x_test, y_train, y_test, features = load_and_split_credit_card(
        config.data_path, config.target, config.test_size, config.seed
    )
    majority, all_minority, visible, unseen = build_minority_partitions(
        x_train, y_train, config.q, config.seed, config.target
    )
    interpolated = interpolate_minority(
        visible, len(majority), features, config.seed, config.target
    )
    pd.DataFrame(
        {
            "dataset": ["full", "train", "test", "D_major", "D*_minor", "D_minor", "unseen", "D_inter"],
            "rows": [len(frame), len(x_train), len(x_test), len(majority), len(all_minority),
                     len(visible), len(unseen), len(interpolated)],
        }
    ).to_csv(config.results_dir / "partition_summary.csv", index=False)
    print("Configuration:", config.as_dict())
    print(f"Prepared {len(interpolated):,} interpolation rows from {len(visible):,} visible minority rows.")


if __name__ == "__main__":
    main()
