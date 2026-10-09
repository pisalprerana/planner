
#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def calculate_tracking_error(input_file: Path, output_file: Path) -> None:
    """Analyze the position-error values recorded in a mission CSV."""

    if not input_file.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_file}")

    df = pd.read_csv(input_file)

    if "position_error_m" not in df.columns:
        raise ValueError(
            f"Column 'position_error_m' not found in {input_file}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    errors = pd.to_numeric(df["position_error_m"], errors="coerce").dropna()

    if errors.empty:
        raise ValueError(
            f"No valid position_error_m values found in {input_file}"
        )

    # Save a copy of the original mission data with the error column retained.
    df.to_csv(output_file, index=False)

    mean_error = errors.mean()
    rmse = (errors.pow(2).mean()) ** 0.5
    min_error = errors.min()
    max_error = errors.max()
    within_5m = (errors <= 5.0).mean() * 100.0

    print("\nTracking Error Analysis")
    print("-----------------------")
    print(f"Input file:             {input_file}")
    print(f"Output file:            {output_file}")
    print(f"Valid samples:          {len(errors)}")
    print(f"Mean position error:    {mean_error:.3f} m")
    print(f"RMSE:                   {rmse:.3f} m")
    print(f"Minimum position error:  {min_error:.3f} m")
    print(f"Maximum position error:  {max_error:.3f} m")
    print(f"Samples within 5 m:      {within_5m:.2f}%")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Analyze tracking error recorded in a mission CSV."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to the mission CSV file (optional)",
    )
    parsed_args = parser.parse_args(args)

    if parsed_args.input_file:
        input_file = Path(parsed_args.input_file).expanduser().resolve()
    else:
        # If no file is provided, analyze the newest Dubins mission CSV.
        results_dir = Path(__file__).resolve().parent / "results"

        if not results_dir.exists():
            results_dir = Path(__file__).resolve().parents[1] / "results"

        mission_files = sorted(
            results_dir.glob("dubins_*.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not mission_files:
            raise FileNotFoundError(
                f"No dubins_*.csv mission files found in {results_dir}"
            )

        input_file = mission_files[0]

    if not input_file.is_file():
        raise FileNotFoundError(f"Mission CSV not found: {input_file}")

    output_file = input_file.with_name(
        f"{input_file.stem}_with_error.csv"
    )

    calculate_tracking_error(input_file, output_file)


if __name__ == "__main__":
    main()