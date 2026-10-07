import csv
import math
from pathlib import Path


def calculate_tracking_error(input_file, output_file):
    errors = []

    with input_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as infile, output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as outfile:

        reader = csv.DictReader(infile)

        fieldnames = reader.fieldnames + [
            "position_error_m"
        ]

        writer = csv.DictWriter(
            outfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in reader:
            x = float(row["x_m"])
            y = float(row["y_m"])

            reference_x = float(
                row["reference_x_m"]
            )
            reference_y = float(
                row["reference_y_m"]
            )

            error = math.hypot(
                x - reference_x,
                y - reference_y,
            )

            row["position_error_m"] = (
                f"{error:.6f}"
            )

            writer.writerow(row)

            errors.append(error)

    if not errors:
        raise ValueError(
            "No trajectory samples were found."
        )

    mean_error = sum(errors) / len(errors)

    rmse = math.sqrt(
        sum(error ** 2 for error in errors)
        / len(errors)
    )

    minimum_error = min(errors)
    maximum_error = max(errors)

    valid_errors = [
        error
        for error in errors
        if error <= 5.0
    ]

    mean_error_5m = (
        sum(valid_errors) / len(valid_errors)
    )

    rmse_5m = math.sqrt(
        sum(error ** 2 for error in valid_errors)
        / len(valid_errors)
    )

    percentage_5m = (
        100.0 * len(valid_errors) / len(errors)
    )

    print()
    print("Tracking error analysis")
    print("-----------------------")
    print(f"Samples: {len(errors)}")
    print(f"Mean error: {mean_error:.3f} m")
    print(f"RMSE: {rmse:.3f} m")
    print(f"Minimum error: {minimum_error:.3f} m")
    print(f"Maximum error: {maximum_error:.3f} m")
    print()
    print("Excluding reference jumps > 5 m")
    print("--------------------------------")
    print(
        f"Valid samples: "
        f"{len(valid_errors)}/{len(errors)}"
    )
    print(
        f"Samples within 5 m: "
        f"{percentage_5m:.1f}%"
    )
    print(
        f"Mean error: "
        f"{mean_error_5m:.3f} m"
    )
    print(
        f"RMSE: "
        f"{rmse_5m:.3f} m"
    )
    print()
    print(f"Error data saved to:")
    print(output_file)


def main():
    input_file = Path(
        "/home/bot/vrx_ws/results/test.csv"
    )

    output_file = Path(
        "/home/bot/vrx_ws/results/test_with_error.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_file}"
        )

    calculate_tracking_error(
        input_file,
        output_file,
    )


if __name__ == "__main__":
    main()