import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch


def object_name_from_mesh_path(mesh_path):
    mesh_path = Path(str(mesh_path))
    if "YCB" in str(mesh_path):
        return mesh_path.parent.name
    return mesh_path.parent.parent.name


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize setup-1 GraspNet simulation results by object.")
    parser.add_argument("--simulation_result", type=Path, required=True)
    parser.add_argument("--inference_results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = torch.load(args.simulation_result, weights_only=False)
    inference = torch.load(args.inference_results, weights_only=False)
    if len(result) > len(inference["mesh_paths"]):
        raise ValueError("Simulation result is longer than the inference result metadata.")

    grouped = defaultdict(lambda: {"success": 0, "total": 0, "first_index": None})
    for index, success in enumerate(result):
        object_name = object_name_from_mesh_path(inference["mesh_paths"][index])
        entry = grouped[object_name]
        entry["success"] += int(bool(success))
        entry["total"] += 1
        if entry["first_index"] is None:
            entry["first_index"] = index

    rows = []
    for object_name, stats in grouped.items():
        rows.append(
            {
                "Object": object_name,
                "SuccessRate": stats["success"] / stats["total"],
                "Success": stats["success"],
                "Total": stats["total"],
                "OriginalIdx": stats["first_index"],
            }
        )
    rows.sort(key=lambda row: row["SuccessRate"], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Rank", "Object", "SuccessRate", "Success", "Total", "OriginalIdx"])
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            row = dict(row)
            row["Rank"] = rank
            writer.writerow(row)
    print(f"wrote {len(rows)} object rows to {args.output}")


if __name__ == "__main__":
    main()
