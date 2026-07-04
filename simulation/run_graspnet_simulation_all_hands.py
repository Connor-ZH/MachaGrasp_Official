import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HANDS = ("allegro", "barrett", "shadow")


def object_name_from_mesh_path(mesh_path):
    mesh_path = Path(str(mesh_path))
    if "YCB" in str(mesh_path):
        return mesh_path.parent.name
    return mesh_path.parent.parent.name


def run_command(command, cwd):
    print(" ".join(str(part) for part in command))
    subprocess.run(command, cwd=cwd, check=True)


def load_success_summary(result_path):
    result = torch.load(result_path, weights_only=False)
    success = int(result.sum())
    total = int(len(result))
    return success, total, success / total


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run setup-1 GraspNet simulation for Allegro, Barrett, and Shadow on the released object split."
    )
    parser.add_argument("--data_root", default=os.environ.get("SYNERGY_GRASP_PROJECT_ROOT", str(project_root)))
    parser.add_argument("--output_dir", type=Path, default=project_root / "results" / "graspnet_simulation")
    parser.add_argument("--checkpoint", type=Path, default=project_root / "checkpoints" / "released_model.pth", help="Model checkpoint used to generate per-hand inference files.")
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "model.yaml")
    parser.add_argument("--visual_encoder_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--grasp_meta",
        type=Path,
        default=project_root / "data" / "graspnet" / "graspnet_meta.npz",
        help="Released GraspNet wrist-pose candidate file for the released 28 objects.",
    )
    parser.add_argument("--hands", nargs="+", choices=HANDS, default=list(HANDS))
    parser.add_argument("--attempts_per_object", type=int, default=50)
    parser.add_argument("--inference_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for hand in args.hands:
        inference_result = args.output_dir / f"graspnet_inference_{hand}.pt"
        simulation_result = args.output_dir / f"graspnet_simulation_{hand}.pt"
        success_csv = args.output_dir / f"graspnet_success_rate_{hand}.csv"

        if args.overwrite:
            for path in (inference_result, simulation_result, success_csv):
                if path.exists():
                    path.unlink()

        if args.overwrite or not inference_result.exists():
            command = [
                args.python,
                "-m",
                "simulation.generate_graspnet_inference",
                "--checkpoint",
                str(args.checkpoint),
                "--config",
                str(args.config),
                "--grasp_meta",
                str(args.grasp_meta),
                "--hand",
                hand,
                "--output",
                str(inference_result),
                "--attempts_per_object",
                str(args.attempts_per_object),
                "--batch_size",
                str(args.inference_batch_size),
            ]
            if args.visual_encoder_checkpoint is not None:
                command.extend(["--visual_encoder_checkpoint", str(args.visual_encoder_checkpoint)])
            run_command(command, project_root)
        inference_data = torch.load(inference_result, weights_only=False)
        grasp_count = int(inference_data["articulations"].shape[0])
        if "object_names" in inference_data:
            object_count = len(set(inference_data["object_names"]))
        else:
            object_count = len({object_name_from_mesh_path(path) for path in inference_data["mesh_paths"]})
        inference_seconds = inference_data.get("inference_time_sec")
        print(f"{hand}: generated {grasp_count} grasps across {object_count} objects")
        if grasp_count != object_count * args.attempts_per_object:
            raise RuntimeError(
                f"{hand}: expected {object_count * args.attempts_per_object} grasps "
                f"({object_count} objects x {args.attempts_per_object}), got {grasp_count}."
            )

        batch_count = (grasp_count + args.batch_size - 1) // args.batch_size
        for batch in range(batch_count):
            command = [
                args.python,
                "-m",
                "simulation.run_graspnet_simulation",
                "--data_root",
                args.data_root,
                "--hand",
                hand,
                "--inference_results",
                str(inference_result),
                "--output",
                str(simulation_result),
                "--batch",
                str(batch),
                "--batch_size",
                str(args.batch_size),
                "--gpu",
                str(args.gpu),
            ]
            if args.headless:
                command.append("--headless")
            run_command(command, project_root)

        run_command(
            [
                args.python,
                "-m",
                "simulation.analyze_simulation_results",
                "--simulation_result",
                str(simulation_result),
                "--inference_results",
                str(inference_result),
                "--output",
                str(success_csv),
            ],
            project_root,
        )

        success, total, success_rate = load_success_summary(simulation_result)
        summary_rows.append(
            {
                "Hand": hand,
                "Objects": object_count,
                "Grasps": total,
                "Success": success,
                "SuccessRate": success_rate,
                "InferenceSeconds": inference_seconds,
                "SimulationResult": str(simulation_result),
                "ObjectCsv": str(success_csv),
            }
        )

    summary_path = args.output_dir / "graspnet_simulation_summary.csv"
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Hand",
                "Objects",
                "Grasps",
                "Success",
                "SuccessRate",
                "InferenceSeconds",
                "SimulationResult",
                "ObjectCsv",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSummary")
    for row in summary_rows:
        inference_text = "skipped" if row["InferenceSeconds"] is None else f"{row['InferenceSeconds']:.4f}"
        print(
            f"{row['Hand']}: {row['Success']}/{row['Grasps']} "
            f"({row['SuccessRate']:.4f}) over {row['Objects']} objects, "
            f"inference_time_sec={inference_text}"
        )
    print(f"summary csv: {summary_path}")


if __name__ == "__main__":
    main()
