import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
HANDS = ("allegro", "barrett", "shadow")


def terminal_print(message):
    print(message, flush=True)


def terminal_result(message):
    line = f"{message}\n"
    try:
        with open("/dev/tty", "a", buffering=1) as terminal:
            terminal.write(line)
            terminal.flush()
    except OSError:
        print(message, flush=True)


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_command(command):
    terminal_print(" ".join(str(part) for part in command))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def read_summary(summary_path):
    with open(summary_path, "r", newline="") as file:
        return list(csv.DictReader(file))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run released GraspNet validation: GraspNet wrist poses, model articulation "
            "prediction, Isaac Gym simulation, and terminal success summary."
        )
    )
    parser.add_argument("--data_root", default=os.environ.get("SYNERGY_GRASP_PROJECT_ROOT", str(PROJECT_ROOT)))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / "released_model.pth")
    parser.add_argument("--config", type=Path, default=Path("configs") / "model.yaml")
    parser.add_argument(
        "--grasp_meta",
        type=Path,
        default=Path("data") / "graspnet" / "graspnet_meta.npz",
    )
    parser.add_argument("--hands", nargs="+", choices=HANDS, default=list(HANDS))
    parser.add_argument("--output_dir", type=Path, default=Path("results") / "validation")
    parser.add_argument("--attempts_per_object", type=int, default=50)
    parser.add_argument("--inference_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1400)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = resolve_project_path(args.checkpoint)
    config = resolve_project_path(args.config)
    output_dir = resolve_project_path(args.output_dir)

    command = [
        args.python,
        "-m",
        "simulation.run_graspnet_simulation_all_hands",
        "--data_root",
        args.data_root,
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config),
        "--output_dir",
        str(output_dir),
        "--hands",
        *args.hands,
        "--attempts_per_object",
        str(args.attempts_per_object),
        "--inference_batch_size",
        str(args.inference_batch_size),
        "--batch_size",
        str(args.batch_size),
        "--gpu",
        str(args.gpu),
    ]
    if args.grasp_meta is not None:
        command.extend(["--grasp_meta", str(resolve_project_path(args.grasp_meta))])
    if args.headless and not args.gui:
        command.append("--headless")
    if args.overwrite:
        command.append("--overwrite")

    terminal_print(
        f"Running GraspNet validation for hands: {', '.join(args.hands)} "
        f"({args.attempts_per_object} attempts/object)"
    )
    run_command(command)

    summary_path = output_dir / "graspnet_simulation_summary.csv"
    rows = read_summary(summary_path)
    terminal_result("")
    terminal_result("FINAL_VALIDATION_SUMMARY")
    for row in rows:
        inference_time = row["InferenceSeconds"]
        try:
            inference_time = f"{float(inference_time):.4f}"
        except ValueError:
            pass
        terminal_result(
            "FINAL_VALIDATION_RESULT "
            f"hand={row['Hand']} "
            f"objects={row['Objects']} "
            f"grasps={row['Grasps']} "
            f"success={row['Success']} "
            f"success_rate={float(row['SuccessRate']):.4f} "
            f"inference_time_sec={inference_time} "
            f"result_file={row['SimulationResult']}"
        )
    terminal_result(f"summary_csv: {summary_path}")


if __name__ == "__main__":
    main()
