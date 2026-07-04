import argparse
import os
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.isaac_validator import IsaacValidator

import numpy as np
import torch
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_hand_name(hand):
    aliases = {
        "shadow": "shadow_hand",
        "shadow_hand": "shadow_hand",
        "allegro": "allegro_hand",
        "allegro_hand": "allegro_hand",
        "barrett": "barrett",
        "robotiq_3f": "robotiq_3f",
    }
    if hand not in aliases:
        raise ValueError(f"Unsupported hand: {hand}")
    return aliases[hand]


def default_hand_asset(data_root, hand):
    data_root = Path(data_root)
    if hand == "shadow_hand":
        return (
            data_root / "assets" / "shadow" / "open_ai_assets",
            "hand/shadow_hand.xml",
        )
    if hand == "allegro_hand":
        return data_root / "assets" / "allegro", "allegro.urdf"
    if hand == "barrett":
        return (
            data_root / "assets" / "barrett",
            "Barrett_format.urdf",
        )
    if hand == "robotiq_3f":
        return data_root / "assets" / "robotiq_3f", "robotiq_3f.urdf"
    raise ValueError(f"Unsupported hand: {hand}")


def rot6d_to_matrix(rot6d):
    if isinstance(rot6d, torch.Tensor):
        rot6d = rot6d.detach().cpu().numpy()
    rot6d = np.asarray(rot6d, dtype=np.float64)
    x_axis = rot6d[:3]
    y_axis = rot6d[3:]
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def matrix_to_quat_wxyz(matrix):
    trace = np.trace(matrix)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        diagonal = np.diagonal(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif axis == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def object_asset_from_mesh_path(mesh_path):
    mesh_path = Path(str(mesh_path))
    if "YCB" in str(mesh_path):
        return mesh_path.parent, "model.urdf"
    return mesh_path.parent.parent, "model.urdf"


def load_existing_results(path):
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


def to_float_list(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [float(value) for value in values]


def parse_args():
    parser = argparse.ArgumentParser(description="Run setup-1 Isaac Gym simulation for GraspNet predictions.")
    parser.add_argument("--data_root", default=os.environ.get("SYNERGY_GRASP_PROJECT_ROOT", str(PROJECT_ROOT)))
    parser.add_argument("--hand", choices=["shadow", "shadow_hand", "allegro", "allegro_hand", "barrett", "robotiq_3f"], required=True)
    parser.add_argument("--inference_results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand_root", type=Path, default=None)
    parser.add_argument("--hand_file", default=None)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--simple_run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    hand = normalize_hand_name(args.hand)
    seed_everything(args.seed)

    data = torch.load(args.inference_results, weights_only=False)
    if args.hand_root is None or args.hand_file is None:
        default_root, default_file = default_hand_asset(args.data_root, hand)
        args.hand_root = args.hand_root or default_root
        args.hand_file = args.hand_file or default_file

    start = args.batch * args.batch_size
    total = int(data["articulations"].shape[0])
    end = min(total, (args.batch + 1) * args.batch_size)
    if start >= total:
        print(f"no simulation needed: start index {start} >= inference result size {total}")
        return

    if not args.simple_run:
        previous = load_existing_results(args.output)
        if previous is not None:
            if len(previous) < start:
                raise RuntimeError(f"Result file has {len(previous)} entries, expected at least {start}.")
            if len(previous) > start:
                print(f"batch starting at {start} already processed; skipping")
                return

    mode = "direct" if args.headless else "gui"
    sim = IsaacValidator(gpu=args.gpu, mode=mode, hand_type=hand)
    sim.set_asset(args.hand_root, args.hand_file)

    print(f"running setup-1 GraspNet simulation for {hand}, indices [{start}, {end})")
    for i in tqdm(range(start, end)):
        translation = data["translations"][i] + data["translation_shifts"][i]
        rotation = matrix_to_quat_wxyz(rot6d_to_matrix(data["rot6ds"][i]))
        articulation = data["articulations"][i]
        scale = float(data["scales"][i])
        object_root, object_file = object_asset_from_mesh_path(data["mesh_paths"][i])
        sim.add_env(
            rotation,
            to_float_list(translation),
            articulation,
            scale,
            str(object_root),
            object_file,
        )

    current_result = np.array(sim.run_sim(), dtype=bool)
    print(f"batch success rate: {current_result.sum() / len(current_result):.4f}")

    if args.simple_run:
        failed = np.where(current_result == False)[0] + start
        print(f"failed indices: {failed}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = load_existing_results(args.output)
    if previous is not None:
        result = np.concatenate([previous, current_result])
    else:
        result = current_result
    torch.save(result, args.output)
    print(f"saved {len(result)} results to {args.output}")
    print(f"total success rate: {result.sum() / len(result):.4f}")


if __name__ == "__main__":
    main()
