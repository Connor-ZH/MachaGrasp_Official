import argparse
import math
import os
import random
import unicodedata
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.isaac_validator import IsaacValidator

import numpy as np
import torch
from tqdm import tqdm


YCB_OBJECTS = {
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "037_scissors",
    "052_extra_large_clamp",
}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def euler_to_quat_xyz(roll, pitch, yaw):
    roll = float(roll)
    pitch = float(pitch)
    yaw = float(yaw)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def normalize_hand_name(hand):
    aliases = {
        "shadow": "shadow_hand",
        "shadow_hand": "shadow_hand",
        "allegro": "allegro_hand",
        "allegro_hand": "allegro_hand",
        "barrett": "barrett",
    }
    if hand not in aliases:
        raise ValueError(f"Unsupported hand: {hand}")
    return aliases[hand]


def default_dataset_path(data_root, hand, split):
    dataset_dir = {
        "shadow_hand": "dataset_shadow",
        "allegro_hand": "dataset_allegro",
        "barrett": "dataset_barrett",
    }[hand]
    return Path(data_root) / "data" / dataset_dir / f"{split}_2.pt"


def default_output_path(hand, split):
    split_name = "unseen" if split == "test_unseen" else split
    hand_name = {
        "shadow_hand": "shadow",
        "allegro_hand": "allegro",
        "barrett": "barrett",
    }[hand]
    return Path("results") / f"simulation_{hand_name}_{split_name}.pt"


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
    raise ValueError(f"Unsupported hand: {hand}")


def resolve_object_asset(data_root, hand, object_id, mesh_root):
    data_root = Path(data_root)
    if hand == "shadow_hand":
        root = Path(mesh_root) / object_id / "coacd"
        return unicodedata.normalize("NFC", str(root)), "coacd.urdf", None

    object_root = data_root / "assets" / "objects"
    if object_id in YCB_OBJECTS:
        root = object_root / "ycb" / object_id
    else:
        root = object_root / "google_scanned_objects" / object_id
    return str(root), "model.urdf", 1.0


def load_predictions(path):
    if path is None:
        return None
    return np.load(path, allow_pickle=True)


def load_existing_results(path):
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


def terminal_print(message):
    print(message, flush=True)


def terminal_result(message):
    line = f"{message}\n".encode()
    os.write(1, line)


def print_final_result(hand, result, output=None):
    success = int(result.sum())
    total = int(len(result))
    success_rate = success / total
    terminal_print("=" * 72)
    terminal_print("Simulation Result")
    terminal_result(f"FINAL_SIMULATION_RESULT hand={hand} success_rate={success_rate:.4f} success={success} total={total}")
    terminal_result(f"SIMULATION_SUCCESS_RATE hand={hand} value={success_rate:.4f} success={success} total={total}")
    if output is not None:
        terminal_result(f"SIMULATION_RESULT_FILE hand={hand} path={output}")
    terminal_print("=" * 72)


def to_float_list(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [float(value) for value in values]


def parse_args():
    parser = argparse.ArgumentParser(description="Run setup-1 Isaac Gym simulation.")
    parser.add_argument("--data_root", default=os.environ.get("SYNERGY_GRASP_PROJECT_ROOT", str(PROJECT_ROOT)))
    parser.add_argument("--hand", choices=["shadow", "shadow_hand", "allegro", "allegro_hand", "barrett"], default="allegro")
    parser.add_argument("--split", choices=["test_seen", "test_unseen", "val", "train"], default="test_unseen")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--mesh_root", type=Path, default=None)
    parser.add_argument("--hand_root", type=Path, default=None)
    parser.add_argument("--hand_file", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--simple_run", action="store_true")
    parser.add_argument("--suppress_final_result", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    hand = normalize_hand_name(args.hand)
    seed_everything(args.seed)

    dataset_path = args.dataset or default_dataset_path(args.data_root, hand, args.split)
    gt_data = torch.load(dataset_path, weights_only=False)
    predictions = load_predictions(args.predictions)
    if args.output is None:
        args.output = default_output_path(hand, args.split)

    if args.mesh_root is None:
        args.mesh_root = Path(args.data_root) / "assets" / "objects" / "meshdata"
    if args.hand_root is None or args.hand_file is None:
        default_root, default_file = default_hand_asset(args.data_root, hand)
        args.hand_root = args.hand_root or default_root
        args.hand_file = args.hand_file or default_file

    start = args.batch * args.batch_size
    total = int(gt_data["pose"].shape[0])
    end = min(total, (args.batch + 1) * args.batch_size)
    if start >= total:
        print(f"no simulation needed: start index {start} >= dataset size {total}")
        return

    if not args.simple_run:
        previous = load_existing_results(args.output)
        if previous is not None:
            if len(previous) < start:
                raise RuntimeError(f"Result file has {len(previous)} entries, expected at least {start}.")
            if len(previous) > start:
                print(f"batch starting at {start} already processed; skipping")
                if not args.suppress_final_result:
                    print_final_result(hand, previous, args.output)
                return

    mode = "gui" if args.gui else "direct"
    sim = IsaacValidator(gpu=args.gpu, mode=mode, hand_type=hand)
    sim.set_asset(args.hand_root, args.hand_file)

    print(f"running setup-1 simulation for {hand}, {args.split}, indices [{start}, {end})")
    for i in tqdm(range(start, end)):
        pose = gt_data["pose"][i]
        translation_shift = gt_data["translation_shift"][i]
        translation = to_float_list(pose[0:3] + translation_shift)
        rotation = to_float_list(euler_to_quat_xyz(*pose[3:6]))

        if predictions is None:
            articulations = pose[6:]
        else:
            articulations = torch.as_tensor(predictions[i])

        object_id = str(gt_data["grasp_code"][i])
        obj_root, obj_file, forced_scale = resolve_object_asset(
            args.data_root,
            hand,
            object_id,
            args.mesh_root,
        )
        obj_scale = forced_scale if forced_scale is not None else gt_data["scale"][i]

        sim.add_env(
            rotation,
            translation,
            articulations,
            float(obj_scale),
            obj_root,
            obj_file,
        )

    current_result = np.array(sim.run_sim(), dtype=bool)
    print(f"batch success rate: {current_result.sum() / len(current_result):.4f}")

    if args.simple_run:
        failed = np.where(current_result == False)[0] + start
        print(f"failed indices: {failed}")
        if not args.suppress_final_result:
            print_final_result(hand, current_result)
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
    if not args.suppress_final_result:
        print_final_result(hand, result, args.output)


if __name__ == "__main__":
    main()
