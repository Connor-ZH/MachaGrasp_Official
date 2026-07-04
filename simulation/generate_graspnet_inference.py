import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import trimesh as tm
from tqdm import tqdm

from models.model import SynergyGrasper
from morphology.embodiment_property import get_embodiment_property
from simulation.gripper_to_hand_pose import gripper_to_hand_pose


EMBODIMENT_IDS = {
    "shadow": 0,
    "shadow_hand": 0,
    "allegro": 1,
    "allegro_hand": 1,
    "robotiq_3f": 2,
    "barrett": 3,
}


def load_grasp_rows(path):
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "rows" in loaded.files:
            rows = loaded["rows"]
        elif "arr_0" in loaded.files:
            rows = loaded["arr_0"]
        else:
            rows = np.array([loaded[key] for key in loaded.files], dtype=object)
    elif isinstance(loaded, dict):
        return [loaded[key] for key in sorted(loaded)]
    else:
        rows = loaded
    if getattr(rows, "shape", None) == ():
        rows = rows.item()
    return list(rows)


def object_name_from_mesh_path(mesh_path):
    mesh_path = Path(str(mesh_path))
    if mesh_path.parent.name == "meshes":
        return mesh_path.parent.parent.name
    if mesh_path.parent.parent.name in {"YCB", "ycb", "GoogleScannedObjects", "google_scanned_objects"}:
        return mesh_path.parent.name
    return mesh_path.parent.parent.name


def resolve_project_path(path):
    path = Path(str(path))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_object_mesh_path(mesh_path):
    mesh_path = Path(str(mesh_path))
    candidates = []
    if mesh_path.is_absolute():
        candidates.append(mesh_path)
    else:
        candidates.append(PROJECT_ROOT / mesh_path)

    object_name = object_name_from_mesh_path(mesh_path)
    object_family = "ycb" if "YCB" in mesh_path.parts or "ycb" in mesh_path.parts else "google_scanned_objects"
    if object_family == "ycb":
        candidates.append(PROJECT_ROOT / "assets" / "objects" / object_family / object_name / mesh_path.name)
    else:
        candidates.append(
            PROJECT_ROOT / "assets" / "objects" / object_family / object_name / "meshes" / mesh_path.name
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def rotmat_to_rot6d(rotation):
    return rotation[:3, :3][:, :2].T.reshape(-1).astype(np.float32)


def load_model(args, device):
    model = SynergyGrasper(
        get_embodiment_property(),
        morphology_encoder_config=args.config,
        visual_encoder_ckpt_path=args.visual_encoder_checkpoint,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_articulations(model, pcs, translations, rot6ds, embodiment_ids, batch_size):
    outputs = []
    forward_seconds = 0.0
    forward_pass_count = 0
    with torch.no_grad():
        for start in range(0, pcs.shape[0], batch_size):
            end = min(pcs.shape[0], start + batch_size)
            if pcs.device.type == "cuda":
                torch.cuda.synchronize(pcs.device)
            forward_start = time.perf_counter()
            _, articulations, _ = model(
                pcs[start:end],
                translations[start:end],
                rot6ds[start:end],
                embodiment_ids[start:end],
            )
            if pcs.device.type == "cuda":
                torch.cuda.synchronize(pcs.device)
            forward_seconds += time.perf_counter() - forward_start
            forward_pass_count += 1
            outputs.append(articulations.detach().cpu())
    return torch.cat(outputs, dim=0), forward_seconds / forward_pass_count


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate hand predictions for GraspNet grasp candidates.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "model.yaml")
    parser.add_argument("--visual_encoder_checkpoint", type=Path, default=None)
    parser.add_argument("--grasp_meta", type=Path, required=True)
    parser.add_argument("--hand", choices=["shadow", "shadow_hand", "allegro", "allegro_hand", "barrett", "robotiq_3f"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts_per_object", type=int, default=50)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    rows = load_grasp_rows(args.grasp_meta)
    model = load_model(args, device)
    embodiment_id = EMBODIMENT_IDS[args.hand]
    object_names = [object_name_from_mesh_path(row["object_mesh_path"]) for row in rows]
    if len(object_names) != len(set(object_names)):
        raise RuntimeError("GraspNet candidate file contains duplicate object rows.")

    print(
        f"objects={len(rows)} "
        f"attempts_per_object={args.attempts_per_object} "
        f"total_grasps={len(rows) * args.attempts_per_object}"
    )

    all_mesh_paths = []
    all_scales = []
    all_translations = []
    all_articulations = []
    all_translation_shifts = []
    all_rot6ds = []
    all_object_names = []
    forward_times = []

    for row in tqdm(rows, desc="objects"):
        object_mesh_path = str(resolve_object_mesh_path(row["object_mesh_path"]))
        object_name = object_name_from_mesh_path(object_mesh_path)
        if not Path(object_mesh_path).exists():
            raise FileNotFoundError(f"Object mesh not found: {object_mesh_path}")

        grasps = list(row["grasps"])[: args.attempts_per_object]
        if len(grasps) < args.attempts_per_object:
            raise RuntimeError(
                f"Object {object_name} has {len(grasps)} GraspNet candidates, "
                f"expected at least {args.attempts_per_object}."
            )

        scale = float(row["scale"])
        object_mesh = tm.load(object_mesh_path).apply_scale(scale)
        point_cloud, _ = tm.sample.sample_surface(object_mesh, args.num_points)
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        translation_shift = point_cloud.mean(axis=0).astype(np.float32)
        normalized_pc = torch.from_numpy((point_cloud - translation_shift).T).float().to(device)
        translation_shift_tensor = torch.from_numpy(translation_shift).float().to(device)

        pcs = []
        translations = []
        rot6ds = []
        embodiment_ids = []
        translation_shifts = []

        for gripper_pose in grasps:
            hand_pose = gripper_to_hand_pose(args.hand, gripper_pose)
            translation = torch.from_numpy(hand_pose[:3, 3]).float().to(device)
            pcs.append(normalized_pc)
            translations.append(translation - translation_shift_tensor)
            rot6ds.append(torch.from_numpy(rotmat_to_rot6d(hand_pose)).float().to(device))
            embodiment_ids.append(embodiment_id)
            translation_shifts.append(translation_shift_tensor)

        if not pcs:
            continue

        pcs_tensor = torch.stack(pcs)
        translations_tensor = torch.stack(translations)
        rot6ds_tensor = torch.stack(rot6ds)
        embodiment_ids_tensor = torch.tensor(embodiment_ids, dtype=torch.long, device=device)
        translation_shifts_tensor = torch.stack(translation_shifts)
        articulations, forward_time = predict_articulations(
            model,
            pcs_tensor,
            translations_tensor,
            rot6ds_tensor,
            embodiment_ids_tensor,
            args.batch_size,
        )
        forward_times.append(forward_time)

        all_mesh_paths.extend([object_mesh_path] * len(translations))
        all_object_names.extend([object_name] * len(translations))
        all_scales.extend([scale] * len(translations))
        all_translations.append(translations_tensor.detach().cpu())
        all_articulations.append(articulations)
        all_translation_shifts.append(translation_shifts_tensor.detach().cpu())
        all_rot6ds.append(rot6ds_tensor.detach().cpu())

    if not all_mesh_paths:
        raise RuntimeError("No GraspNet rows matched the released object split.")

    output = {
        "mesh_paths": all_mesh_paths,
        "object_names": all_object_names,
        "scales": torch.tensor(all_scales, dtype=torch.float32),
        "translations": torch.cat(all_translations, dim=0),
        "articulations": torch.cat(all_articulations, dim=0),
        "translation_shifts": torch.cat(all_translation_shifts, dim=0),
        "rot6ds": torch.cat(all_rot6ds, dim=0),
        "attempts_per_object": args.attempts_per_object,
        "inference_time_sec": float(np.mean(forward_times)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"saved {len(all_mesh_paths)} predicted grasps to {args.output}")
    print(f"inference_time_sec={output['inference_time_sec']:.4f}")


if __name__ == "__main__":
    main()
