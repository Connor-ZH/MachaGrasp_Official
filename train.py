import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print("TensorBoard is not installed; metrics will not be written.")

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


PROJECT_ROOT = Path(__file__).resolve().parent
USE_EIGENGRASP_LOSS = True
USE_DATA_AUGMENTATION = True


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_lr_scheduler(optimizer, num_epochs, version):
    if version == 1:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, eps=1e-16
        )

    warmup_epochs = 5
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-16
    )
    return SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )


def save_checkpoint(model, optimizer, epoch, best_loss, file_path):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        },
        file_path,
    )


def load_checkpoint(model, path):
    if not path.is_file():
        print(f"No checkpoint found at '{path}'")
        model.load_pretrained_pcl_extractor_if_needed()
        return float("inf"), 0

    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Loaded checkpoint with loss {checkpoint['best_loss']:.4f}")
    return checkpoint["best_loss"], checkpoint["epoch"] + 1


def r_squared(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def compute_loss(
    articulations_pred,
    articulations_gt,
    articulation_criterion,
    weight_regression,
    eigengrasp_pred,
    eigengrasp_gt,
    eigengrasp_criterion,
    weight_eigengrasp,
    jacobian_weight,
):
    regression_loss = weight_regression * articulation_criterion(
        articulations_pred, articulations_gt, jacobian_weight
    )

    eigengrasp_loss = eigengrasp_criterion(eigengrasp_pred, eigengrasp_gt)
    return regression_loss + weight_eigengrasp * eigengrasp_loss, eigengrasp_loss


def validate(
    model,
    dataloader,
    articulation_criterion,
    eigengrasp_criterion,
    device,
    use_eigengrasp_loss,
    weight_regression,
    args,
    desc="validation",
):
    model.eval()
    val_loss = 0.0
    val_eigengrasp_loss = 0.0
    predictions = []
    labels = []

    with torch.no_grad():
        for (
            trans,
            rots,
            pcls,
            articulations,
            embodiment_ids,
            eigengrasps,
            _,
            _,
            _,
            _,
            _,
            jacobian_weight,
        ) in tqdm(dataloader, desc=desc):
            trans = trans.to(device)
            rots = rots.to(device)
            pcls = pcls.to(device)
            articulations = articulations.to(device)
            embodiment_ids = embodiment_ids.to(device)
            eigengrasps = eigengrasps.to(device)
            jacobian_weight = jacobian_weight.to(device)

            eigengrasps_pred, articulations_pred, _ = model(
                pcls, trans, rots, embodiment_ids
            )
            loss, eigengrasp_loss = compute_loss(
                articulations_pred,
                articulations,
                articulation_criterion,
                weight_regression,
                eigengrasps_pred,
                eigengrasps,
                eigengrasp_criterion,
                1 if use_eigengrasp_loss else 0,
                jacobian_weight,
            )
            val_loss += loss.item() * articulations.size(0)
            val_eigengrasp_loss += eigengrasp_loss.item() * articulations.size(0)
            predictions.append(articulations_pred)
            labels.append(articulations)

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return (
        val_loss / len(dataloader.dataset),
        r_squared(labels, predictions),
        val_eigengrasp_loss / len(dataloader.dataset),
    )


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    articulation_criterion,
    eigengrasp_criterion,
    optimizer,
    scheduler,
    args,
    tag,
    device,
):
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{tag}_best.pth"
    best_val_loss, current_epoch = load_checkpoint(model, checkpoint_path)

    if args.ignore_loading_val_loss:
        best_val_loss = float("inf")

    writer = SummaryWriter(log_dir=str(PROJECT_ROOT / "logs" / tag))
    model.to(device)
    eigengrasp_loss_weight = 1 if args.use_eigengrasp_loss else 0

    for epoch in range(current_epoch, args.num_epochs):
        model.train()
        running_loss = 0.0
        running_eigengrasp_loss = 0.0
        predictions = []
        labels = []

        for (
            trans,
            rots,
            pcls,
            articulations,
            embodiment_ids,
            eigengrasps,
            _,
            _,
            _,
            _,
            _,
            jacobian_weight,
        ) in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.num_epochs}"):
            trans = trans.to(device)
            rots = rots.to(device)
            pcls = pcls.to(device)
            articulations = articulations.to(device)
            embodiment_ids = embodiment_ids.to(device)
            eigengrasps = eigengrasps.to(device)
            jacobian_weight = jacobian_weight.to(device)

            optimizer.zero_grad()
            eigengrasps_pred, articulations_pred, _ = model(
                pcls, trans, rots, embodiment_ids, True
            )
            loss, eigengrasp_loss = compute_loss(
                articulations_pred,
                articulations,
                articulation_criterion,
                args.regression_loss_weight,
                eigengrasps_pred,
                eigengrasps,
                eigengrasp_criterion,
                eigengrasp_loss_weight,
                jacobian_weight,
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * articulations.size(0)
            running_eigengrasp_loss += eigengrasp_loss.item() * articulations.size(0)
            predictions.append(articulations_pred.detach())
            labels.append(articulations.detach())

        train_loss = running_loss / len(train_loader.dataset)
        train_eigengrasp_loss = running_eigengrasp_loss / len(train_loader.dataset)
        train_r2 = r_squared(torch.cat(labels, dim=0), torch.cat(predictions, dim=0))
        print(
            f"Epoch [{epoch + 1}/{args.num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_r2={train_r2:.4f} "
            f"train_eigengrasp_loss={train_eigengrasp_loss:.4f}"
        )
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("R2/Train", train_r2, epoch)
        writer.add_scalar("Eigengrasp Loss/Train", train_eigengrasp_loss, epoch)

        if epoch % args.val_epoch == 0 and epoch != 0:
            val_loss, val_r2, val_eigengrasp_loss = validate(
                model,
                val_loader,
                articulation_criterion,
                eigengrasp_criterion,
                device,
                args.use_eigengrasp_loss,
                args.regression_loss_weight,
                args,
                desc="validation",
            )
            print(
                f"val_loss={val_loss:.4f} "
                f"val_r2={val_r2:.4f} "
                f"val_eigengrasp_loss={val_eigengrasp_loss:.4f}"
            )
            if args.lr_scheduler_version == 1:
                scheduler.step(val_loss)
            else:
                scheduler.step()
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("R2/Validation", val_r2, epoch)
            writer.add_scalar("Eigengrasp Loss/Validation", val_eigengrasp_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path)
                print(f"Model improved at epoch {epoch + 1}; saved {checkpoint_path}")

        if test_loader is not None and epoch % 20 == 0 and epoch != 0:
            test_loss, test_r2, test_eigengrasp_loss = validate(
                model,
                test_loader,
                articulation_criterion,
                eigengrasp_criterion,
                device,
                args.use_eigengrasp_loss,
                args.regression_loss_weight,
                args,
                desc="test_unseen",
            )
            print(
                f"test_loss={test_loss:.4f} "
                f"test_r2={test_r2:.4f} "
                f"test_eigengrasp_loss={test_eigengrasp_loss:.4f}"
            )
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("R2/Test", test_r2, epoch)
            writer.add_scalar("Eigengrasp Loss/Test", test_eigengrasp_loss, epoch)


def build_tag(args):
    tag_parts = ["synergy_grasper"]
    if args.use_eigengrasp_loss:
        tag_parts.append("eigengrasp")
    if args.augment:
        tag_parts.append("augmented")
    return "_".join(tag_parts)


def validate_training_config(config):
    expected_values = {
        "embodiment_encoding": "graphormer",
        "share_embedding": True,
        "use_whitened_eigengrasp": True,
        "enable_energy_concentration_loss": False,
        "use_rot6d": True,
        "use_pretrain_visual_encoder": True,
        "visual_encoder_trainable": True,
        "visual_encoder_type": 0,
        "use_morphology_aware_loss": True,
        "morphology_aware_loss_type": 2,
    }
    for key, expected in expected_values.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"Expected {key}: {expected}, got {actual}")

    dataset_cfg = config.get("dataset", {})
    if dataset_cfg.get("name") != "synergy":
        raise ValueError("The released training recipe expects dataset.name: synergy.")
    if dataset_cfg.get("hands") != ["allegro", "shadow", "barrett"]:
        raise ValueError("The released training recipe expects hands [allegro, shadow, barrett].")

    pooling_cfg = config.get("morphology_pooling", {})
    if pooling_cfg.get("enable") is not True or pooling_cfg.get("method") != "attention":
        raise ValueError("The released training recipe expects attention morphology pooling.")

    policy_cfg = config.get("policy_transformer", {})
    if policy_cfg.get("amplitude_model") != 2:
        raise ValueError("The released training recipe expects policy_transformer.amplitude_model: 2.")

    transformer_cfg = config.get("transformer", {})
    if transformer_cfg.get("enable_input_layer_norm") is not True:
        raise ValueError("The released training recipe expects transformer.enable_input_layer_norm: True.")


def main():
    parser = argparse.ArgumentParser(description="Synergy grasp training script")
    parser.add_argument("--morphology_conf", default="model")
    parser.add_argument("--regression_loss_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=4000)
    parser.add_argument("--lr_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=48)
    parser.add_argument("--val_epoch", type=int, default=1)
    parser.add_argument(
        "--data_root",
        default=os.environ.get("SYNERGY_GRASP_PROJECT_ROOT", str(PROJECT_ROOT)),
        help="Path to the project/data root that contains the data/ directory.",
    )
    parser.add_argument("--ignore_loading_val_loss", action="store_true")
    args = parser.parse_args()
    args.use_eigengrasp_loss = USE_EIGENGRASP_LOSS
    args.augment = USE_DATA_AUGMENTATION

    from data.dataset import SynergyDataset
    from models.model import SynergyGrasper
    from morphology.embodiment_property import get_embodiment_property
    from morphology.loss import JacobianMSELoss

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = PROJECT_ROOT / "configs" / f"{args.morphology_conf}.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    validate_training_config(config)

    eigengrasp_head_count = config["heads"]["eigengrasp_head"]["count"]
    max_dof_count = config["max_degree_count"]
    args.lr_scheduler_version = config["lr_scheduler_version"]

    model = SynergyGrasper(
        get_embodiment_property(),
        morphology_encoder_config=str(config_path),
        visual_encoder_ckpt_path=str(PROJECT_ROOT / "checkpoints" / "pretrained_pointnet_encoder.pth"),
    ).to(device)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(
        trainable_parameters,
        lr=args.lr_rate,
        weight_decay=args.weight_decay,
    )
    articulation_criterion = JacobianMSELoss()
    eigengrasp_criterion = nn.MSELoss()
    scheduler = build_lr_scheduler(optimizer, args.num_epochs, args.lr_scheduler_version)

    train_data = SynergyDataset(
        "train",
        eigengrasp_head_count,
        max_dof_count,
        augment=args.augment,
        data_root=args.data_root,
    )
    val_data = SynergyDataset(
        "val",
        eigengrasp_head_count,
        max_dof_count,
        data_root=args.data_root,
    )
    test_data = SynergyDataset(
        "test_unseen",
        eigengrasp_head_count,
        max_dof_count,
        data_root=args.data_root,
    )
    train_loader = train_data.get_loader(args.batch_size, args.num_workers)
    val_loader = val_data.get_loader(args.batch_size, args.num_workers)
    test_loader = test_data.get_loader(args.batch_size, args.num_workers)

    tag = build_tag(args)
    train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        articulation_criterion,
        eigengrasp_criterion,
        optimizer,
        scheduler,
        args,
        tag,
        device,
    )


if __name__ == "__main__":
    main()
