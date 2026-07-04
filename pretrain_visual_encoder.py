import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pcl_dataset import PointCloudDataset
from models.visual_encoder import PointCloudDecoder, PointnetEncoder


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = os.environ.get("SYNERGY_GRASP_PROJECT_ROOT", str(PROJECT_ROOT))


def chamfer_distance(pred, target):
    distances = torch.cdist(pred, target, p=2) ** 2
    pred_to_target = distances.min(dim=2)[0].mean(dim=1)
    target_to_pred = distances.min(dim=1)[0].mean(dim=1)
    return (pred_to_target + target_to_pred).mean()


def train_encoder_decoder(encoder, decoder, dataloader, optimizer, device, num_epochs):
    encoder.train()
    decoder.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for pcd in tqdm(dataloader, desc=f"epoch {epoch + 1}/{num_epochs}"):
            pcd = pcd.to(device)
            visual_feat = encoder(pcd.permute(0, 2, 1))
            recon_pcd = decoder(visual_feat)
            loss = chamfer_distance(recon_pcd, pcd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"epoch={epoch + 1} loss={epoch_loss / len(dataloader):.6f}")


def default_pointcloud_dirs(data_root):
    data_root = Path(data_root) / "data"
    return [
        data_root / "pointcloud_allegro",
        data_root / "pointcloud_barrett",
        data_root / "pointcloud_robotiq_3f",
        data_root / "pointcloud",
    ]


def main():
    parser = argparse.ArgumentParser(description="Pretrain the PointNet visual encoder.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pcl_dirs", nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--encoder_out",
        default=str(PROJECT_ROOT / "checkpoints" / "pretrained_pointnet_encoder.pth"),
    )
    parser.add_argument(
        "--decoder_out",
        default=str(PROJECT_ROOT / "checkpoints" / "pretrained_pointcloud_decoder.pth"),
    )
    args = parser.parse_args()

    pcl_dirs = [Path(path) for path in args.pcl_dirs] if args.pcl_dirs else default_pointcloud_dirs(args.data_root)
    dataset = PointCloudDataset(pcl_dirs, augment=args.augment)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PointnetEncoder(path_to_cfg=PROJECT_ROOT / "configs" / "visual_encoder.yaml").to(device)
    decoder = PointCloudDecoder(input_dim=1024, num_points=args.num_points).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    train_encoder_decoder(encoder, decoder, dataloader, optimizer, device, args.num_epochs)

    Path(args.encoder_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.decoder_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), args.encoder_out)
    torch.save(decoder.state_dict(), args.decoder_out)
    print(f"wrote {args.encoder_out}")
    print(f"wrote {args.decoder_out}")


if __name__ == "__main__":
    main()
