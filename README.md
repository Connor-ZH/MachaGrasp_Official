# Synergy Grasp Sampler

This repository contains the release code for training and validating a
multi-hand synergy grasp sampler. The released path includes model training,
visual encoder pretraining, morphology metadata generation from hand URDFs, and
GraspNet-based validation for Allegro, Barrett, and Shadow hands.

The default validation flow uses 50 GraspNet wrist-pose candidates for each of
the released 28 unseen objects, predicts hand articulations with the released
model, runs Isaac Gym simulation, and prints one success summary per hand.

## Installation

Use `environment.yml` as the single dependency specification:

```bash
conda env create -f environment.yml
conda activate synergy_isaac
python -m pip install -e .
```

The environment includes `pytorch-kinematics==0.7.6`, imported in Python as
`pytorch_kinematics`.

Check the base installation:

```bash
python -c "import torch, yaml, trimesh, pytorch_kinematics; print(torch.__version__)"
python train.py --help
python pretrain_visual_encoder.py --help
python -m morphology.meta_generator --help
```

NVIDIA Isaac Gym is required only for simulation validation and is distributed
separately from conda. To enable validation:

1. Download Isaac Gym Preview 4 from NVIDIA.
2. Unpack it outside this repository.
3. Install its Python package inside the `synergy_isaac` environment:

```bash
python -m pip install -e /path/to/isaacgym/python
python -c "from isaacgym import gymapi; print('isaacgym ok')"
```

If Isaac Gym cannot find Python/CUDA libraries at runtime:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Download Release Files

Large release files are distributed separately from the code repository. Download
and extract the released checkpoints, data, and assets with:

```bash
python download_release_assets.py
```

The script downloads these archives from Google Drive and extracts them into the
repository root:

```text
machagrasp_checkpoints.tar.gz
machagrasp_data.tar.gz
machagrasp_assets.tar.gz
```

The same files are also available for manual download from:

https://drive.google.com/drive/folders/1qb4T-0zB5JNEWcF9TGoJc1XEly7vAjff?usp=sharing

If you download them manually, place the archives under `release_archives/` and
extract them with:

```bash
python download_release_assets.py --skip_download
```

This extracts the archives into the repository root:

```text
machagrasp_checkpoints.tar.gz -> checkpoints/
machagrasp_data.tar.gz        -> data/
machagrasp_assets.tar.gz      -> assets/
```

You can override the default Google Drive file IDs with direct archive URLs or
different file IDs:

```bash
python download_release_assets.py \
  --checkpoints_file_id GOOGLE_DRIVE_FILE_ID_FOR_MACHAGRASP_CHECKPOINTS \
  --data_file_id GOOGLE_DRIVE_FILE_ID_FOR_MACHAGRASP_DATA \
  --assets_file_id GOOGLE_DRIVE_FILE_ID_FOR_MACHAGRASP_ASSETS
```

Expected data and checkpoint layout:

```text
checkpoints/released_model.pth
checkpoints/pretrained_pointnet_encoder.pth
configs/model.yaml
configs/visual_encoder.yaml
data/allegro/{train_2.pt,val_2.pt,test_unseen_2.pt,eigengrasps_2_whitened.pt}
data/barrett/{train_2.pt,val_2.pt,test_unseen_2.pt,eigengrasps_2_whitened.pt}
data/shadow/{train_2.pt,val_2.pt,test_unseen_2.pt,eigengrasps_2_whitened.pt}
data/pointcloud_allegro/
data/pointcloud_barrett/
data/pointcloud_shadow/
data/graspnet/graspnet_meta.npz
data/splits/unseen_object_list.json
assets/
morphology/meta/
```

The prepared GraspNet wrist-pose metadata in `data/graspnet/graspnet_meta.npz`
was generated with
[H-Freax/GraspNet_Pointnet2_PyTorch1.13.1](https://github.com/H-Freax/GraspNet_Pointnet2_PyTorch1.13.1.git).
Low-quality wrist poses were further filtered using the classifier from
[tasbolat1/graspflow](https://github.com/tasbolat1/graspflow). You may
regenerate the wrist poses with those tools, but this release already includes
the prepared metadata used by the validation scripts.

## Validation

Run the default three-hand validation:

```bash
python validation.py --gpu 0 --headless
```

This command loads `checkpoints/released_model.pth`, reads
`data/graspnet/graspnet_meta.npz`, predicts hand articulations for Allegro,
Barrett, and Shadow, runs Isaac Gym simulation, and prints final terminal lines
of the form:

```text
FINAL_VALIDATION_RESULT hand=allegro objects=28 grasps=1400 ...
FINAL_VALIDATION_RESULT hand=barrett objects=28 grasps=1400 ...
FINAL_VALIDATION_RESULT hand=shadow objects=28 grasps=1400 ...
```

Validation outputs are written under `results/validation/`:

```text
graspnet_inference_allegro.pt
graspnet_inference_barrett.pt
graspnet_inference_shadow.pt
graspnet_simulation_allegro.pt
graspnet_simulation_barrett.pt
graspnet_simulation_shadow.pt
graspnet_success_rate_allegro.csv
graspnet_success_rate_barrett.csv
graspnet_success_rate_shadow.csv
graspnet_simulation_summary.csv
```

To run one hand or change batching:

```bash
python validation.py \
  --hands allegro \
  --gpu 0 \
  --batch_size 200 \
  --inference_batch_size 256 \
  --headless
```

## Training

The released trainer uses the default recipe: data augmentation, eigengrasp
loss, whitened eigengrasps, trainable pretrained PointNet features, rot6d wrist
rotations, and Jacobian-weighted articulation loss (KAL Loss).

```bash
python train.py \
  --data_root . \
  --morphology_conf model \
  --regression_loss_weight 1.0 \
  --batch_size 256 \
  --num_workers 8
```

## Visual Encoder Pretraining

The released model checkpoint already contains visual encoder weights. To
reproduce the PointNet autoencoder pretraining stage:

```bash
python pretrain_visual_encoder.py \
  --data_root . \
  --augment \
  --batch_size 64 \
  --num_workers 8
```

This writes:

```text
checkpoints/pretrained_pointnet_encoder.pth
checkpoints/pretrained_pointcloud_decoder.pth
```

## Repository Layout

```text
train.py                         training entry point
validation.py                    default three-hand validation entry point
pretrain_visual_encoder.py       PointNet visual encoder pretraining
download_release_assets.py       download/extract helper for Drive-hosted files
environment.yml                  conda environment exported from synergy_isaac
configs/                         model and visual encoder configs
data/                            released splits, point clouds, and GraspNet metadata
assets/                          hand and object URDF/mesh assets
models/                          neural network modules
morphology/                      morphology metadata, loss, and URDF metadata generator
simulation/                      GraspNet inference, Isaac Gym validation, result analysis
checkpoints/                     released model and visual encoder checkpoints
```

## Citation

If you find this work useful, please consider citing the paper:

```bibtex
@misc{zhang2026machagraspmorphologyawarecrossembodimentdexterous,
      title={MachaGrasp: Morphology-Aware Cross-Embodiment Dexterous Hand Articulation Generation for Grasping}, 
      author={Heng Zhang and Kevin Yuchen Ma and Mike Zheng Shou and Weisi Lin and Yan Wu},
      year={2026},
      eprint={2510.06068},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.06068}, 
}
```

## Acknowledgements

The prepared GraspNet wrist poses are generated with
[H-Freax/GraspNet_Pointnet2_PyTorch1.13.1](https://github.com/H-Freax/GraspNet_Pointnet2_PyTorch1.13.1.git)
and filtered with the classifier from
[tasbolat1/graspflow](https://github.com/tasbolat1/graspflow). The embodiment
transformer implementation is adapted from GET-Zero. The Isaac Gym simulation
setup is adapted from DexGraspNet. The README organization is inspired by
DRO-Grasp.
