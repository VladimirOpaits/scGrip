#!/usr/bin/env python
"""
export_mmd_to_scvi.py — конвертация MMD lightning-чекпоинта в scVI-формат.

Запуск:
  python export_mmd_to_scvi.py \
      --ckpt /path/to/best-epoch=06-mmd_ab=0.0046.ckpt \
      --warmup-dir /home/vlad/scGRIP/runs/warmup_clean \
      --healthy /home/vlad/scGRIP/data/bank/healthy_train.h5ad \
      --out-dir /home/vlad/scGRIP/runs/mmd_clean_lambda10_v2/exported
"""

import argparse
from pathlib import Path

import anndata as ad
import scvi
import torch


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt', type=Path, required=True,
                   help='Путь к .ckpt файлу')
    p.add_argument('--warmup-dir', type=Path,
                   default=Path('/home/vlad/scGRIP/runs/warmup_clean'),
                   help='Папка с warmup scVI-моделью (шаблон архитектуры)')
    p.add_argument('--healthy', type=Path,
                   default=Path('/home/vlad/scGRIP/data/bank/healthy_train.h5ad'),
                   help='Данные для setup_anndata')
    p.add_argument('--out-dir', type=Path, required=True,
                   help='Куда сохранить scVI-совместимую модель')
    return p.parse_args()


def main():
    args = parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    print(f"Loading {args.healthy}")
    adata = ad.read_h5ad(args.healthy)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    scvi.model.SCVI.setup_anndata(adata, batch_key='dataset_id', layer='counts')

    print(f"Loading warmup template: {args.warmup_dir}")
    model = scvi.model.SCVI.load(str(args.warmup_dir), adata=adata)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    print(f"  epoch: {ckpt.get('epoch', '?')}")
    print(f"  global_step: {ckpt.get('global_step', '?')}")

    new_sd = {k[len('module.'):]: v for k, v in sd.items() if k.startswith('module.')}
    if not new_sd:
        print("  WARNING: no keys with 'module.' prefix, using raw keys")
        new_sd = dict(sd)

    print(f"  State dict: {len(sd)} keys -> {len(new_sd)} after prefix strip")

    missing, unexpected = model.module.load_state_dict(new_sd, strict=False)
    print(f"  missing keys:    {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    if missing:
        print(f"    first missing: {missing[:5]}")
    if unexpected:
        print(f"    first unexpected: {unexpected[:5]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out_dir), overwrite=True)
    print(f"\nSaved: {args.out_dir}")


if __name__ == '__main__':
    main()
