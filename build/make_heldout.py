import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

BANK = Path('/home/vlad/scGRIP/data/bank')


def parse_args():
    p = argparse.ArgumentParser(description='Отложить случайных доноров из банка в holdout')
    p.add_argument('--input', type=Path, default=BANK / 'healthy_only.h5ad')
    p.add_argument('--out-train', type=Path, default=BANK / 'healthy_train.h5ad')
    p.add_argument('--out-heldout', type=Path, default=BANK / 'healthy_heldout.h5ad')
    p.add_argument('--manifest', type=Path, default=None, help='CSV с разбиением (по умолчанию рядом с --out-heldout)')
    p.add_argument('--dataset', default=None,
                   help='dataset_id[:N] через запятую или "all"; N = число доноров в holdout; без аргумента печатает таблицу')
    p.add_argument('--dataset-col', default='dataset_id')
    p.add_argument('--donor-col', default='donor_id')
    p.add_argument('--frac', type=float, default=0.2, help='доля доноров датасета в holdout (если не задано N)')
    p.add_argument('--n-donors', type=int, default=None, help='число доноров на датасет по умолчанию (вместо --frac)')
    p.add_argument('--min-cells', type=int, default=50, help='донор подходит для holdout, если у него не меньше стольких клеток')
    p.add_argument('--min-keep', type=int, default=2, help='сколько доноров датасета минимум остаётся в обучении')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def prune_categories(x):
    for c in x.obs.columns:
        if isinstance(x.obs[c].dtype, pd.CategoricalDtype):
            x.obs[c] = x.obs[c].cat.remove_unused_categories()
    return x


def main():
    a = parse_args()
    h = ad.read_h5ad(a.input)

    ds = h.obs[a.dataset_col].astype(str).to_numpy()
    dn = h.obs[a.donor_col].astype(str).to_numpy()
    per = pd.DataFrame({'dataset': ds, 'donor': dn}).groupby(['dataset', 'donor']).size().rename('n_cells').reset_index()
    per['eligible'] = per['n_cells'] >= a.min_cells

    if a.dataset is None:
        pd.set_option('display.width', 200)
        tab = per.groupby('dataset').agg(
            cells=('n_cells', 'sum'), donors=('donor', 'size'), eligible_donors=('eligible', 'sum'))
        print(tab.sort_values('cells', ascending=False).to_string())
        return

    for path in (a.out_train, a.out_heldout):
        assert path.resolve() != a.input.resolve(), f'{path} совпадает с входным файлом'
        assert a.overwrite or not path.exists(), f'{path} уже существует, добавьте --overwrite'

    specs = {}
    if a.dataset == 'all':
        specs = {n: None for n in sorted(per['dataset'].unique())}
    else:
        for tok in a.dataset.replace(' ', '').split(','):
            if not tok:
                continue
            n, _, k = tok.partition(':')
            specs[n] = int(k) if k else None
    unknown = set(specs) - set(per['dataset'])
    assert not unknown, f'нет таких датасетов: {sorted(unknown)}'

    rng = np.random.default_rng(a.seed)
    per['split'] = 'train'
    for name, k in specs.items():
        in_ds = per['dataset'] == name
        pool = per.loc[in_ds & per['eligible'], 'donor'].to_numpy()
        if k is None:
            k = a.n_donors
        n_hold = k if k is not None else int(round(a.frac * len(pool)))
        n_hold = min(max(1, n_hold), len(pool))
        if len(pool) == 0 or in_ds.sum() - n_hold < a.min_keep:
            msg = f'{name[:12]}: доноров {in_ds.sum()}, подходящих {len(pool)}, после отбора {n_hold} в обучении останется < {a.min_keep}'
            if a.dataset == 'all':
                print('пропуск ->', msg)
                continue
            raise ValueError(msg)
        picked = rng.choice(pool, size=n_hold, replace=False)
        per.loc[in_ds & per['donor'].isin(picked), 'split'] = 'heldout'
        print(f'{name[:12]}: доноров {in_ds.sum()}, подходящих {len(pool)}, в holdout {n_hold}')

    hp = per[per['split'] == 'heldout']
    assert len(hp), 'ни один донор не отобран'
    held_keys = set(hp['dataset'] + '||' + hp['donor'])
    is_held = pd.Series(ds + '||' + dn).isin(held_keys).to_numpy()

    prune_categories(h[~is_held].copy()).write_h5ad(a.out_train)
    prune_categories(h[is_held].copy()).write_h5ad(a.out_heldout)
    manifest = a.manifest or a.out_heldout.with_suffix('.donors.csv')
    per.to_csv(manifest, index=False)

    print(f'\ntrain:   {(~is_held).sum():>8,} клеток -> {a.out_train}')
    print(f'heldout: {is_held.sum():>8,} клеток -> {a.out_heldout}')
    print(f'manifest: {manifest}')


if __name__ == '__main__':
    main()
