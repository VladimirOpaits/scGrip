---
name: healthy-reference-mmd-invariant
description: "Invariant of the current reference-mapping method: replay is anchor-only; MMD aligns new healthy query cells to an independent healthy Census metric reference."
---

# Healthy-reference MMD: invariant метода

## Нельзя путать роли Census-клеток

| Набор | Роль | Может быть стороной MMD? |
| --- | --- | --- |
| replay | Anchor: сохраняет старую геометрию модели | **Нет** |
| eval_old | Held-out проверка сохранения reference geometry | **Нет** |
| metric reference | Фиксированная независимая выборка здоровых Census | **Да, только как здоровый target MMD** |
| healthy A новой когорты | Обучающая сторона healthy-to-healthy MMD | Да |
| healthy B новой когорты | Held-out проверка alignment; ELBO без MMD | Нет |
| disease новой когорты | Только post-training encoding/evaluation | Нет |

metric reference, replay и eval_old обязаны быть попарно непересекающимися по
клеткам. metric reference выбирается с фиксированным seed и не зависит от размера
replay или экспериментального seed.

## Цель

MMD реализует MapQC-подобное предположение: если healthy query сближается с independent
healthy reference при фиксированных coarse cell compartments, доступная адаптеру свобода
тратится на устранение технического cohort/batch-сдвига в mapping counts → latent.
Replay-anchor одновременно не даёт этой адаптации деформировать уже выученное Census
пространство.

В текущем эксперименте ELBO считается только на healthy A и healthy B; MMD — только
на healthy A. Больные клетки не входят ни в один loss и не дают градиента. После обучения
то же преобразование применяется к больным клеткам. Их residual displacement от healthy
reference может отражать disease/state signal и служить кандидатом для последующих
проверок. Это не разрешает оптимизировать MMD по больным и не доказывает, что любой
остаток уже является биомаркером.

## Минимальные проверки в коде

1. Перед началом обучения assert-проверить попарочную непересекаемость metric_idx,
   replay_idx, eval_idx.
2. Лог loss называть явно: mmd_healthy_A_to_metric_reference, не mmd_ref.
3. Сопоставлять compartments между healthy-A и metric reference; не использовать
   replay_ct в MMD helper.
4. В CSV сохранять mmd_reference="independent_census_metric_reference".
