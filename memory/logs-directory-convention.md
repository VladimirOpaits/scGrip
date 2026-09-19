---
name: logs-directory-convention
description: "In scEris, run logs belong in logs/ at the repo root, not scattered in the working directory -- Vlad moves them there, so write them there in the first place."
metadata:
  node_type: memory
  type: feedback
---

2026-08-29. Background-job logs in the scEris repo go in **`logs/`** at the repo root.

I had been writing them to the repo root (`rerun_c3.log`, `paper_numbers.log`,
`daseq_final.log`, ...). Vlad moved the whole set into `logs/` and said "логи в папке с
логами". A log I was tailing appeared to vanish mid-run because of that move, not
because the job died.

**Why:** the repo root is for code and config; dozens of `*.log` files there make
`ls` useless and they are not artifacts anyone wants to see alongside `build/` and
`scenarios/`.

**How to apply:** redirect every long-running job to `logs/<name>.log` from the start:
`setsid nohup python build/... > logs/<name>.log 2>&1 < /dev/null & disown`.
When a log you were tailing disappears, check `logs/` before concluding the job died --
and check the process itself (`ps -eo pid,etime,cmd | grep '[m]y_script'`) rather than
the log's absence.

Related: [[workflow-notebook]] (scripts live in their own directories, run from repo root).
