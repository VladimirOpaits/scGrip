---
name: sql-flag-oracle-differences
description: "When giving SQL for the gestión-de-datos exam, flag where Oracle syntax differs from the local Postgres"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 506da36b-625b-49b9-b619-f9185b7b5811
---

When writing SQL during exam prep (gestión de datos), Vladimir works on a local Postgres playground (`psql -d playground`) but his exam material is **Oracle**-flavored (the source files use `varchar2`, `number`; CSV migration exercises). He asked: whenever an Oracle command would differ significantly from the Postgres one we just used, **call out the Oracle variant explicitly**.

**Why:** he practices on Postgres but is graded on Oracle; silent dialect gaps (types, ALTER syntax, identifier case-folding, date/time handling) would burn him on the exam.

**How to apply:** give the working Postgres statement first, then a short "Oracle:" note with the diff. Common gaps to watch: `varchar2`/`number` vs `varchar`/`numeric`/`integer`; `ALTER TABLE t ADD (col TYPE DEFAULT ...)` (Oracle, parens, no `COLUMN`) vs `ADD COLUMN`; unquoted identifiers fold to UPPER in Oracle vs lower in Postgres; `to_date`/`interval`; `DUAL`; sequences/`ROWNUM` vs `SERIAL`/`LIMIT`. Don't dump the diff when dialects match — only when it materially differs. Related: [[project-direction]].
