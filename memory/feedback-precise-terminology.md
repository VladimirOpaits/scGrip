---
name: feedback-precise-terminology
description: "Vlad requires exact, defined terminology at all times — no vague or undefined lexicon, especially for technical terms in the lupus MIL pipeline"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: eb2fb708-8d4a-4796-b609-724e6e330ac3
  modified: 2026-09-02T22:06:28.147Z
---

Always use precise, checkable definitions for technical terms (e.g. "pool", "mixed pool", "stratum") — never fall back on vague or undefined lexicon ("группа", "какая-то смесь") when a term has a specific operational meaning in the code or data.

**Why**: caught directly — asked "что именно за смешанный пул? не используй неопределенный лексикон, всегда будь точен" after an explanation used "pool"/"mixed" without spelling out the exact set-membership condition behind them.

**How to apply**: when introducing or using a domain term, state the precise construction (what groupby/mask/condition defines it) inline or point to the memory/code that defines it, rather than assuming shared informal understanding. See [[lupus-donor-pool-definition]] for the concrete case this came from.
