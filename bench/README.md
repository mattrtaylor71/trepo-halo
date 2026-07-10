# trepo model-upgrade benchmark

Local harness that compares the **currently deployed** OpenAI models against
**upgrade candidates** on 4 money-path surfaces, using **real data**. Pure API
calls — **NO prod changes, NO env flips, NO deploys.** Matt picks upgrades from
the evidence in `REPORT.md`.

## Keys
Read at runtime from `/tmp/bench_secrets/{openai,anthropic}.key` (extracted from
Lambda env — never committed, never printed). Set them before running:
```
mkdir -p /tmp/bench_secrets   # then drop openai.key + anthropic.key
```

## Surfaces
1. `01_thyme.mjs`   — Thyme tool discipline + quality (real system prompt + 48 tools)
2. `02_recipes.mjs` — Recipe generation (real kitchen inventories, blind opus judge)
3. `03_identify.mjs`— Fast photo identify (real S3 capture images, DB ground truth)
4. `04_home.mjs`    — Home suggestions (real context, blind opus judge)

Raw per-call outputs land in `out/*.json`; fixtures (real data snapshots) in `fixtures/`.
