# trepo-voice-dualwrite-canary

**Hourly** (EventBridge `rate(1 hour)`, rule `trepo-voice-dualwrite-canary-1h`), detects
voice writes in the last 48h whose per-user, app-visible copy is MISSING — the
WRITE_SHARED_ONLY dual-write class — and **auto-heals** them:
- shopping: `shared_shopping_list` voice rows whose `household_item_uuid` is absent
  from every household member's `{member}_new_list`
- dishes: `shared_dishes` voice rows whose `_id` is absent from `{user_id}_dishes`
- discards: `shared_discards` voice rows whose `_id` is absent from `{owner_id}_discards`

## Detect + auto-heal

When a miss is found, the authoritative **shared** row is copied (idempotent
INSERT-if-absent by key; common columns only, so original values + timestamps are
preserved; `owner_id` rewritten to the member id) into the missing per-user/member
table(s). This makes the system self-heal the rare "silent-success lost write"
runtime anomaly (see the 2026-07-11 incident) without taxing every voice write.

## Metrics (namespace `Trepo/Capture`, dimension `Domain`)

| Metric | Meaning |
|---|---|
| `VoiceDualWriteMiss` | misses **detected** this run (observability / dashboards) |
| `VoiceDualWriteReconciled` | misses **auto-healed** this run |
| `VoiceDualWriteUnhealed` | misses the heal did **not** fix, OR a **stale** miss whose shared row is older than `STALE_THRESHOLD_HOURS` (2h) — it survived a prior hourly heal, so it's reappearing/persistent **corruption**, not the transient anomaly |

Each auto-heal is logged loudly: `{evt:"voice_dualwrite_autohealed", domain, rows, count}`.

**Alarm `trepo-capture-voice-dualwrite-miss`** now fires on
`SUM(VoiceDualWriteUnhealed across domains) >= 1` over 1h (`notBreaching`) →
SNS `trepo-capture-alerts` → triage lambda. **So Matt is paged only when
self-healing FAILS — not when it works.** (Previously it fired on raw
`VoiceDualWriteMiss`, i.e. every transient miss.)

On INTERNAL DB/emit failure the canary emits NOTHING and logs loudly (never a 0
that would mask a real miss). Least-privilege role: `cloudwatch:PutMetricData`
(`Trepo/Capture` only) + own logs. Node 22, 512MB.

## Fault-injection self-tests

Two distinct patterns:

1. **Detection path** — synthetic rows prefixed `canary-fault-test-`
   (`SELF_TEST_PREFIX`). All three checks EXCLUDE that prefix, so a self-test never
   counts as a real miss or trips the prod alarm. (History: on 2026-07-06 a self-test
   row run twice right after the alarm was wired produced a Sum=2 false alarm — hence
   the exclusion.)

2. **Auto-heal path** — `faultInject.mjs` injects a **NON**-prefixed `shared_dishes`
   voice row for the **SAFE test user** with no per-user copy (a row the canary treats
   as a REAL miss), so you can validate healing end-to-end:
   ```
   node faultInject.mjs inject     # create the synthetic miss (SAFE user only)
   # invoke the canary → expect {detected:1, reconciled:1, unhealed:0}
   node faultInject.mjs verify     # per-user copy present (healed) ?
   node faultInject.mjs cleanup    # remove shared row + healed copy
   ```
   Validated 2026-07-11: inject → canary `reconciled:1 unhealed:0`, `_createdDate`
   preserved, `voice_dualwrite_autohealed` logged, no alarm; cleanup → next run all zero.

## Deploy

Standalone Lambda (not SAM). Bundle + ship:
```
zip -qr /tmp/canary.zip index.mjs package.json node_modules
aws lambda update-function-code --function-name trepo-voice-dualwrite-canary --zip-file fileb:///tmp/canary.zip
```
