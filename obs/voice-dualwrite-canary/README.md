# trepo-voice-dualwrite-canary

Every 6h (EventBridge rate(6 hours)), detects voice writes in the last 48h whose
per-user, app-visible copy is MISSING — the WRITE_SHARED_ONLY dual-write class:
- shopping: shared_shopping_list voice rows whose household_item_uuid is absent
  from every household member's {member}_new_list
- dishes: shared_dishes voice rows whose _id is absent from {user_id}_dishes
- discards: shared_discards voice rows whose _id is absent from {owner_id}_discards

Emits Trepo/Capture:VoiceDualWriteMiss (Sum, dimension Domain) + logs the missing
rows. On internal DB failure it emits NOTHING and logs loudly (never a 0 that
would mask a real miss). Alarm trepo-capture-voice-dualwrite-miss (metric-math
SUM across domains >=1, notBreaching) -> SNS trepo-capture-alerts -> the triage
lambda auto-diagnoses. Least-privilege role: cloudwatch:PutMetricData
(Trepo/Capture only) + own logs. Node 22, 512MB, 60s. DB creds via standard env.

Validated: 0 misses post-recovery -> injected a shared_dishes voice row with no
per-user copy -> canary reported 1 (metric spiked) -> deleted -> 0 again.

## Fault-injection self-tests

Synthetic validation rows MUST use `_id` (or shopping `household_item_uuid`)
prefixed `canary-fault-test-` (`SELF_TEST_PREFIX`). All three domain checks
exclude that prefix, so a self-test never counts as a real miss or trips the
prod alarm `trepo-capture-voice-dualwrite-miss`. (History: on 2026-07-06 a
self-test row `canary-fault-test-0001` run twice right after the alarm was wired
produced a Sum=2 false alarm — hence this exclusion.) To fault-inject: insert a
row with this prefix, confirm the canary logs it as excluded (0 real misses),
and rely on the unit/local path — do not expect the prod metric to move.
