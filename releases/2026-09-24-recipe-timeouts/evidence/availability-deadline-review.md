# Optional availability timer fix — v3

Prepared September 24, 2026 (America/Los_Angeles). Source frozen and locally qualified; no cloud deployment or commit by this agent. V2 source, package and evidence remain unchanged.

## Reproduction

The parent identified that `_compute_recipe_availability` runs `match_recipes_fast` inside a main-thread deadline while that library submits even a single recipe to a `ThreadPoolExecutor`. When the alarm unwinds the executor, its `shutdown(wait=True)` waits for the still-running provider. This invalidated the claim that optional availability always stayed within its provider budget.

`reproduction-v2.json` records an actual-library reproduction: a 40 ms work budget returned after **312.7 ms**; the provider ran on another thread and completed after the deadline. The same problem applies to a substitution provider invoked by that worker. The timeout cannot safely abandon the worker, because it would leave work running after the response.

## Minimal correction

`app.py` opts into `inline_single=True` only for the optional single-recipe lookup already inside the provider deadline. `recipe_inventory_llm.py` accepts that keyword and executes its existing compact call/normalization/merge path directly on the caller's thread. An accidental multi-recipe inline call fails before a provider starts.

Default single- and multi-recipe callers keep the existing executor behavior. Prompt arguments, model selection, deterministic matching, substitution semantics, merging, quantity checks, metadata, and public HTTP responses are unchanged. The timeout helper itself is unchanged. No deployment configuration, binary, layer, IAM or schema changes are needed.

Because the synchronous provider now runs under the active signal boundary, timeout unwinds that provider directly before the app returns the existing deterministic fallback. It does not launch or abandon an extraction worker.

## Verification

- `availability-before.txt`: both actual-library threaded deadline regressions fail on v2; the already-inline deterministic-only control passes.
- `availability-after-final.txt`: **8 new cases pass**. They cover slow matching and substitution providers, no completion after returning, no surviving worker, visible invocation context, equal prompts/results between inline and existing execution, unchanged one/two-recipe default executor behavior, invalid bulk-inline rejection, and provider failure fallback.
- `focused-final.txt`: all previous 235 cases plus those 8: **243 passed, 0 failed, 0 skipped**. Real SQL/Dynamo, media limits, legacy receipts, explicit recovery, edit preservation, readiness, actual SDK behavior and audio selector cases remain included.
- Host tests ran on Python 3.8; candidate files passed Python 3.9 syntax checks. Source test inputs are byte-identical to ZIP application entries.
- `availability_runtime_probe.py` is an **unshipped** standalone Lambda handler with no network or database calls. It exercises the actual app and matching library with a sleeping provider, verifies main-thread execution, deterministic response, no late work and timer/context restoration. Local execution stopped at 67 ms under a 60 ms work budget. Parent owns its exact Python 3.9 cloud qualification.

## Artifacts

- Agent ZIP CodeSha256: `2Ayfml12YXGNBK5DqyoTFoNbvv5AB7z/NBePO30K+RE=`.
- `app.py`: `7ec3a867cafaabbe86e4fcae1120e1f2db941e10f75e6a700ee4ba9bee1a625d`.
- `recipe_inventory_llm.py`: `b74ec0b075e265d01abdf5a60d61030537a66d6e9dd7cb9bb73e056f8266a03d`.
- Unchanged helper: `77432c2fbb608ec5fb1daa8e90bb37291e2af0c1b53b342e95e58eafd1161116`.
- `manifest.json` records source/package hashes and counts; `recipes-v3.patch` is relative to frozen v2.

Only the app and matching library change from v2; all other ZIP entries are byte-identical. The runtime probe is excluded. Parent may use different ZIP metadata while preserving application bytes.

## Cleanup and remaining gates

Owned DynamoDB Local PID 69015 on port 38001 was restarted with telemetry disabled, then stopped after verification with zero remaining fixture tables. MySQL33317 and the parent's native fixture schema remain running and unmodified. See `cleanup.json`.

Exact cloud-runtime requalification, final review/archive/commit/deployment, live canary and native Release qualification remain parent-owned. No canonical source, initial release manifest, production record or serving function was changed by this agent.
