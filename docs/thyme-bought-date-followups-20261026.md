# Thyme bought-date follow-up repair — October 26, 2026

Status: deployed. Matt reported the screenshot at 16:18 Pacific while testing the explicitly authorized bought-date rollout. This is a backend repair; no app binary changed.

## Confirmed failure

The exact serving logs show the first request, “Hey I bought that celery on sept 28”, received an unnecessary year question without a tool call. The next request, “2026”, produced the correct `update_item_details(item_name="Celery", bought_date="2026-09-28")` arguments. The mutation guard rejected it with HTTP 409 because the current utterance did not repeat the item name. Final outcome formatting replaced that actionable error with “Some requested changes couldn't be completed.” No date had been saved for that item.

## Change

- Named month/day dates accept full and abbreviated months, with or without an explicit year. An omitted year resolves to the most recent valid occurrence on or before the household's current calendar day, including year rollover and February 29. Ambiguous numeric dates, vague dates, invalid days and explicitly future dates remain rejected. Prompts now tell the model not to demand a year for “Sept 28”.
- A bare year may complete only the adjacent server-loaded user request and assistant year question. The original user words supply the item and date. It never searches older turns past another request or cancellation, never trusts an assistant-invented item, and permits only the bought-date mutation. The server fixes the confirmed year rather than accepting a different model date. Existing household, duplicate-target, revision, idempotency and durable-result checks remain in place.
- “Hey I bought…” now counts as an explicit kitchen correction. If target context genuinely cannot be recovered, the known validation error produces an actionable message instead of a generic failure. Arbitrary service/database errors are not exposed.

## Verification and recovery

The pre-fix parser reproduced rejection of `Sept 28`. Fourteen focused legacy date tests and two pilot canonical-edit tests passed, including the exact year-only follow-up, duplicate target protection, unrelated-field rejection, cancellations, date boundaries and persisted-result verification. All six actual candidate handler packages imported successfully.

Live tests used one uniquely named item in the documented review account. The first request used “Hey I bought that [item] on sept 28” and saved September 28 directly. A separate isolated session was seeded with the already-issued year clarification to reproduce the reported “2026” reply; it also confirmed and saved September 28. The saved date was reset between cases, so the second test proved an actual change. Both emitted the successful legacy refresh event. Canonical check-in time stayed unchanged; build 205 received 28 days old. The exact fixture and the two seeded test messages were removed and absence verified.

The specific reported celery was then repaired through the canonical revision-checked API, only after verifying its exact ID, owner, title, original check-in and absence of a competing later date correction. Its saved bought date is September 28, 2026; the legacy app response returns 28 days old. Quantity, storage, opened state, expiration and original check-in were preserved. Private transaction evidence is retained outside Git.

An additional local brain-confirmation suite could not run correctly because its fixture lacks the authenticated consent context required by the real provider guard (14 cases failed at that guard). The initial local attempts also needed the Lambda import/dependency paths mapped. No production consent guard was bypassed or changed. The 10 action-claim and three legacy-refresh checks in the same run passed. This repair does not claim the broader brain suite passed; the two live date journeys used the actual authenticated, consent-gated deployed endpoint.

## Deployment

Six serving packages were patched from fresh captured ZIPs, retaining independent deployed changes: `trepo-quick-ack-stream-dev`, `trepo-quick-ack-sam-dev`, `trepo-quick-ack-async-worker-dev`, and the native/API/worker variants of `trepo-thyme-agent-v2-pilot`. Only existing affected modules were changed. The locally updated future runtime policy was absent from the serving ZIPs and was not introduced as part of this hotfix. The previously deployed old-app age bridge and refresh signal remain enabled.

Private captures, exact candidate manifests, revision-guarded deployment receipts, live test responses, the reported-item repair receipt and final code/configuration readbacks: `/Users/MattTaylor/Library/Caches/trepo-thyme-celery-20261026/`. These configuration files contain credentials; do not print or commit them. `scripts/prepare_thyme_date_followup_rollout.py` reproduces the scoped package edits from those captures without deploying them.

Company context: `apps/trepo/screens/kitchen-and-capture`. No shared-brain save requested.
