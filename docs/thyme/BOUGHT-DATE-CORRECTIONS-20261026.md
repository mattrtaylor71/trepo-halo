# Thyme bought-date corrections

Date: October 26, 2026. Author: Codex for Matt. Status: deployed and verified.

## Behavior

Users can ask Thyme to change an existing kitchen item's bought/acquired date, for example:

- “Change the bought date for the milk to October 19, 2026.”
- “Change the eggs' bought date to yesterday.”
- “Clear the bought date for the milk. I don't know when I bought it.”

The existing `update_item_details` tool accepts `bought_date`. Legacy app Thyme applies the requested edit; the new pilot retains its existing visible approval boundary. A tool cannot report success until the canonical Kitchen API returns the exact saved date. Thyme can read the corrected date afterward.

Relative dates resolve in the household's pinned shelf calendar, with the existing America/Los_Angeles fallback for an unseeded household. Supported phrases include today, yesterday, N days/weeks ago and last weekday; fully specified named and ISO calendar dates also work. Vague dates, ambiguous numeric formats, invalid dates and future dates require clarification. Null means explicitly unknown, never zero days old.

Fresh named corrections resolve against current household membership, including items added by a partner. Model-supplied item IDs cannot bypass duplicate-name clarification. For a duplicate, retain the normalized date, bounded candidates and their revisions; a follow-up such as “the oldest one” selects exactly one candidate. Preserve the request across retries so a midnight retry does not reinterpret yesterday. Concurrent changes fail rather than overwrite a newer item revision.

## Storage and boundaries

Thyme sends `kind:edit`, `fields:{}` and `shelf_dates:{acquired_on:YYYY-MM-DD|null}` through the existing amount-operation endpoint. The protected `kitchen_item_edits` sidecar stores `__acquired_on`; the API exposes `acquired_on`, and Thyme maps it to `bought_date`. The operation journals the date correction and marks the shelf projection for refresh. Existing inventory update triggers handle household-brain invalidation.

This changes neither `_createdDate` nor expiration, quantity, storage, opening history or receipt transaction evidence. The canonical backend now accepts date-only edits and generates valid SQL for an empty ordinary-field patch. Bought-date-only corrections are available outside the shelf pilot; opened/frozen/cooked date edits retain their existing gate. No schema or feature-flag changes, notification rollout, model change or new paid service.

**UI limitation:** the existing native “days old” calculation still uses `_createdDate`. This backend release does not add a date picker or fix that label. The shelf lifecycle respects acquired dates where that engine is enabled; the wider lifecycle/notification rollout remains unchanged. Corrected item dates and original receipt purchase dates remain distinct evidence.

## Verification

- 28 targeted legacy/shared regression tests, 140 pilot regression tests and six canonical date tests passed.
- Exact prepared Lambda packages successfully imported their configured Node handlers; Python overlays compiled.
- Eight real-database checks in an isolated synthetic household: a partner's item, exact saved value, preservation of unrelated fields, idempotent replay, stale revision rejection, foreign-scope rejection, retained pilot gate for other dates and explicit unknown.
- Authenticated live legacy stream and synchronous requests exercised exact dates, yesterday, duplicate-name clarification, oldest-item continuation, reading the saved date, rejecting a future date and clearing an unknown date. API readback verified the selected row and that the other duplicate was unchanged.
- The exact new-pilot package's gateway was exercised against the live canonical API using only the synthetic household. Catalog validation, normalization before approval, partner-item editing, verified readback and idempotent retry passed. Test enrollment existed only in the local harness; deployed pilot enrollment was unchanged. This does not claim a new simulator or full native approval-UI test.

The live partner-item test exposed an older rename/correction selector that searched only the speaker's rows despite household-wide read context. The correction selector now re-resolves current household rows. Displayed confirmations now include the authoritative bought date; clarification requests are presented as questions rather than generic failed-update text.

## Release and source provenance

`scripts/deploy_thyme_bought_date_overlay.py` applies only the changed modules to each captured serving ZIP, preserving service-specific differences. It checks live code hash/revision immediately before update and verifies successful deployment with unchanged environment, role, handler, runtime, layers, VPC, memory and timeout. Captured baselines are retained for guarded rollback; do not overwrite a later release.

Targets: Kitchen API; legacy stream, synchronous and async worker; new Thyme pilot API, native adapter and worker. No HomeSuggestions, iOS binary, notification or unrelated service deployment.

Private evidence: `/Users/MattTaylor/Library/Caches/trepo-thyme-bought-date-20261026/`. Config captures contain credentials and must stay outside Git. Relevant shared context consulted: `apps/trepo/screens/kitchen-and-capture`, `brand/trepo-brand-guidelines`; no brand exception or new UI. Shared memory was not modified.

Cleanup verified: the two synthetic accounts and three kitchen items were removed, the QA brain was disabled, and active-fixture readback was empty. Test audit traces remain for diagnosis. No customer inventory was changed.
