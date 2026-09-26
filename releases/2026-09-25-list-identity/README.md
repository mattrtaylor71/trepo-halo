# Household shopping-list identity repair

Issue: TR-1790374208533179. Matt requested commit and deployment in the parent Codex conversation on September 25, 2026. This authorizes this release; the triage system still requires founder approval for future releases.

Two different household groceries can select mirrored database rows with the same local numeric primary key. Clients then render duplicate identities and can select the wrong item when checking a row. Public `id` now uses the existing household UUID, matching `itemUUID`. UUID-less legacy rows retain their original numeric string fallback. Request formats, membership resolution, SQL mutation logic and response fields are unchanged.

## Exact release boundary

The package replaces only `listHandler.js` in the independently downloaded serving archive. It does not ship the private native test transport, dependencies, environment, IAM, runtime, layers or schema changes. The preceding reconciliation commit records already-serving canonical item responses for deduplicated adds; that behavior is preserved, not newly deployed here.

Function: `trepo-list-handler`, region `us-east-1`.
Serving revision: `0979490f-c6d0-4f57-a6db-1440ab896566`.
Serving ZIP SHA-256: `18b9906b3a0b2393d3f397b8a7c1a2cf157cf117779e5164fd544e76c16063df`.
Candidate ZIP SHA-256: `618fd5c6b61b53b0134d94bdb8712ecceba3128678a06951625401ed2ef573be`.
Candidate handler SHA-256: `e514e1f6267b52eeaf76f29594da3c19f524ce287580f9d3b5bb03cb13612ec5`.

The current GitHub default branch is an older, separate application tree without this handler. This release is based on the existing backend release lineage `e375af1`, on `codex/triage-list-release-20260925`; it must not merge the whole release history into the unrelated default branch as part of this one-line repair.

## Verified tests

- 8 integration checks against actual baseline/candidate handlers and isolated MySQL: reproduced duplicate ID/wrong target, stable unique identity, mirrored check and uncheck persistence, untouched sibling and foreign household, winner changes, single-table behavior, UUID-less legacy fallback, unknown UUID 404.
- 2 compiled historical/current Swift model contracts accepted the candidate and selected the intended item. Android string-ID and itemUUID usage were inspected, not UI-tested.
- Native Release simulator test `20260925-153736-0970b3`: 1/1 passed, checked/un-checked the intended item with SQL verification and cold relaunches. All 484 native inputs match the manifest. Screenshots inspected in the original run.
- Baseline native duplicate rendering reproduced. Failure collection stalled and was interrupted; this is not counted as a passing test.
- Release rerun: all 8 SQL checks passed again on the Mac mini, plus 2 real-SQL fresh-add/deduplicated-add identity and persistence cases. Only asynchronous cloud categorization dispatch was stubbed; SQL and handler behavior were real.
- 7 new dependency-free regression cases run directly against the actual source mapper: `node --test releases/2026-09-25-list-identity/identity.test.cjs`.
- JavaScript syntax and Git whitespace checks pass.

Private original reports, test inputs, screenshots, baseline/candidate ZIPs and rollback artifact are retained under the issue's triage work/evidence directories. Production customer records were not edited as test fixtures. Physical devices, Android UI and every old app screen are not certified by these checks.

## Deployment and rollback

The release operator must compare the current Lambda revision/code hash, verify the committed handler and candidate ZIP hashes, then use the current `RevisionId` for an optimistic update. Keep the original ZIP. Independently read back active/successful state, code hash and configuration; preserve a sanitized receipt here. Code-only rollback uses the original ZIP and the then-current revision, after checking for concurrent changes.

Released at 2026-09-26 04:54:53 UTC (September 25 Pacific). See deployment.json and live-canary.json. Independent Lambda readback confirmed Active/Successful, the exact candidate hash and unchanged configuration. The affected household live read returned 20 items with 20 unique canonical string IDs; the former numeric mapping still demonstrated one collision. No customer write requests were used. Code commit: 3c2877d5ff2154b1703c9cd071c596b09c3ef75c. This documentation commit does not change serving code.

Shared context consulted: `apps/trepo/architecture`. No customer-facing design changed and no brand exception applies.
