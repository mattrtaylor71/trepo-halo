# Household brain verification — October 6, 2026

Author: Codex for Matt Taylor. Status: local qualification passed; not deployed.

## Results

- **59 Python tests passed**, including the new MySQL/service/projection/package
  tests and the existing Family Brain projection regressions.
- **117 Thyme tests passed**, including its existing 113 checks and four new
  shared-brain/management/recovery contracts. No skipped tests in these runs.
- Python compilation, JavaScript syntax checks and Git whitespace checks passed.
- AWS CloudFormation `ValidateTemplate` accepted the deployment template and
  reported `CAPABILITY_IAM`. Validation did not create or update a stack.
- The deterministic standalone service zip built and imported successfully.
- Minimal Home, native Thyme V2 and legacy Quick Ack overlays built against
  freshly downloaded serving packages. All target baseline hashes matched.
  The native candidate contains its required Lambda SDK dependency. No function
  code, environment, IAM role, live database or customer record was changed.

MySQL tests used a dedicated **127.0.0.1:33826 MySQL 8.4.11** instance, synthetic
households, and a fresh randomly named database per test. The instance was stopped
after qualification. Neither production credentials nor customer data were used
by these tests.

## Meaningful cases exercised

| Area | Evidence |
|---|---|
| Atomic capture | Rolled-back domain writes leave no revision/evidence; committed writes invalidate old contexts |
| Current-source correction | Save/unsave/edit/undo remove obsolete contributions on the next read |
| Privacy | Another member's saved library is absent; shared consumers receive shared context; foreign records are excluded |
| Membership | Departures revoke old contexts, remove shared service contributions and clean up audience caches |
| Internal controls | Optimistic revisions, lost-response retry, changed-payload conflict, audit, explicit scope, household/personal pause and reset |
| Purchase meaning | List checks and undated receipt imports never invent purchases/cadence; cadence requires three distinct dated purchase days |
| Time | Receipt dates retain the household's calendar date; future evidence is ignored; expired dates are marked for checking |
| Reversal | A reversal received before its original event prevents that event from contributing later |
| Concurrency | A second real MySQL connection commits during projection; the old snapshot is rejected and rebuilt |
| Failure recovery | Missing/oversized sources reject stale serving; expired caches repair; worker failures do not starve subsequent households |
| Boundaries | Restrictions remain separate/current; active caller transactions are never committed by a brain read |
| Agent recovery | A resumed Thyme answer validates its original brain boundary after reset, rather than substituting a newer fence |
| Rollback compatibility | Legacy projection tests pass with V2 off; V2 signed entry point rejects forged identity and customer memory controls |
| Packaging | Deterministic output, unrelated app fixes preserved, mismatched baseline and changed target files rejected |

## Review findings resolved

The implementation was tightened during testing for empty MySQL result types,
receipt import timestamps versus purchase dates, receipt timezone handling,
per-member pause/reset persistence, leaving-member service evidence, exhausted
households in the worker queue, and validation of recovered model requests. The
template's initial missing closing brace was repaired before AWS validation.

## Practical limits

This is not a live rollout or an end-to-end purchase test. The source limits,
production indexes, trigger permissions/definer, write contention, VPC/TLS secret,
IAM grants and real latency still need staged rollout qualification. No externally
hosted model or purchase provider was called by the tests. CloudFormation syntax
validation does not prove those operational prerequisites.

The separately running Shop repository has not adopted the shared adapter. Its
selection producer, verified order producer and recipe-feedback producer must
commit durable, correctly scoped receipts before those service events are live.
Their receiver contract is tested, but this report does not claim end-to-end
producer delivery. Legacy per-user-only database writes are not covered by the
canonical trigger set.

The live founder brain remains the serving system until migration and activation.
Review/rollout instructions and the complete action matrix are in `README.md`.

## Reproduce

From the backend repository, using a Python 3.12 environment with pytest and the
pinned PyMySQL dependency, and an isolated localhost MySQL instance on 33826:

```sh
HOUSEHOLD_BRAIN_TEST_MYSQL=1 PYTHONPATH=playground6_groceryRec python -m pytest \
  playground6_groceryRec/household_brain/tests \
  playground6_groceryRec/family_brain/test_core.py -q
```

Run `npm ci --ignore-scripts` and `npm test` in `thyme-agent-v2` for the consumer
regressions. With the MySQL opt-in absent, the SQL suite is skipped; that does not
qualify a release. Never point these synthetic fixtures at a shared database.

Build without deployment:

```sh
PYTHONPATH=playground6_groceryRec python -m household_brain.build \
  --output /private/output/household-brain-service.zip
```

For a minimal overlay, also supply `--kind python`, `--kind thyme-v2` or
`--kind quick-ack`, `--base-zip`, and its exact hexadecimal `--expected-sha256`.
At deployment, compare the current Lambda revision/code hash to the captured
serving receipt again; the iOS/backend agent may have shipped newer work meanwhile.
