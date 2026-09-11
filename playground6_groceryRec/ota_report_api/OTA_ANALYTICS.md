# OTA report analytics deployment baseline

The canonical `app.py` and `analytics.py` are the exact two modules deployed and
verified with the raw numeric correction on 2026-09-11 UTC. The baseline preserves the already deployed September 8
D3/H4 diagnostic validator and its exclusion of retained records from current
state. The former local `app.py` lacked that validator and must not be restored.

Function: `trepo-grocery-backend-dev-OtaReportApiFunction-cyUvBswGXwDb`,
`us-east-1`, account `566667681926`, handler `app.handler`, runtime Python 3.9.
The deployment ZIP SHA-256 (base64) is
`iTXcTy8iFKIwf6ncnYda8/FmW7q4624gtT5wjUDt4VY=`.

## Subsequent code deployment

Use the narrow helper from this directory. It packages only `app.py` and
`analytics.py`. The default invocation reads the live configuration and reports
hashes without a mutation. It refuses a missing analytics secret and preserves
the complete existing environment without calling UpdateFunctionConfiguration.

```sh
python3 deploy_analytics.py --package-only
python3 deploy_analytics.py --profile trepo
# After reviewing the preview and local tests:
python3 deploy_analytics.py --profile trepo --deploy --expected-code-sha '<reviewed current base64 CodeSha256>'
```

The code update uses the current RevisionId and checks the resulting code,
environment, and other configuration. It never changes tables, routes, IAM,
retention, or secrets; it does not automatically roll back concurrent changes.
AWS CLI credentials are read through the selected boto3 profile, never printed.

**Do not deploy this module through the current broad SAM template unchanged.**
The analytics key was merged into the live Lambda environment separately. A
future full SAM deployment must explicitly preserve/provision the existing
`ANALYTICS_READ_KEY` and `ANALYTICS_RETENTION_DAYS` settings first. An empty default
for the key would disable analytics. The dashboard uses the same secret only in
its server runtime. Do not place it in code, browser bundles, documentation,
command arguments, or commit history. Any separately reviewed environment update
must merge the complete fresh existing Variables map with only authorized
additions and use its current RevisionId; never replace unrelated variables.

## Raw numeric validation regression

New reports preserve the original raw numeric validity for analytics before
legacy normalization coerces values. Boolean counters, fractional numeric values,
and out-of-bounds measurements are omitted from analytics; valid Boolean flags
and integer strings remain supported. Legacy stored values and reads retain their
previous behavior. The server-derived rejection metadata also protects event
history and a latest-state fallback if its optional analytics snapshot fails.

Historical rows that already lost their raw types cannot be repaired accurately
from the stored integer alone. They remain unchanged; no historical migration or
deletions were performed. The D3/H4 validator and exclusion from current state
remain intact.

Run the focused actual-handler regression suite from the repository root:

```sh
python3 -m unittest discover -s ota_report_api/tests -v
```

These 13 tests use in-memory tables and dummy process-only credentials, with no
network requests. The full 48-test candidate suite, baseline failing reproducer,
reviewed ZIP/diff, code-only deployment, and post-deployment live QA cleanup are
archived under:
`/Users/MattTaylor/halo-device-analytics-2026-09-10/simulation-2026-09-11/backend/candidate/`.

## API and evidence

Existing POST `/ota/report` and legacy GET routes remain compatible. Authenticated
views use `X-Analytics-Key` (at least 32 characters):

- `/ota/report/latest?view=analytics&limit=50`
- `/ota/report/events?view=analytics&device_id=halo-example&limit=50`

The analytics projection excludes owner identifiers, network addresses, SSIDs,
credentials, raw media, transcripts, and raw diagnostic blobs. Existing legacy
raw routes retain their previous access behavior. Page limits are bounded and
cursors signed; loaded-page receipt ordering is not global chronological ordering.
Age uses server receipt; missing/invalid receipt stays unknown. List counters are
cumulative only within the reported boot. Cached Sense observations of LCD
firmware do not establish independent LCD presence.

Both DynamoDB tables had TTL disabled at review. No expiry attributes or TTL
settings were changed. The configurable 90-day default analytics event window is
a read filter, not deletion or a change to historical storage retention.

Private/local implementation and evidence directory:
`/Users/MattTaylor/halo-device-analytics-2026-09-10/backend/`

- `contract.json`: exact field names, bounds, enums, pagination, and semantics.
- `TEST-RESULT.json`: 35 offline tests, including unchanged deployed validator AST.
- `DEPLOYMENT-CANDIDATE.json` and `DEPLOYMENT-RESULT.json`: exact code/config baseline
  and root-run deployment verification.
- `live-ingest-smoke/RESULT.json`: synthetic ingestion, boot-scoped counters,
  receipt-time independence, and exact fixture cleanup. This is not device proof.
- `deployment/TTL-OBSERVED.json`: read-only existing retention observation.
- `original/ota_report_api/app.py`: older local source archive, **not a rollback target**.
- `deployment/deployed-code.zip` and `deployment/deployed-function.private.json`:
  pre-analytics live rollback artifacts, including existing D3/H4. The private
  configuration includes secrets and must never be committed or distributed.
  Rollback requires reviewing current code/environment and a new RevisionId;
  preserve the environment rather than replaying a stale secret snapshot.

No broad template patch or unrelated repository changes were adopted with these
modules. No changes were staged or committed by the adoption step.
