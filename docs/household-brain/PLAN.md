# Household brain backend — October 6, 2026

Author: Codex for Matt Taylor. Status: backend implemented and tested, not deployed.
Authorization: Matt requested building the unified household brain, managed by
Trepo internally, without customer-facing controls. Company context:
`trepo-company/apps/trepo/architecture` and the September 25 founder implementation.

## Architecture decisions

Reuse MySQL, current dietary records and the existing personal food-preference
tables. Add one logical household state and audience-specific prepared contexts.
No model call is needed to record an action or rebuild a projection. The shopping
pilot's deterministic product evidence rules inform the shared projection; its
copied inventory and demo identity never become production authority.

Committed canonical SQL changes mark their household dirty in the same transaction
through additive triggers. This covers old clients, Android, iOS, Thyme and Halo
when they commit through the same canonical tables. The signal stores no food text.
An asynchronous bounded worker rebuilds from current canonical rows, coalescing
multiple writes. Reads also repair dirty state, so correctness does not depend on
the schedule. Updates, deletes, corrections and Undo remove obsolete contributions.

    canonical transaction -> household input revision / dirty flag
                                     |
    operator correction ------------+--> bounded projection -> consumer context
    verified service events --------+          ^                   |
                                          fresh membership,        |
                                          restrictions, state <----+

Fresh inventory, restrictions, list and calendar remain operational data, not
inferred facts. Learned soft patterns always retain evidence and uncertainty.
Private saved recipes, dishes and consumption remain actor-scoped. Shared kitchen
and list observations can inform the household. A new membership cannot migrate
someone else's private records into a household context.

## Implementation tasks

1. Shared schema, deterministic projection, current-source reader and durable dirty
   tracking; provenance, bounded history, Undo and current-row reconciliation.
2. Internal operator controls with audit, optimistic revisions and idempotency;
   suppression, pause/resume, reset, lifecycle and explicit preferences.
3. Shared service/consumer adapters for legacy Python/Thyme and new Thyme;
   default-off rollout flags and no response-contract changes to installed apps.
   Shop's shared reader adapter/event contract is delivered; adoption by its
   separate authenticated relay remains a distinct integration before unification.
4. Real isolated MySQL tests, adversarial contract tests and consumer regression
   tests. Build deterministic deployment overlays against fresh serving packages.
5. Operator runbook and exact action/update matrix. Report implemented and activated
   coverage separately. Customer UI is excluded by Matt; no memory-dashboard work.

Sequential implementation: core contracts precede consumer adapters. Other active
agents' checkouts are preserved. Source is isolated at this worktree. Production
enrollment, database triggers and serving changes must use the reviewed rollout;
no broad deployment from a stale checkout.

## Engineering review

Gstack architecture/code/test/performance checklist applied to this plan. The
installed skill refers to an absent `sections/review-sections.md`; no complete
formal gstack plan-review run is claimed. Outside-provider review and telemetry
are disabled by the Trepo policy. No shared-company-brain write is authorized.

Risks addressed: committed action versus accepted job, duplicated household mirrors,
private/shared cache separation, stale membership, corrections versus inference,
Undo before retry, paused-period recovery, event retention, source read failure,
worker interruption and compare-and-swap publication. Capture triggers coalesce
work; reads are bounded and fail visibly on incomplete source coverage. Optional
personalization failure must never reverse or repeat a completed domain action.
