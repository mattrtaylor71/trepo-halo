# Gstack repair review — September 24, 2026

Status: DONE_WITH_CONCERNS. Implementation and review are complete; release qualification is bounded below. No incident deployment, commit or TestFlight upload is implied.

The gstack investigation traced each reported failure to its writer, state contract or UI path before changing it. Scoped serving-derived patches were reviewed against their own baselines. The broad dirty repository diff is not the release unit. Native changes are separately preserved against their pre-integration bytes in a ten-file manifest.

An in-host adversarial reviewer examined transaction boundaries, source replacement, signed identity, replay/concurrency, temporary test routing and the final recovery callback. Its last pass found no new P1/P2 issue. That pass requested a direct edited-draft assertion; the final Release test edits quantity to 2, opens recovery, closes it and asserts 2 remains. No outside-provider review was run; that is disabled by company policy.

## Findings addressed

- Readback can disagree with the requested Kitchen edit. Verify the authoritative fields before committing the transaction and idempotency receipt.
- Slow recipe repair can race a human edit, source change or deletion. Revalidate under lock and make required shared/legacy writes atomic.
- A terminal import job can still contain no usable recipe. Preserve legacy error channels and retain the bookmark identity, with explicit same-ID recovery in the new app.
- Follow-up guards can intercept another feature's confirmation. Preserve cancellation/question rejection while respecting task scope.
- A retained receipt source list was excluded from worker redrive. Real DynamoDB retry tests reproduce and cover both single- and multi-image lists.
- A recovery action could appear while disabled or missing retained media. Require the server's per-image capability; missing/false offers retake.
- An outer recovery banner covered the review header. Native Release reproduced the unreachable target; the action now lives within the real review layout.
- A stored legacy machine note contradicted the terminal recipe state. Hide only that exact obsolete progress note in the retained-link presentation; human notes remain stored and editable.

## Qualification boundaries

The test report distinguishes source/database/model tests, exact-runtime boot probes, iOS native journeys and Android JVM contracts. Booting a cloud ZIP is not a cloud customer journey. Capture's native empty/mixed/expired-state tests do not establish successful retained-photo child-job commit. That new capability remains disabled until its signed configuration and controlled end-to-end activation check pass. Production rollout, physical speech/camera/Kitchen Assistant checks and new alert thresholds remain separate gates.

## Local operational learnings

1. Frozen test overlays must modify the actual nested app path, and evidence must compare every app input back to the current checkout. A pass from a similarly named cache directory is insufficient.
2. Native accessibility identifiers can propagate from a parent card to its buttons. Inspect the actual hierarchy and target the visible control; never remove the underlying behavior assertion to get a pass.
3. A top safe-area inset around an already structured review screen can cover its header. Verify both the screenshot and the hit frame, then keep actions inside the owning layout.
4. A remote command's reported exit code can be unreliable in this SSH environment. Independent xcresult, coverage, explicit exit receipts and source hashes are the acceptance authority.
5. The seven additive backend recipe envelopes were accepted by the actual Android parser after installing the required SDK under existing licenses. An initial missing-cache/SDK failure was an environment gap, not proof of client incompatibility.

These learnings are a local artifact only. Nothing was automatically saved to the shared company brain.
