# grocery-identifier (bulk grocery / receipt scan → kitchen)

SAM stack backing `grocery-identifier-dev-*` (identify-async, bulk-commit,
enrich-kitchen-item, health, …). TypeScript source in `src/` compiles to `dist/`;
a few top-level JS modules (`bulkKitchenWriter.js`, `bulkEnrichmentEngine.js`,
`bulkKitchenSimilarity.js`, `quickIdentify/`) are authored directly in JS.

## Deploy — the ONE blessed command

```bash
npm run build && bash deploy-sam.sh
```

`npm run build` runs `tsc` (src → dist); `deploy-sam.sh` loads `.env` and
`sam build && sam deploy`s the stack. Always build before deploying so `dist/`
matches `src/`.

### ⚠️ Live-vs-repo reconcile in progress

The three lambdas currently run **independently-drifted builds**, and at least one
carries a **prod-only** patch (identify-async `bulkKitchenWriter.js` create-retry +
20s timeout) that is not yet in git. Until that reconcile lands (repo made canonical,
preserving the prod patch), a blanket `deploy-sam.sh` could clobber live behavior —
**verify deployed-vs-repo (hash the deployed zips) before deploying**, and prefer
surgical per-fn updates for hotfixes. See the reconcile task before treating the repo
as the source of truth for these three functions.

## Test

```bash
npm run test:characterization    # node --experimental-test-module-mocks --test test/*.test.mjs
```
