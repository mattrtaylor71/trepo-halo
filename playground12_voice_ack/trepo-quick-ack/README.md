# trepo-quick-ack (voice / Thyme capture-ack stack)

SAM stack of 5 Lambdas (stream, async-worker, sam, async-ingest, image-postprocess)
that share `lib/data-access.mjs` + `lib/device-assistant.mjs` and the
`shared/voice-assistant/*` files.

## Deploy — the ONE blessed command

```bash
make deploy
```

That's it. `make deploy` (== `npm run sam:deploy`) **regenerates `.sam-src` from
`lib/` + `shared/`**, runs the freshness guard, then `sam build && sam deploy`.

### ⛔ Never run `sam deploy` (or `sam build`) directly

The functions use `CodeUri: .sam-src` — a *generated* bundle. A bare `sam deploy`
ships whatever is in `.sam-src`, which can be **stale** relative to `lib/` — this is
exactly what caused the 07-02 / 07-06 voice dual-write reverts (fixes that were live
got clobbered by a deploy from an old bundle). `.sam-src` is git-ignored and never
committed.

Guardrail: `npm run sam:build` / `npm run sam:deploy` run `sam:guard`
(`scripts/assert-sam-src-fresh.mjs`), which **fails loudly** if `.sam-src` is missing
or its source hash doesn't match live `lib/`+`shared/`. Run just the check with
`make guard`.

For a one-off hotfix to a single deployed function, the surgical zip-swap pattern is
fine — but always follow up with `make deploy` so `.sam-src` and all 5 fns reconverge.

## Test

```bash
npm test    # node --experimental-test-module-mocks --test test/*.test.mjs
```
Characterization suite (mocked DB) pinning the dual-write scope invariants.
