# obs — observability lambdas (standalone, not a SAM stack)

Each obs lambda is deployed on its own via raw CLI (kept independent of the big SAM
stacks so they stay surgical). Current fns: `trepo-alarm-triage`,
`trepo-voice-dualwrite-canary`.

## Deploy — the ONE blessed command (per lambda)

```bash
bash obs/<lambda>/deploy.sh        # e.g. bash obs/alarm-triage/deploy.sh
```

`deploy.sh` zips `index.mjs` + `package.json` + `node_modules` and
`update-function-code`s; on first run it also creates the IAM role / SNS / schedule
(the infra blocks are documented in each lambda's own README). No SAM, no `.sam-src`.
For a canary (`voice-dualwrite-canary`) with no deploy.sh yet, the blessed path is the
zip + `aws lambda update-function-code` shown in its README.
