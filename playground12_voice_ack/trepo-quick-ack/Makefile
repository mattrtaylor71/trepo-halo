# ── trepo-quick-ack deploy ────────────────────────────────────────────────────
# There is exactly ONE blessed deploy command. `sam deploy` on its own can ship a
# STALE .sam-src bundle (the 07-02 / 07-06 voice dual-write reverts) — DO NOT run it.
#
#   make deploy      ← regenerates .sam-src from lib/+shared, guards freshness, deploys
#   make guard       ← just check .sam-src is fresh (fails loudly if not)
#
.PHONY: deploy guard prepare
deploy:
	npm run sam:deploy
guard:
	npm run sam:guard
prepare:
	npm run sam:prepare
