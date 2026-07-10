# playground6_groceryRec (main grocery backend)

SAM stack `trepo-grocery-backend-dev` — the kitchen / dishes / discards / recipes /
saved-recipes / home-suggestions / analyze-on-upload APIs (Python + a couple Node
functions). Each function is its own `CodeUri: <dir>/` (per-dir, not a generated
bundle).

## Deploy — the ONE blessed command

```bash
bash scripts/deploy_safe.sh      # run from the playground6_groceryRec/ dir
```

`deploy_safe.sh` builds safe artifacts and `sam deploy`s the whole stack;
CloudFormation updates only the functions whose code changed. Always
`python3 -m py_compile <file>.py` first.

### Surgical single-fn hotfix (allowed)

For an urgent one-function fix, a surgical zip-swap (download the deployed zip, swap
the changed file, `aws lambda update-function-code`) is fine — **drift-check first**
(`diff` the deployed file vs `git show HEAD:...`) and never clobber newer deployed
state. Follow up with `bash scripts/deploy_safe.sh` when convenient so the stack
reconverges. Note: several service `app.py` files here are historically **untracked**
in git — treat the deployed zip as authoritative and reconcile the repo to it.
