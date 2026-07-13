# trepo-errors-feed

A phone-friendly, token-gated live feed of real backend errors. Lambda + Function URL,
reads `TrepoAnalyticsEvents` (event_name=`backend_error`, newest-first via `EventNameIndex`),
filters known-benign/self-recovering noise (master_feed dup-PK, auth bad-phone, mealplan
count-retry), dedupes repeats into one card with a ×count.

- Function: `trepo-errors-feed` (nodejs20.x, role `trepo-errors-feed-role` → dynamodb:Query only)
- URL: Function URL, AuthType=NONE, gated by `?k=<FEED_TOKEN>` (env var, NOT in git)
- Deploy: `zip fn.zip index.js && aws lambda update-function-code --function-name trepo-errors-feed --zip-file fileb://fn.zip`
- Endpoints: `/?k=TOKEN` (HTML page, auto-refresh 20s) · `/?format=json&k=TOKEN` (data)
- To add/adjust benign filters: edit `isBenign()`.
