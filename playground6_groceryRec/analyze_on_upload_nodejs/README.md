# Grocery Analyzer (Node.js)

This package contains the Node.js Lambda that processes grocery image uploads from S3, identifies the grocery item with OpenAI-backed logic, and writes the result into the household kitchen table named `<owner>_prod_kitchen`.

## What this backend does

- Listens for uploaded grocery images through the parent SAM stack
- Downloads the uploaded image from S3
- Identifies the grocery item, ingredients, nutrition summary, UPF flag, and alternatives
- Optionally extracts expiration date and store availability
- Writes one or more rows into `<owner>_prod_kitchen`
- Creates the kitchen table on demand if it does not already exist
- Updates the DynamoDB job record and publishes the result over IoT/MQTT

## Repo layout

- `app.js` - Lambda entrypoint used by SAM
- `src/openai/*` - grocery identification, expiration extraction, and store lookup logic
- `src/utils/mysqlWriter.ts` - MySQL table creation and writes to `<owner>_prod_kitchen`
- `src/utils/mysqlFeedWriter.ts` - feed/event persistence
- `src/utils/householdSync.ts` - household fan-out so multiple members can receive the same item
- `../template.yaml` - parent SAM template that deploys this function as `AnalyzeOnUpload`

## Build locally

```bash
npm install
npm run build
```

The compiled JavaScript goes into `dist/`. That folder is intentionally ignored in git, so engineers should build locally before deploying.

## Deploy with SAM

Run these commands from `playground6_groceryRec/`:

```bash
sam build
sam deploy
```

The SAM template points `AnalyzeOnUpload` at `analyze_on_upload_nodejs/` and sets the runtime to `nodejs20.x`.

## Required environment variables

- `BUCKET_NAME` - S3 bucket that stores uploaded grocery images
- `JOBS_TABLE` - DynamoDB table that tracks upload job state
- `KEY_PREFIX` - S3 key prefix to watch, usually `images/`
- `IOT_ENDPOINT` - AWS IoT Data endpoint used for result publishing
- `RESULT_TOPIC_TEMPLATE` - MQTT topic template for job results
- `DB_HOST` - MySQL host
- `DB_PORT` - MySQL port
- `DB_USER` - MySQL user
- `DB_PASS` - MySQL password
- `DB_NAME` - MySQL database name
- `OPENAI_API_KEY` - API key used by the grocery identification and enrichment calls

Optional environment variables are also used for model selection, image verification, stock image lookup, and downstream recipe or meal-plan triggers.

## Output table behavior

Rows are written to `<owner>_prod_kitchen`, where `owner` is sanitized to alphanumeric, `_`, and `-`.

The writer stores:

- core product fields such as `product_name`, `brand`, `variant`, `category`, and `confidence`
- product metadata such as `ingredients`, `nutrition_summary`, `upf`, and `harmful_ingredients`
- recommendation fields such as `similar_items`, `alternatives`, and `healthier_alternatives`
- image and job metadata such as `images`, `s3_key`, `job_id`, `user_id`, and `product_expiration`

If the owner belongs to a household, the backend can fan the same job out to each member's kitchen table.

## End-to-end flow

1. The app requests a presigned upload URL from the API layer.
2. The client uploads the grocery image to S3.
3. EventBridge invokes this Lambda after the object is created.
4. The Lambda analyzes the image and writes the grocery row to MySQL.
5. The Lambda marks the job as `DONE` or `FAILED` in DynamoDB.
6. The result is published to the configured IoT topic for the client to consume.
