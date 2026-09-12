# Ask Thyme recipe formatting repair — September 12, 2026

A live iOS review-account request returned structured recipe text in `final_text` but collapsed headings and numbered instructions in `app_output.message.text`. Shipped clients also selected a description immediately above Ingredients as the recipe name and rejected `**Ingredients:**` headings.

`formatRecipeResponse` converts complete, recognized recipe blocks to the existing shipped text convention: dish name immediately followed by Ingredients, ingredient bullets, then numbered Steps. Descriptions are retained before the recipe blocks; notes and follow-up text remain in Notes. Ambiguous or incomplete text returns null and retains the existing fallback. This does not generate ingredients or instructions.

Only app-surface responses use this canonical form. The streaming final event, synchronous response and app_output agree. Request/response field names, ownership/auth checks, tools, session persistence and firmware behavior are unchanged. No database migration or native update is required for the server repair.

Validation: 18 Node tests pass, including idempotence and preservation of allergy notes. Fourteen non-recipe/firmware response comparisons match the pre-deployment implementation. Twelve canonical formatting variants pass the original shipped Swift parser. Live initial/followup streaming and synchronous responses passed with matching final/app_output text and conversation memory. Native simulator tests additionally cover repeated card taps, EOF, saving and cold relaunch.

Production functions: `trepo-quick-ack-stream-dev` and `trepo-quick-ack-sam-dev`. Each package was built from its exact live baseline ZIP, replacing only index.js, index-stream.js and lib/app-response.mjs, and adding lib/recipe-response.mjs. Unrelated dirty local backend files were excluded. No configuration changes were made. Live code hashes and successful update status were verified. Private baseline ZIPs, revision records, package hashes and bounded review-account responses are retained in the operator's release cache for rollback.

The initial refinement attempt correctly stopped because Lambda had advanced RevisionId when the preceding asynchronous deployment completed. The code hash and LastModified still matched the recorded deployment. The retry used the freshly read revision with the expected code hash, preserving concurrency protection.
