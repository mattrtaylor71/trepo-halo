# Grocery Catalog Strategy

The packaged-goods path now uses a lightweight local catalog layer before any optional external product index.

## Current strategy
- Default catalog entries live in `quickIdentify/catalog.js`.
- Matching uses:
  - item name tokens
  - brand tokens
  - OCR/visible text tokens
  - alias phrase matches
- Produce is routed separately through a small canonical produce taxonomy.

## Why this shape
- Packaged goods benefit from OCR plus alias matching.
- Loose produce does not map well to UPC/GTIN-style catalog matching.
- A local catalog gives deterministic corrections for recurring pantry items without depending on an external cloud index.

## Extension path
- Set `GROCERY_CATALOG_PATH` to a JSON file containing additional packaged-goods entries.
- Future external catalog sources can append:
  - UPC / GTIN
  - normalized title
  - brand
  - category
  - OCR aliases
  - reference images

## Recommended long-term index
- Packaged goods:
  - OCR aliases
  - UPC / GTIN where available
  - brand
  - canonical title
  - front-of-pack reference images
- Produce:
  - canonical produce taxonomy
  - visual synonyms only

This keeps the runtime system hybrid:
- catalog matching for packaged items
- taxonomy matching for produce
- final LLM reconciliation for whole-scene cleanup
