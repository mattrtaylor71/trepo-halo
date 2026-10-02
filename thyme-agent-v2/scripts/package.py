"""Build from the immutable serving tool package plus this isolated pilot source."""
import argparse,base64,hashlib,json,pathlib,zipfile
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);args=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];base=pathlib.Path(args.base);data=base.read_bytes()
expected='DisVUkfocqTfH/57DT0pOzgZjfe0URtcRuvZc9J4dkU='
if base64.b64encode(hashlib.sha256(data).digest()).decode()!=expected:raise SystemExit('Serving tool package hash changed; inspect and repin explicitly.')
out=pathlib.Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
patched='playground12_voice_ack/trepo-quick-ack/lib/data-access.mjs'
with zipfile.ZipFile(base) as src,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as dst:
 for name in src.namelist():
  content=src.read(name)
  if name==patched:
   old=b'const limitedItems = items.slice(0, clampLimit(options.limit, 25, 100));'
   if content.count(old)!=1:raise SystemExit('Full shopping read patch no longer matches exactly.')
   content=content.replace(old,b'const limitedItems = options.all === true ? items : items.slice(0, clampLimit(options.limit, 25, 100));')
   fields=b'product_name, product_brand, store, action, _createdDate, created_at, updated_at'
   # Only getShoppingRows has these exact two SELECT clauses. Include the persisted amount.
   if content.count(fields)!=2:raise SystemExit('Shopping quantity SELECT patch no longer matches exactly.')
   content=content.replace(fields,b'product_name, product_brand, quantity, store, action, _createdDate, created_at, updated_at')
   unit=b'function normalizeUnit(value) {'
   if content.count(unit)!=1:raise SystemExit('Unit normalizer export patch changed.')
   content=content.replace(unit,b'export function normalizeUnit(value) {')
   store=b'async function resolveShoppingStore(connection, context, requestedStore) {'
   if content.count(store)!=1:raise SystemExit('Store normalizer export patch changed.')
   content=content.replace(store,b'export async function resolveShoppingStore(connection, context, requestedStore) {')
   for mapper in [b'mapKitchenRow',b'mapShoppingRow',b'mapDishRow']:
    old=b'function '+mapper+b'(row) {'
    if content.count(old)!=1:raise SystemExit('Row mapper export changed.')
    content=content.replace(old,b'export '+old)
   old=b'''    if (rowById?.action === "IN") {
      return [rowById];
    }
  }

  if (!resolved.item_name)'''
   if content.count(old)!=1:raise SystemExit('Exact kitchen ID resolver patch changed.')
   content=content.replace(old,old.replace(b'    }\n  }',b'    }\n    if (context?.requireExactKitchenId) return [];\n  }'))
  dst.writestr(name,content)
 for directory in ['src','node_modules']:
  for f in sorted((root/directory).rglob('*')):
   if f.is_file():dst.write(f,'thyme-agent-v2/'+str(f.relative_to(root)))
 dst.write(root/'package.json','thyme-agent-v2/package.json')
manifest={'base_serving_sha256':expected,'pilot_archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'base_patch':'Explicit all=true returns the complete shopping collection; default behavior unchanged.'}
out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
