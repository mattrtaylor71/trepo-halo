"""Build review-only compatibility packages from exact captured serving ZIPs.

Preparation only: never deploys. Credentials/config captures stay outside Git.
The API wrapper targets one synthetic item in the documented review account.
"""
import copy
import hashlib
import base64
import io
import json
from pathlib import Path
import sys
import zipfile
from verify_node_lambda_package import verify_package

root = Path(__file__).resolve().parents[1]
capture = Path(sys.argv[1])
settings = json.loads((capture / 'settings.json').read_text())
actor = settings['review_actor']; item = settings['item_id']; build = settings['build']
assert actor == '8fcb37c1-0927-4c31-856a-5b761efc226c' and build == 150
sha = lambda blob: base64.b64encode(hashlib.sha256(blob).digest()).decode()
for name in ['trepo-grocery-backend-dev-KitchenApiFunction-lC0BHStmZf1k', 'trepo-quick-ack-stream-dev']:
    folder = capture / name
    config = json.loads((folder / 'config.private.json').read_text())
    original = (folder / 'before.zip').read_bytes()
    assert sha(original) == config['CodeSha256']
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        if 'KitchenApi' in name:
            wrapper = f'''

# Temporary review-only bought-age compatibility canary; exact synthetic item.
from legacy_bought_date import project_response as _legacy_bought_projection
_before_bought_canary_handler = handler
def handler(event, context):
    response = _before_bought_canary_handler(event, context)
    path = event.get('rawPath') or event.get('path') or ''
    if path.rstrip('/') != '/kitchen/{actor}':
        return response
    return _legacy_bought_projection(response, event,
        corrected_item_ids=('{item}',), max_build={build})
'''
            changes = {'app.py': archive.read('app.py') + wrapper.encode(),
                       'legacy_bought_date.py': (root / 'playground6_groceryRec/kitchen_api/legacy_bought_date.py').read_bytes()}
            for key, value in changes.items():
                compile(value, key, 'exec')
        else:
            entry = 'playground12_voice_ack/trepo-quick-ack/index-stream.js'
            source = archive.read(entry).decode()
            anchor = '  const toolEvents = result?.toolEvents || [];'
            assert source.count(anchor) == 1
            addition = f'''
  // Temporary canary: completed date edit of the one review-account fixture.
  if (ownerId === {json.dumps(actor)} && toolEvents.some(event =>
      event.result?.ok === true && event.result?.toolResult?.item?.id === {json.dumps(item)})) {{
    for (const compatibilityEvent of legacyDateRefreshEvents(result)) writeLine(responseStream, compatibilityEvent);
  }}
'''
            source = "import {legacyDateRefreshEvents} from '../../shared/voice-assistant/legacy-date-refresh.mjs';\n" + source.replace(anchor, anchor + addition)
            changes = {entry: source.encode(), 'shared/voice-assistant/legacy-date-refresh.mjs': (root / 'shared/voice-assistant/legacy-date-refresh.mjs').read_bytes()}
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as dest:
            for info in archive.infolist():
                dest.writestr(copy.copy(info), changes.get(info.filename, archive.read(info.filename)))
            for path in set(changes) - set(archive.namelist()):
                dest.writestr(path, changes[path])
        blob = output.getvalue()
        with zipfile.ZipFile(io.BytesIO(blob)) as candidate:
            assert {p for p in archive.namelist() if archive.read(p) != candidate.read(p)} == set(changes) & set(archive.namelist())
        verification = verify_package(blob, config['Handler']) if 'nodejs' in config['Runtime'] else {'compile': 'passed'}
        (folder / 'candidate.zip').write_bytes(blob)
        manifest = {'before_sha': sha(original), 'after_sha': sha(blob), 'changed': sorted(changes), 'verification': verification,
                    'review_actor': actor, 'item_id': item, 'build': build}
        (folder / 'candidate-manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps({'function': name, **manifest}))
