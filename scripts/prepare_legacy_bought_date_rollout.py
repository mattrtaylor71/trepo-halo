"""Prepare minimal production patches against fresh captured serving packages.

No deployment or config changes. Preserve all other live bytes and verify the
actual candidate handler imports. Captures/credentials remain outside Git.
"""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
from verify_node_lambda_package import verify_package

root = Path(__file__).resolve().parents[1]
capture = Path(sys.argv[1])
sha = lambda blob: base64.b64encode(hashlib.sha256(blob).digest()).decode()
for name in ['trepo-grocery-backend-dev-KitchenApiFunction-lC0BHStmZf1k', 'trepo-quick-ack-stream-dev']:
    folder = capture / name
    config = json.loads((folder / 'config.private.json').read_text())
    original = (folder / 'before.zip').read_bytes()
    assert sha(original) == config['CodeSha256']
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        if 'KitchenApi' in name:
            source = archive.read('app.py').decode()
            anchor = '            return _get_kitchen_items(owner, event)'
            assert source.count(anchor) == 1
            source = source.replace(anchor, '            from legacy_bought_date import project_saved_corrections\n'
                                    '            return project_saved_corrections(_get_kitchen_items(owner, event), event)')
            # Verify the source contract used to select only explicit corrections.
            annotation = archive.read('amount_operations.py').decode()
            assert "if '__acquired_on' in saved:row['acquired_on']=saved['__acquired_on']" in annotation
            changes = {'app.py': source.encode(), 'legacy_bought_date.py':
                       (root / 'playground6_groceryRec/kitchen_api/legacy_bought_date.py').read_bytes()}
            for key, value in changes.items():
                compile(value, key, 'exec')
        else:
            entry = 'playground12_voice_ack/trepo-quick-ack/index-stream.js'
            source = archive.read(entry).decode()
            anchor = '  const toolEvents = result?.toolEvents || [];'
            assert source.count(anchor) == 1
            addition = '''
  // Restore legacy app refresh only after a verified successful date edit.
  if (requestInput?.responseSurface === "app") {
    for (const event of legacyDateRefreshEvents(result)) writeLine(responseStream, event);
  }
'''
            source = "import {legacyDateRefreshEvents} from '../../shared/voice-assistant/legacy-date-refresh.mjs';\n" + source.replace(anchor, anchor + addition)
            changes = {entry: source.encode(), 'shared/voice-assistant/legacy-date-refresh.mjs':
                       (root / 'shared/voice-assistant/legacy-date-refresh.mjs').read_bytes()}
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as dest:
            for info in archive.infolist():
                dest.writestr(copy.copy(info), changes.get(info.filename, archive.read(info.filename)))
            for path in set(changes) - set(archive.namelist()):
                dest.writestr(path, changes[path])
        blob = output.getvalue()
        with zipfile.ZipFile(io.BytesIO(blob)) as candidate:
            assert {p for p in archive.namelist() if archive.read(p) != candidate.read(p)} == set(changes) & set(archive.namelist())
            assert set(candidate.namelist()) - set(archive.namelist()) == set(changes) - set(archive.namelist())
        verification = verify_package(blob, config['Handler']) if 'nodejs' in config['Runtime'] else {'compile': 'passed'}
        (folder / 'candidate.zip').write_bytes(blob)
        manifest = {'before_sha': sha(original), 'after_sha': sha(blob), 'changed': sorted(changes),
                    'verification': verification, 'legacy_max_build': 205, 'scope': 'authorized explicit bought-date corrections'}
        (folder / 'candidate-manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps({'function': name, **manifest}))
