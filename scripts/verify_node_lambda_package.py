"""Import the configured handler from the ZIP that will actually be deployed.

No handler invocation or production credentials. Map Lambda's /var/task prefix
to the extracted package; load the real consent guard rather than stubbing it.
This checks startup imports, not dynamic branches or business behavior.
"""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def verify_package(package, handler, node=None):
    module, export = handler.rsplit('.', 1)
    with tempfile.TemporaryDirectory(prefix='trepo-lambda-startup-') as directory:
        root = Path(directory).resolve()
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            for name in archive.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise ValueError('Unsafe ZIP path')
            archive.extractall(root)
        entry = next((root / (module + ext) for ext in ('.js', '.mjs', '.cjs')
                      if (root / (module + ext)).is_file()), None)
        if entry is None:
            raise ValueError('Configured Lambda handler is absent')
        loader = root / '_package-startup-loader.mjs'
        loader.write_text('''import {pathToFileURL} from 'node:url';
export async function resolve(specifier, context, nextResolve) {
  if (specifier.startsWith('/var/task/'))
    specifier = pathToFileURL(process.env.PACKAGE_ROOT + specifier.slice(9)).href;
  return nextResolve(specifier, context);
}
''')
        source = '''
globalThis.awslambda = {streamifyResponse: fn => fn, HttpResponseStream: {from: value => value}};
const loaded = await import(ENTRY);
if (typeof loaded[EXPORT] !== 'function') throw new Error('Handler export missing');
console.log('PACKAGE_HANDLER_IMPORT_OK');
process.exit(0);
'''.replace('ENTRY', json.dumps(entry.as_uri())).replace('EXPORT', json.dumps(export))
        env = {'PATH': os.environ.get('PATH', ''), 'PACKAGE_ROOT': str(root),
               'AWS_EC2_METADATA_DISABLED': 'true', 'AWS_REGION': 'us-east-1',
               'NODE_PATH': os.pathsep.join(str(p) for p in root.rglob('node_modules'))}
        result = subprocess.run([node or shutil.which('node') or 'node', '--no-warnings',
                                 '--experimental-loader', str(loader), '--input-type=module',
                                 '-e', source], env=env, capture_output=True, text=True, timeout=30)
        if result.returncode or 'PACKAGE_HANDLER_IMPORT_OK' not in result.stdout:
            raise RuntimeError((result.stderr or result.stdout)[-6000:])
        return {'handler': handler, 'startup_import': 'passed', 'handler_invoked': False}
