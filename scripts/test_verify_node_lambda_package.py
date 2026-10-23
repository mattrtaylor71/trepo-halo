import io
import unittest
import zipfile
from verify_node_lambda_package import verify_package


def package(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return buffer.getvalue()


class PackageStartupTests(unittest.TestCase):
    def test_rejects_missing_transitive_dependency(self):
        files = {'package.json': '{"type":"module"}',
                 'worker.js': "export {handler} from './lib/worker.mjs';",
                 'lib/worker.mjs': "import './recipe-response.mjs'; export const handler=()=>{};"}
        with self.assertRaisesRegex(RuntimeError, 'ERR_MODULE_NOT_FOUND'):
            verify_package(package(files), 'worker.handler')
        files['lib/recipe-response.mjs'] = 'export const formatter=()=>{};'
        self.assertEqual(verify_package(package(files), 'worker.handler')['startup_import'], 'passed')

    def test_loads_packaged_absolute_consent_import_without_invoking_handler(self):
        files = {'package.json': '{"type":"module"}',
                 'worker.js': "import guard from '/var/task/guard.cjs'; export const handler=()=>{throw Error('must not execute')}; if(!guard.real)throw Error('guard not loaded');",
                 'guard.cjs': 'module.exports={real:true};'}
        self.assertFalse(verify_package(package(files), 'worker.handler')['handler_invoked'])

    def test_rejects_missing_handler(self):
        with self.assertRaisesRegex(ValueError, 'handler is absent'):
            verify_package(package({}), 'worker.handler')


if __name__ == '__main__':
    unittest.main()
