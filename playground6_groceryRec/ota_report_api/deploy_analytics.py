#!/usr/bin/env python3
"""Deploy only this Lambda's two code modules; its live environment is never written.
Default is a read-only preview. --deploy requires the reviewed current code SHA.
"""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile
import boto3

FUNCTION = 'trepo-grocery-backend-dev-OtaReportApiFunction-cyUvBswGXwDb'
ARN = 'arn:aws:lambda:us-east-1:566667681926:function:' + FUNCTION
ROOT = Path(__file__).resolve().parent


def package():
    content = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(content, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('app.py', 'analytics.py'):
            data = (ROOT / name).read_bytes()
            ast.parse(data, filename=name)
            hashes[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 10, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return content.getvalue(), hashes


def run(args):
    code, hashes = package()
    candidate = base64.b64encode(hashlib.sha256(code).digest()).decode()
    if args.package_only:
        return {'status': 'LOCAL_PACKAGE_ONLY', 'candidate_code_sha256': candidate, 'files': hashes}
    client = boto3.Session(profile_name=args.profile, region_name='us-east-1').client('lambda')
    before = client.get_function_configuration(FunctionName=FUNCTION)
    if before.get('FunctionArn') != ARN or before.get('Handler') != 'app.handler' or before.get('Runtime') != 'python3.9':
        raise ValueError('Unexpected function identity/runtime; review before deploying')
    if before.get('State') != 'Active' or before.get('LastUpdateStatus') == 'InProgress':
        raise ValueError('Function is not ready for a bounded code update')
    environment = before.get('Environment', {}).get('Variables', {})
    if len(environment.get('ANALYTICS_READ_KEY', '')) < 32:
        raise ValueError('Live analytics secret is absent; this code-only tool will not provision or overwrite it')
    plan = {'status': 'READ_ONLY_PREVIEW', 'observed_epoch': time.time(), 'function_name': FUNCTION,
            'current_code_sha256': before['CodeSha256'], 'current_revision_id': before['RevisionId'],
            'candidate_code_sha256': candidate, 'files': hashes,
            'environment_action': 'preserve_without_update', 'routes_tables_iam_action': 'unchanged'}
    if not args.deploy:
        return plan
    if not args.expected_code_sha or args.expected_code_sha != before['CodeSha256']:
        raise ValueError('Current code SHA does not match the explicitly reviewed --expected-code-sha')
    if candidate == before['CodeSha256']:
        return dict(plan, status='ALREADY_CURRENT_NO_MUTATION')
    client.update_function_code(FunctionName=FUNCTION, ZipFile=code, Publish=False, RevisionId=before['RevisionId'])
    client.get_waiter('function_updated_v2').wait(FunctionName=FUNCTION, WaiterConfig={'Delay': 2, 'MaxAttempts': 30})
    after = client.get_function_configuration(FunctionName=FUNCTION)
    unchanged = ('FunctionArn', 'Runtime', 'Handler', 'Role', 'Timeout', 'MemorySize', 'Architectures',
                 'VpcConfig', 'Layers', 'TracingConfig', 'DeadLetterConfig', 'KMSKeyArn', 'EphemeralStorage')
    if after.get('CodeSha256') != candidate or after.get('Environment', {}).get('Variables', {}) != environment:
        raise ValueError('Post-update code/environment verification failed; inspect current state without automatic rollback')
    if any(before.get(key) != after.get(key) for key in unchanged):
        raise ValueError('Concurrent configuration change observed; inspect without automatic rollback')
    return dict(plan, status='CODE_UPDATED_ENVIRONMENT_PRESERVED', finished_epoch=time.time(),
                after_revision_id=after['RevisionId'], verified_environment_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='trepo')
    parser.add_argument('--package-only', action='store_true', help='Offline package hashes; no AWS calls')
    parser.add_argument('--deploy', action='store_true', help='Explicitly apply a reviewed code update')
    parser.add_argument('--expected-code-sha', help='Base64 CodeSha256 from the reviewed read-only preview')
    args = parser.parse_args()
    if args.deploy and args.package_only:
        parser.error('--package-only and --deploy are mutually exclusive')
    try:
        print(json.dumps(run(args), indent=2))
    except Exception as exc:
        # Do not serialize SDK errors or the live environment into console logs.
        message = str(exc) if type(exc) is ValueError else type(exc).__name__
        print(json.dumps({'status': 'REFUSED_OR_UPDATE_UNVERIFIED', 'error': message}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
