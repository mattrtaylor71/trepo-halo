"""Add the omitted formatter to an exact captured live worker ZIP only."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile
import boto3
from verify_node_lambda_package import verify_package


def sha(value):
    return base64.b64encode(hashlib.sha256(value).digest()).decode()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    name = 'trepo-quick-ack-async-worker-dev'
    member = 'playground12_voice_ack/trepo-quick-ack/lib/recipe-response.mjs'
    expected = json.loads((args.evidence / (name + '.config.private.json')).read_text())
    before = (args.evidence / (name + '.zip')).read_bytes()
    assert sha(before) == expected['CodeSha256']
    addition = (Path(__file__).resolve().parents[1] / member).read_bytes()
    stream = zipfile.ZipFile(args.evidence / 'trepo-quick-ack-stream-dev.zip')
    assert stream.read(member) == addition, 'Use the formatter already serving app requests'
    output = io.BytesIO(before)
    with zipfile.ZipFile(output, 'a', zipfile.ZIP_DEFLATED) as archive:
        assert member not in archive.namelist()
        archive.writestr(member, addition)
    package = output.getvalue()
    with zipfile.ZipFile(io.BytesIO(before)) as old, zipfile.ZipFile(io.BytesIO(package)) as new:
        assert set(new.namelist()) - set(old.namelist()) == {member}
        assert all(new.read(path) == old.read(path) for path in old.namelist())
    startup = verify_package(package, expected['Handler'])
    receipt = {'function': name, 'before_sha': sha(before), 'after_sha': sha(package),
               'added': [member], 'existing_files_changed': [], 'startup': startup, 'applied': False}
    (args.evidence / 'worker-repaired.zip').write_bytes(package)
    client = boto3.Session(profile_name='trepo', region_name='us-east-1').client('lambda')
    current = client.get_function_configuration(FunctionName=name)
    assert current['RevisionId'] == expected['RevisionId'] and current['CodeSha256'] == expected['CodeSha256'], 'Live worker changed; recapture and review'
    if args.apply:
        client.update_function_code(FunctionName=name, ZipFile=package, RevisionId=current['RevisionId'])
        deadline = time.monotonic() + 110
        while time.monotonic() < deadline:
            latest = client.get_function_configuration(FunctionName=name)
            if latest['LastUpdateStatus'] != 'InProgress':
                break
            time.sleep(2)
        assert latest['LastUpdateStatus'] == 'Successful' and latest['CodeSha256'] == sha(package)
        for key in ['Environment', 'Handler', 'Runtime', 'Role', 'Layers', 'VpcConfig', 'Timeout', 'MemorySize']:
            assert latest.get(key) == current.get(key), key + ' changed'
        live = client.get_function(FunctionName=name)
        readback = urllib.request.urlopen(live['Code']['Location']).read()
        assert readback == package
        # Empty payload exits before looking up or claiming a job. This exercises
        # a real Lambda cold start without AI requests or customer writes.
        smoke = client.invoke(FunctionName=name, InvocationType='RequestResponse',
                              Payload=b'{}', LogType='Tail')
        payload = smoke['Payload'].read().decode()
        logs = base64.b64decode(smoke.get('LogResult', '')).decode()
        (args.evidence / 'live-cold-start.private.json').write_text(json.dumps({
            'function_error': smoke.get('FunctionError'), 'payload': payload, 'logs': logs}, indent=2))
        assert not smoke.get('FunctionError'), payload
        assert 'skipped invoke without jobId' in logs
        receipt.update(applied=True, revision=latest['RevisionId'], modified=latest['LastModified'],
                       live_cold_start='passed; no job supplied')
    (args.evidence / 'repair-receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
