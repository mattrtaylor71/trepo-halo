"""trepo-kitchen-handler — voice kitchen-bucket shim.

The HALO voice handler (trepo-voice-handler) routes bucket=kitchen commands to a
Lambda by this name; the original was decommissioned, which crashed every kitchen
voice command (ResourceNotFoundException). This shim restores the route by
translating the legacy voice payload into the modern KitchenApi text-add call
(POST /kitchen/{owner}/text-add: LLM parse + create + shelf-life enrichment) —
the same path the in-app "Add with text" uses.

Supported: operation=add. Anything else returns 501 with a log marker so we can
see real demand before building it.
"""
import json
import os

import boto3

lambda_client = boto3.client('lambda')
KITCHEN_API = os.environ.get(
    'KITCHEN_API_FUNCTION',
    'trepo-grocery-backend-dev-KitchenApiFunction-lC0BHStmZf1k',
)


def lambda_handler(event, context):
    event = event or {}
    operation = str(event.get('operation') or '').lower()
    owner = str(event.get('ownerId') or '').strip()
    product_name = str(event.get('product_name') or '').strip()

    if not owner:
        return {'statusCode': 400, 'body': json.dumps({'error': 'missing ownerId'})}

    if operation == 'add' and product_name:
        gw_event = {
            'requestContext': {'http': {'method': 'POST', 'path': f'/kitchen/{owner}/text-add'}},
            'rawPath': f'/kitchen/{owner}/text-add',
            'pathParameters': {'owner': owner},
            'headers': {},
            'body': json.dumps({
                'text': product_name,
                'device_id': str(event.get('device') or 'halo-voice'),
                'user_id': owner,
            }),
        }
        resp = lambda_client.invoke(
            FunctionName=KITCHEN_API,
            InvocationType='RequestResponse',
            Payload=json.dumps(gw_event).encode('utf-8'),
        )
        parsed = json.loads(resp['Payload'].read().decode('utf-8'))
        status = parsed.get('statusCode', 502)
        print(json.dumps({
            'evt': 'voice_kitchen_add', 'owner_id': owner,
            'product': product_name, 'status': status,
        }))
        return {'statusCode': status, 'body': parsed.get('body')}

    print(json.dumps({
        'evt': 'voice_kitchen_unsupported_op', 'operation': operation,
        'owner_id': owner, 'product': product_name,
    }))
    return {'statusCode': 501, 'body': json.dumps({
        'error': f'kitchen voice operation not supported yet: {operation or "(empty)"}',
    })}
