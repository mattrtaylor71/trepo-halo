"""Read immutable generic artwork only. Original user photos cannot be requested."""
import base64
import hashlib
import os
import re
from .catalog import STYLE, APP_STYLE

KEY = re.compile(r'/icons/(?:trepo-grocery-v1|trepo-bold-outline-pilot-v1|trepo-app-bold-outline-v1)/[0-9a-f]{64}/[0-9a-f]{64}\.png')
def handle(event, client, bucket):
    path = event.get('rawPath') or ''
    if not KEY.fullmatch(path):return {'statusCode':404,'body':''}
    try:
        try:
            obj=client.get_object(Bucket=bucket,Key=path[1:])
        except Exception as error:
            code = getattr(error, 'response', {}).get('Error', {}).get('Code')
            # Exact immutable bytes through the legacy namespace. Never replace
            # an existing object, and never remap a digest to different artwork.
            # S3 returns 403 for a missing key without ListBucket permission.
            if code not in {'NoSuchKey', '404', 'NotFound', 'AccessDenied', '403'} or not path.startswith('/icons/' + STYLE + '/'):
                raise
            target = path.replace('/icons/' + STYLE + '/', '/icons/' + APP_STYLE + '/', 1)
            obj=client.get_object(Bucket=bucket,Key=target[1:])
        if obj['ContentLength']>800_000:return {'statusCode':404,'body':''}
        data = obj['Body'].read(800_001)
        if len(data)>800_000 or hashlib.sha256(data).hexdigest()!=path.split('/')[-1][:-4]:
            return {'statusCode':404,'body':''}
        return {'statusCode':200,'headers':{'Content-Type':'image/png','Cache-Control':'public,max-age=31536000,immutable','ETag':'"'+path.split('/')[-1][:-4]+'"','X-Content-Type-Options':'nosniff'},'isBase64Encoded':True,'body':base64.b64encode(data).decode()}
    except Exception:return {'statusCode':404,'body':''}

def lambda_handler(event, context):
    import boto3
    return handle(event,boto3.client('s3'),os.environ['ASSET_BUCKET'])
