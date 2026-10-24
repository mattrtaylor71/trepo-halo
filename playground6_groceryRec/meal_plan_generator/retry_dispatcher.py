"""Scheduled retry delivery only; never runs a model or modifies customer plans."""
import json
import os
import boto3
from app import _mysql_conn
import generation_jobs

def handler(event, context):
    conn = _mysql_conn()
    jobs = generation_jobs.due(conn)
    sent, errors = 0, 0
    client = boto3.client('lambda')
    for job in jobs:
        try:
            payload = job['payload']
            if isinstance(payload,str):
                payload = json.loads(payload)
            response = client.invoke(FunctionName=os.environ['MEAL_PLAN_GENERATOR_ARN'],InvocationType='Event',
                Payload=json.dumps({**payload,'generation_retry_token':job['run_token'],
                    'generation_retry_scope':job['scope_id']}).encode())
            if response.get('StatusCode') != 202:
                raise RuntimeError('dispatch_not_acknowledged')
            sent += 1
        except Exception as exc:
            errors += 1
            print(json.dumps({'evt':'meal_plan_retry_dispatch_failed','type':type(exc).__name__}))
    # An accepted delivery is not a completed plan. The queue row stays due until
    # the worker claims it, so a lost acknowledgement safely causes redelivery.
    return {'due':len(jobs),'dispatched':sent,'errors':errors}
