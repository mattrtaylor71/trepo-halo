"""python -m household_brain.cli <command>. Internal operators only."""
import argparse
import json
import os
import sys
import uuid
from . import migrate,operator,runtime,store
from .core import encode,digest


def operator_identity():
    import boto3
    arn=boto3.client('sts').get_caller_identity()['Arn']
    if arn not in json.loads(os.getenv('HOUSEHOLD_BRAIN_OPERATORS','[]')):
        raise PermissionError('operator_arn_not_allowlisted')
    return arn


def main():
    parser=argparse.ArgumentParser(description='Internal household brain management; no customer endpoint.')
    parser.add_argument('command',choices=['install','capture-on','capture-off','default-serving-on','default-serving-off','provision','inspect','change','refresh','status'])
    parser.add_argument('--reason',help='Required for a global capture/default-serving change.')
    parser.add_argument('--household')
    parser.add_argument('--request-file',help='Exact JSON change with revision, operation ID and reason. Never credentials.')
    args=parser.parse_args()
    identity=operator_identity()
    conn=runtime.connect()
    try:
        if args.command=='install':
            migrate.install(conn);result={'installed':True,'capture':'unchanged','serving':'unchanged'}
        elif args.command in {'capture-on','capture-off','default-serving-on','default-serving-off'}:
            if not args.reason or not 5<=len(args.reason)<=500:parser.error('--reason (5–500 characters) is required')
            field='capture_enabled' if args.command.startswith('capture-') else 'default_serving'
            enabled=args.command.endswith('-on')
            operation_id=uuid.uuid4().hex
            request={'action':args.command,'reason':args.reason}
            result={field:enabled,'operation_id':operation_id}
            conn.begin()
            try:
                with conn.cursor() as cur:
                    cur.execute('UPDATE household_brain_config SET '+field+'=%s WHERE singleton=1',[enabled])
                    cur.execute('UPDATE household_brain_state SET input_revision=input_revision+1,dirty=1,updated_at=UTC_TIMESTAMP(6)')
                    cur.execute('DELETE FROM household_brain_contexts')
                    cur.execute('INSERT INTO household_brain_audit(operation_id,operator_id,household_id,action,request_hash,request,result) VALUES(%s,%s,%s,%s,%s,%s,%s)',
                        [operation_id,identity,'*',args.command,digest(request),encode(request),encode(result)])
                conn.commit()
            except BaseException:
                conn.rollback();raise
        elif args.command=='provision':
            migrate.provision(conn);result={'provisioned':True}
        elif args.command=='change':
            if not args.request_file:parser.error('--request-file is required')
            with open(args.request_file) as f:request=json.load(f)
            result=operator.change(conn,identity,request)
        elif args.command=='inspect':
            if not args.household:parser.error('--household is required')
            result=operator.inspect(conn,args.household)
        elif args.command=='refresh':
            result=store.rebuild(conn,args.household) if args.household else store.run_worker(conn)
        else:
            with conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) households,SUM(enabled) enabled,SUM(paused) paused,SUM(dirty AND enabled) pending FROM household_brain_state')
                result=cur.fetchone()
                # SUM() is Decimal in MySQL; avoid accidental content or secret logging.
                result={k:int(v or 0) for k,v in result.items()}
                cur.execute('SELECT capture_enabled,default_serving FROM household_brain_config WHERE singleton=1')
                result['configuration']=cur.fetchone()
        print(encode(result))
    finally:
        conn.close()


if __name__=='__main__':
    main()
