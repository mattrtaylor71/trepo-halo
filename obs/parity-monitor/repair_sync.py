#!/usr/bin/env python3
"""Repair resync: per-owner (source of truth) -> shared, ALL intersection columns
including _createdDate/_updatedDate. UPSERT per table. saved_recipes rows are
applied oldest-first so the latest hash-twin wins, and its ODKU also updates _id
so shared converges to the twin the per-owner table considers current."""
import boto3, pymysql, sys, time
SUFFIX, SHARED, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
s=boto3.Session(profile_name='trepo-dev',region_name='us-east-1')
E=s.client('lambda').get_function_configuration(FunctionName='trepo-grocery-backend-dev-RecipesApiFunction-zjwZ1RP9bUAQ')['Environment']['Variables']
def cn(): return pymysql.connect(host=E['DB_HOST'],user=E['DB_USER'],password=E['DB_PASS'],database=E['DB_NAME'],port=int(E['DB_PORT']),cursorclass=pymysql.cursors.DictCursor,connect_timeout=10,read_timeout=60,write_timeout=60,autocommit=True)
conn=cn()
def q(sql,params=None):
    global conn
    for a in range(3):
        try:
            with conn.cursor() as c:
                c.execute(sql,params or []); return c
        except (pymysql.err.OperationalError,pymysql.err.InterfaceError):
            try: conn.close()
            except: pass
            time.sleep(1); conn=cn()
    raise RuntimeError('fail 3x')
shd={r['cn'] for r in q("SELECT COLUMN_NAME cn FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=%s",[SHARED]).fetchall()}
tbls=[r['tn'] for r in q("SELECT table_name tn FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name LIKE %s AND CHAR_LENGTH(table_name)=%s AND table_name NOT LIKE 'shared\\_%%'",['%'+SUFFIX,36+len(SUFFIX)]).fetchall()]
print(f"[{SUFFIX}] repairing {len(tbls)} tables",flush=True)
touched=err=0; t0=time.time()
for i,t in enumerate(tbls):
    o=t[:-len(SUFFIX)]
    try:
        cols=[r['cn'] for r in q("SELECT COLUMN_NAME cn FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=%s",[t]).fetchall() if r['cn'] in shd and r['cn']!='owner_id']
        if '_id' not in cols: continue
        cs=', '.join('`'+x+'`' for x in cols); ss=', '.join('s.`'+x+'`' for x in cols)
        if MODE=='by_hash':
            upd=', '.join(f'`{x}`=VALUES(`{x}`)' for x in cols)  # incl _id: twin converge
            order=" ORDER BY COALESCE(s.`_updatedDate`, s.`_createdDate`) ASC" if '_updatedDate' in cols else " ORDER BY s.`_createdDate` ASC"
            where=""
        else:
            upd=', '.join(f'`{x}`=VALUES(`{x}`)' for x in cols if x!='_id')
            order=""
            where=" WHERE s.`_id`='current'" if MODE=='current' else ""
        c=q(f"INSERT INTO `{SHARED}` (owner_id, {cs}) SELECT %s, {ss} FROM `{t}` s{where}{order} ON DUPLICATE KEY UPDATE owner_id=VALUES(owner_id), {upd}",[o])
        touched+=c.rowcount
    except Exception as e:
        err+=1
        if err<=5: print(f"[{SUFFIX}] err {o[:8]}: {str(e)[:100]}",flush=True)
    if (i+1)%1000==0:
        try: conn.close()
        except: pass
        conn=cn()
        print(f"[{SUFFIX}] {i+1}/{len(tbls)} touched={touched} err={err} {time.time()-t0:.0f}s",flush=True)
print(f"[{SUFFIX}] REPAIR DONE touched={touched} err={err} {time.time()-t0:.0f}s",flush=True)
