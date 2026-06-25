import pymysql, uuid as uuidlib, sys, re
EXECUTE = '--execute' in sys.argv
conn = pymysql.connect(host="database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com",
    user="admin", password="Nbmqyq17!", database="mysqlTutorial", connect_timeout=15, autocommit=False)
cur = conn.cursor(pymysql.cursors.DictCursor)

def norm(s): return re.sub(r'\s+',' ',(s or '').strip().lower())

cur.execute("""SELECT owner_id FROM new_users WHERE owner_id IS NOT NULL AND owner_id<>''
  GROUP BY owner_id HAVING COUNT(*)>1""")
households=[r['owner_id'] for r in cur.fetchall()]

def members(hid):
    cur.execute("SELECT user_id, first_name FROM new_users WHERE owner_id=%s ORDER BY created_at ASC",(hid,))
    return cur.fetchall()
def listrows(uid):
    try:
        cur.execute(f"SELECT * FROM `{uid}_new_list`"); return cur.fetchall()
    except Exception: return None

mode = "EXECUTE" if EXECUTE else "DRY RUN"
print(f"=== RECONCILE ({mode}) ===\n")
total_ins=0; total_uuid_backfill=0

for hid in households:
    mem=members(hid)
    memlists={m['user_id']:(m['first_name'],listrows(m['user_id'])) for m in mem}
    memlists={k:v for k,v in memlists.items() if v[1] is not None}
    if len(memlists)<2: continue

    # Build canonical groups. Key = existing uuid, else normalized-name.
    # group -> {uuid, name(canonical/newest), action, store, qty, brand, present:set(uid), rows:[(uid,row)]}
    groups={}
    def gkey(r):
        return r['household_item_uuid'] if r['household_item_uuid'] else 'NAME:'+norm(r['product_name'])
    for uid,(nm,rows) in memlists.items():
        for r in rows:
            k=gkey(r)
            g=groups.setdefault(k,{'uuid':None,'row':None,'present':set(),'rows':[]})
            if r['household_item_uuid']: g['uuid']=r['household_item_uuid']
            # pick canonical row = newest updated_at (fallback createdDate)
            if g['row'] is None or ((r['updated_at'] or r['_createdDate']) and (g['row']['updated_at'] or g['row']['_createdDate']) and (r['updated_at'] or r['_createdDate'])>(g['row']['updated_at'] or g['row']['_createdDate'])):
                g['row']=r
            g['present'].add(uid)
            g['rows'].append((uid,r))

    hh_actions=[]
    for k,g in groups.items():
        missing=[uid for uid in memlists if uid not in g['present']]
        needs_uuid = g['uuid'] is None
        if not missing and not needs_uuid: continue
        # assign uuid if group has none
        assigned = g['uuid'] or str(uuidlib.uuid4())
        r=g['row']
        hh_actions.append((k,g,missing,assigned,needs_uuid))

    if not hh_actions: continue
    names=",".join(str(m['first_name']) for m in mem)
    print(f"Household {hid} ({names}):")
    for k,g,missing,assigned,needs_uuid in hh_actions:
        r=g['row']; nm=r['product_name']
        if needs_uuid:
            # backfill uuid onto all existing rows in this group
            for uid,row in g['rows']:
                if row['household_item_uuid'] is None:
                    print(f"   ~ set uuid {assigned[:13]} on '{nm}' in {memlists[uid][0]}")
                    total_uuid_backfill+=1
                    if EXECUTE:
                        cur.execute(f"UPDATE `{uid}_new_list` SET household_item_uuid=%s WHERE _id=%s AND household_item_uuid IS NULL",(assigned,row['_id']))
        for uid in missing:
            who=memlists[uid][0]
            print(f"   + insert '{nm}' [{r['action']}] into {who} (uuid {assigned[:13]})")
            total_ins+=1
            if EXECUTE:
                cur.execute(f"""INSERT INTO `{uid}_new_list`
                  (_owner,_device,product_name,quantity,product_brand,images,product_barcode,store,action,household_item_uuid,sort_order)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                  (uid,'household-reconcile',r['product_name'],r['quantity'],r['product_brand'],
                   r['images'],r['product_barcode'],r['store'],r['action'],assigned,r['sort_order']))
    print()

if EXECUTE:
    conn.commit(); print(f"COMMITTED. inserts={total_ins} uuid_backfills={total_uuid_backfill}")
else:
    print(f"=== WOULD: inserts={total_ins} uuid_backfills={total_uuid_backfill} (run with --execute) ===")
cur.close(); conn.close()
