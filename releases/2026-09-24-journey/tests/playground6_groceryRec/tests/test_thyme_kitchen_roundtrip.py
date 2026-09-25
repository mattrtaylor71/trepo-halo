import json,os,subprocess,sys
from pathlib import Path
import pytest
from test_amount_operations import db,row,run,a


@pytest.mark.parametrize('mode',['rename','retry','stale','foreign','mismatch','duplicate','newest','unrelated-id','shortening','fresh-retry','fresh-later-edit','fresh-deleted','fresh-conflict'])
def test_thyme_to_canonical_kitchen_real_database(db,mode):
    assert os.environ.get('TREPO_AMOUNT_TEST_PORT') == '33317'
    # Exercise the actual mixed collations used by the protected edit sidecar.
    with db[0].cursor() as cur:
        cur.execute('ALTER TABLE shared_kitchen CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci')
        cur.execute('ALTER TABLE shared_kitchen ADD _createdDate DATETIME DEFAULT CURRENT_TIMESTAMP')
        if mode in ('duplicate','newest','unrelated-id','shortening'):
            cur.execute("UPDATE shared_kitchen SET product_name='Beef Broth', _createdDate='2026-08-01', _updatedDate='2026-09-24' WHERE _id='beef'")
            cur.execute("UPDATE shared_kitchen SET owner_id='owner', product_name=%s, _createdDate='2026-09-01', _updatedDate='2026-09-01' WHERE _id='eggs'", ('Beef Broth' if mode in ('duplicate','newest') else 'Garlic Oil',))
        if mode in ('stale','mismatch'):
            cur.execute("INSERT INTO kitchen_item_edits(item_id,overrides) VALUES('beef',JSON_OBJECT())")
        if mode=='mismatch':
            cur.execute("CREATE TRIGGER fixture_mismatch BEFORE UPDATE ON shared_kitchen FOR EACH ROW FOLLOWS trepo_preserve_item_edits SET NEW.product_name=OLD.product_name")
    schema=db[0].db.decode()
    root=Path(__file__).resolve().parents[2]
    script=Path(os.environ.get('TREPO_THYME_TEST_SOURCE',str(root/'playground12_voice_ack/trepo-quick-ack')))/'test/fixtures/kitchen-edit-roundtrip.mjs'
    worker=Path(__file__).with_name('kitchen_roundtrip_worker.py')
    result=subprocess.run(['/opt/homebrew/bin/node','--experimental-test-module-mocks',str(script),schema,mode,str(worker),sys.executable],
        capture_output=True,text=True,timeout=20,env={**os.environ,'WRITE_SHARED_ONLY':'true','READ_SHARED_KITCHEN':'true'})
    assert result.returncode==0,result.stderr
    result=json.loads(result.stdout.splitlines()[-1])
    if mode.startswith('fresh-'):
        if mode=='fresh-conflict':
            assert result['error']['status']==409 and result['requests']==1,result
            assert row(db)['product_name']=='Garlic Olive Oil'
        else:
            assert not result.get('error') and result['requests']==2,result
            assert result['item']['item_name']=='Garlic Olive Oil' and result['item']['amount_revision']==1
            if mode=='fresh-deleted': assert row(db) is None
            else:
                assert row(db)['product_name']==('Later human name' if mode=='fresh-later-edit' else 'Garlic Olive Oil')
                assert row(db)['quantity_value']==25
        return
    if mode in ('duplicate','newest','unrelated-id','shortening'):
        if mode=='duplicate':
            assert result['error']['status']==409 and result['requests']==0
            assert row(db)['product_name']==row(db,'eggs')['product_name']=='Beef Broth'
        else:
            assert not result.get('error'), result
            target='eggs' if mode=='newest' else 'beef'
            assert row(db,target)['product_name']==('Broth' if mode=='shortening' else 'Organic Beef Broth')
            assert row(db,'beef' if target=='eggs' else 'eggs')['product_name']==('Beef Broth' if mode=='newest' else 'Garlic Oil')
            assert result['requests']==1
        return
    if mode in ('stale','foreign','mismatch'):
        assert result['error']['status'] in (404,409),result
        assert row(db)['product_name']=='beef'
    else:
        assert not result.get('error'),result
        assert row(db)['product_name']==result['item']['item_name']=='Garlic Olive Oil'
        assert result['item']['amount_revision']==1
        assert row(db)['quantity_value']==25
    assert result['requests']==(0 if mode=='foreign' else 2 if mode=='retry' else 1)
