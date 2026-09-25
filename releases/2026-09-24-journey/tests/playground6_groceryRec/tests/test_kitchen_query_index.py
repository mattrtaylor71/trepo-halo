"""Real MySQL regression for the observed Explore cross-owner range scan."""
import os
from pathlib import Path
import uuid

import pytest
import pymysql


def test_owner_index_preserves_results_and_bounds_examined_rows():
    port = os.getenv('TREPO_TEXT_JOB_TEST_PORT')
    if not port:
        pytest.skip('Requires isolated local MySQL fixture container')
    conn = pymysql.connect(host='127.0.0.1', port=int(port), user='root',
        database='trepo_text_jobs_test', autocommit=True,
        cursorclass=pymysql.cursors.DictCursor)
    root = Path(__file__).resolve().parents[1]
    table = 'fixture_owner_index_' + uuid.uuid4().hex
    owner = str(uuid.uuid4())
    query = f'''SELECT product_name, product_description FROM `{table}`
        WHERE _owner=%s AND action='IN'
        AND (analysis_stage='final' OR analysis_stage IS NULL)
        AND (analysis_status='ready' OR analysis_status IS NULL)
        ORDER BY COALESCE(_updatedDate,_createdDate) DESC,_createdDate DESC'''
    try:
        with conn.cursor() as cur:
            cur.execute((root/'tests/fixtures/shared_kitchen_20260909.sql').read_text()
                        .replace('`shared_kitchen`', '`'+table+'`'))
            rows = [(str(uuid.uuid4()), str(uuid.uuid4()), owner if i < 5 else str(uuid.uuid4()),
                     'Fixture '+str(i), 'IN', 'final', 'ready') for i in range(5000)]
            # owner_id is intentionally different: changing the filter would break identity.
            rows += [(str(uuid.uuid4()), owner, owner, 'Excluded', action, stage, status)
                     for action,stage,status in [('OUT','final','ready'),('IN','preliminary','ready'),('IN','final','failed')]]
            cur.executemany(f'INSERT INTO `{table}` (_id,owner_id,_owner,product_name,action,analysis_stage,analysis_status) VALUES (%s,%s,%s,%s,%s,%s,%s)', rows)
            cur.execute(query, [owner]); before = cur.fetchall()
            assert len(before) == 5
            cur.execute((root/'migrations/shared_kitchen_query_owner_index.sql').read_text()
                        .replace('ALTER TABLE shared_kitchen', 'ALTER TABLE `'+table+'`'))
            cur.execute('ANALYZE TABLE `'+table+'`')
            cur.execute(query, [owner]); after = cur.fetchall()
            assert sorted(before,key=lambda r:r['product_name']) == sorted(after,key=lambda r:r['product_name'])
            cur.execute('EXPLAIN '+query,[owner]); plan=cur.fetchone()
            assert plan['key'] == 'idx_query_owner_action'
            assert plan['rows'] <= 10
    finally:
        with conn.cursor() as cur:
            cur.execute('DROP TABLE IF EXISTS `'+table+'`')
        conn.close()
