from datetime import datetime, timedelta
import json
from unittest.mock import Mock, patch
import os
import uuid
import pytest
import pymysql

from test_saved_recipes_image_mirroring import load_app_module


def test_permanent_failure_survives_cold_start_until_source_is_replaced():
    app = load_app_module()
    url = 'https://fixture.invalid/unavailable.jpg'
    row = {'image_url': url, 'image_mirror_failed_at': datetime.now() - timedelta(days=20),
           'image_mirror_failure': json.dumps({'kind': 'source_unavailable', 'url_hash': app._sha256(url)})}
    assert app._row_mirror_failed_recently(row)
    app._image_mirror_failures.clear()
    assert app._row_mirror_failed_recently(row)
    row['image_url'] = 'https://fixture.invalid/replacement.jpg'
    assert not app._row_mirror_failed_recently(row)


def test_transient_failure_recovers_after_five_minutes_and_legacy_marker_is_preserved():
    app = load_app_module()
    url = 'https://fixture.invalid/image.jpg'
    row = {'image_url': url, 'image_mirror_failed_at': datetime.now() - timedelta(minutes=6),
           'image_mirror_failure': {'kind': 'transient', 'url_hash': app._sha256(url)}}
    assert not app._row_mirror_failed_recently(row)
    row['image_mirror_failed_at'] = datetime.now()
    assert app._row_mirror_failed_recently(row)
    row.pop('image_mirror_failure')
    assert app._row_mirror_failed_recently(row)


def test_403_is_recorded_permanently_without_reextracting_the_recipe_page():
    app = load_app_module()
    url = 'https://fixture.invalid/blocked.jpg'
    row = {'_id': 'fixture', 'image_url': url, 'source_url': 'https://fixture.invalid/recipe'}
    response = Mock(status_code=403)
    with patch.object(app, '_mirror_recipe_image', side_effect=app.requests.HTTPError('fixture', response=response)), \
         patch.object(app, '_mark_mirror_failed_in_db') as mark, \
         patch.object(app, '_extract_content') as extract, \
         patch.object(app, '_image_mirror_budget_left', return_value=10):
        assert app._ensure_owned_saved_recipe_image(Mock(), 'owner', row) is row
    assert mark.call_args.kwargs == {'failure_kind': 'source_unavailable', 'source_url': url}
    extract.assert_not_called()


def test_unchecked_transient_candidate_cannot_make_whole_recipe_permanently_failed():
    app = load_app_module()
    temporary = 'https://fixture.invalid/temporary.jpg'
    blocked = 'https://fixture.invalid/blocked.jpg'
    app._note_mirror_failure(temporary)
    row = {'_id': 'fixture', 'image_url': blocked, 'source_image_url': temporary}
    with patch.object(app, '_mirror_recipe_image', side_effect=app.requests.HTTPError('fixture', response=Mock(status_code=404))), \
         patch.object(app, '_mark_mirror_failed_in_db') as mark, \
         patch.object(app, '_image_mirror_budget_left', return_value=10):
        app._ensure_owned_saved_recipe_image(Mock(), 'owner', row)
    assert mark.call_args.kwargs['failure_kind'] == 'transient'


def test_failure_metadata_roundtrips_real_mysql_without_touching_another_owner():
    port = os.getenv('TREPO_TEXT_JOB_TEST_PORT')
    if not port:
        pytest.skip('Requires explicit local MySQL fixture container')
    app = load_app_module()
    conn = pymysql.connect(host='127.0.0.1', port=int(port), user='root',
        database='trepo_text_jobs_test', autocommit=True,
        cursorclass=pymysql.cursors.DictCursor)
    suffix = uuid.uuid4().hex
    legacy, shared = 'fixture_mirror_' + suffix, 'fixture_shared_' + suffix
    class Cursor:
        def __enter__(self):
            self.cursor = conn.cursor()
            return self
        def __exit__(self, *args):
            self.cursor.close()
        def execute(self, sql, params=None):
            return self.cursor.execute(sql.replace('`shared_saved_recipes`', '`'+shared+'`'), params)
    class Connection:
        def cursor(self): return Cursor()
        def commit(self): conn.commit()
    try:
        with conn.cursor() as cur:
            for table in (legacy, shared):
                cur.execute(f'CREATE TABLE `{table}` (_id VARCHAR(36), owner_id VARCHAR(36), image_url TEXT, PRIMARY KEY(_id,owner_id)) ENGINE=InnoDB')
                cur.execute(f'INSERT INTO `{table}` VALUES (%s,%s,%s)', ['recipe','owner','https://fixture.invalid/bad.jpg'])
                cur.execute(f'INSERT INTO `{table}` VALUES (%s,%s,%s)', ['other','other','https://fixture.invalid/other.jpg'])
        with patch.object(app, '_saved_recipes_table', return_value=legacy):
            app._mark_mirror_failed_in_db(Connection(), 'owner', 'recipe',
                failure_kind='source_unavailable', source_url='https://fixture.invalid/bad.jpg')
            # Repeat exercises duplicate-column handling with the actual driver.
            app._mark_mirror_failed_in_db(Connection(), 'owner', 'recipe',
                failure_kind='source_unavailable', source_url='https://fixture.invalid/bad.jpg')
        with conn.cursor() as cur:
            for table in (legacy, shared):
                cur.execute(f'SELECT * FROM `{table}` WHERE _id=%s', ['recipe'])
                assert app._row_mirror_failed_recently(cur.fetchone())
                cur.execute(f'SELECT * FROM `{table}` WHERE _id=%s', ['other'])
                row = cur.fetchone()
                assert row['image_mirror_failure'] is None and row['image_mirror_failed_at'] is None
            cur.execute(f'DROP TABLE `{legacy}`')
            cur.execute(f'UPDATE `{shared}` SET image_mirror_failure=NULL, image_mirror_failed_at=NULL')
        with patch.object(app, '_saved_recipes_table', return_value=legacy):
            app._mark_mirror_failed_in_db(Connection(), 'owner', 'recipe',
                failure_kind='source_unavailable', source_url='https://fixture.invalid/bad.jpg')
        with conn.cursor() as cur:
            cur.execute(f'SELECT * FROM `{shared}` WHERE _id=%s', ['recipe'])
            assert app._row_mirror_failed_recently(cur.fetchone()), 'Missing legacy table lost shared marker'
    finally:
        with conn.cursor() as cur:
            for table in (legacy, shared): cur.execute(f'DROP TABLE IF EXISTS `{table}`')
        conn.close()


def test_known_unavailable_image_returns_placeholder_even_without_request_budget():
    app = load_app_module()
    url = 'https://fixture.invalid/dead.jpg'
    row = {'_id': 'fixture', 'image_url': url, 'image_urls': [url],
           'source_image_url': url, 'source_image_urls': [url],
           'image_mirror_failed_at': datetime.now(),
           'image_mirror_failure': {'kind': 'source_unavailable', 'url_hash': app._sha256(url)}}
    with patch.object(app, '_image_mirror_budget_left', return_value=0), \
         patch.object(app, '_mirror_recipe_image') as mirror, \
         patch.object(app, '_update_saved_recipe_image_fields') as update:
        app._ensure_owned_saved_recipe_image(Mock(), 'owner', row)
    assert row['image_url'] is None and row['image_urls'] == []
    assert row['source_image_url'] == url and row['source_image_urls'] == [url]
    mirror.assert_not_called()
    update.assert_not_called()


def test_first_permanent_failure_hides_broken_cover_without_erasing_original():
    app = load_app_module()
    url = 'https://fixture.invalid/dead.jpg'
    row = {'_id': 'fixture', 'image_url': url, 'image_urls': [url], 'source_image_url': url}
    with patch.object(app, '_mirror_recipe_image', side_effect=app.requests.HTTPError('fixture', response=Mock(status_code=404))), \
         patch.object(app, '_mark_mirror_failed_in_db') as mark, \
         patch.object(app, '_image_mirror_budget_left', return_value=10):
        app._ensure_owned_saved_recipe_image(Mock(), 'owner', row)
    assert row['image_url'] is None and row['image_urls'] == []
    assert row['source_image_url'] == url
    assert mark.call_args.kwargs['source_url'] == url


def test_warm_cached_permanent_failure_also_gets_placeholder_without_scraping():
    app = load_app_module()
    url = 'https://fixture.invalid/dead.jpg'
    app._note_mirror_failure(url, permanent=True)
    row = {'_id': 'fixture', 'image_url': url, 'source_url': 'https://fixture.invalid/recipe'}
    with patch.object(app, '_image_mirror_budget_left', return_value=10), \
         patch.object(app, '_mirror_recipe_image') as mirror, \
         patch.object(app, '_extract_content') as extract, \
         patch.object(app, '_mark_mirror_failed_in_db') as mark:
        app._ensure_owned_saved_recipe_image(Mock(), 'owner', row)
    assert row['image_url'] is None
    mirror.assert_not_called()
    extract.assert_not_called()
    assert mark.call_args.kwargs['failure_kind'] == 'source_unavailable'


def test_temporary_failure_retains_image_and_replacement_is_not_hidden():
    app = load_app_module()
    old, new = 'https://fixture.invalid/old.jpg', 'https://fixture.invalid/new.jpg'
    row = {'_id': 'fixture', 'image_url': old, 'image_mirror_failed_at': datetime.now(),
           'image_mirror_failure': {'kind': 'transient', 'url_hash': app._sha256(old)}}
    with patch.object(app, '_image_mirror_budget_left', return_value=0):
        app._ensure_owned_saved_recipe_image(Mock(), 'owner', row)
    assert row['image_url'] == old
    row['image_mirror_failure']['kind'] = 'source_unavailable'
    row['image_url'] = new
    with patch.object(app, '_image_mirror_budget_left', return_value=0):
        app._ensure_owned_saved_recipe_image(Mock(), 'owner', row)
    assert row['image_url'] == new
