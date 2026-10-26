import copy
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legacy_bought_date import project_response


class LegacyBoughtDateTests(unittest.TestCase):
    def setUp(self):
        self.item = {'_id': 'fixture', '_createdDate': '2026-10-26T22:00:00Z',
                     'acquired_on': '2026-10-20', 'analysis_stage': 'final', 'quantity_value': 1}
        self.response = {'statusCode': 200, 'headers': {'Content-Type': 'application/json'},
                         'body': json.dumps({'items': [self.item, {**self.item, '_id': 'unrelated'}]})}
        self.event = {'rawPath': '/kitchen/8fcb37c1-0927-4c31-856a-5b761efc226c',
                      'requestContext': {'http': {'method': 'GET'}}, 'headers': {
                          'x-trepo-app-platform': 'ios', 'x-trepo-app-build': '150',
                          'x-trepo-time-zone': 'America/Los_Angeles'}}
        self.now = datetime(2026, 10, 26, 23, tzinfo=timezone.utc)

    def project(self, response=None, event=None, **kw):
        return project_response(response or self.response, event or self.event,
                                now=self.now, corrected_item_ids=['fixture'], max_build=150, **kw)

    def test_exact_item_and_response_only(self):
        original = copy.deepcopy(self.response)
        result = self.project(); rows = json.loads(result['body'])['items']
        self.assertEqual(rows[0]['_createdDate'], '2026-10-20T07:00:00Z')
        self.assertEqual(rows[0]['days_old'], 6)
        self.assertEqual(rows[0]['checked_in_at'], self.item['_createdDate'])
        self.assertEqual(rows[0]['acquired_on'], '2026-10-20')
        self.assertEqual(rows[0]['quantity_value'], 1)
        self.assertEqual(rows[1], {**self.item, '_id': 'unrelated'})
        self.assertEqual(self.response, original)
        self.assertEqual(result['headers']['Cache-Control'], 'no-store')

    def test_disabled_by_default(self):
        self.assertIs(project_response(self.response, self.event), self.response)

    def test_client_gates(self):
        for key, value in [('x-trepo-app-build', '151'), ('x-trepo-app-build', 'garbage'),
                           ('x-trepo-app-platform', 'web'), ('x-trepo-time-zone', 'invalid'),
                           ('x-trepo-kitchen-age', 'bought-date-v1')]:
            event = copy.deepcopy(self.event); event['headers'][key] = value
            self.assertIs(self.project(event=event), self.response)
        for key in ['x-trepo-app-build', 'x-trepo-app-platform', 'x-trepo-time-zone']:
            event = copy.deepcopy(self.event); del event['headers'][key]
            self.assertIs(self.project(event=event), self.response)

    def test_method_path_auth_and_error_gates(self):
        for method in ['POST', 'PATCH', 'DELETE']:
            event = copy.deepcopy(self.event); event['requestContext']['http']['method'] = method
            self.assertIs(self.project(event=event), self.response)
        event = {**self.event, 'rawPath': self.event['rawPath'] + '/fixture'}
        self.assertIs(self.project(event=event), self.response)
        for status in [401, 403, 404, 500]:
            response = {**self.response, 'statusCode': status}
            self.assertIs(self.project(response=response), response)

    def test_unknown_future_invalid_and_pending_untouched(self):
        for patch in [{'acquired_on': None}, {'acquired_on': '2026-10-27'}, {'acquired_on': '2026-02-30'},
                      {'acquired_on': '2026-1-20'}, {'analysis_stage': 'processing'}, {'analysis_stage': None}]:
            response = {**self.response, 'body': json.dumps({'items': [{**self.item, **patch}]})}
            self.assertIs(self.project(response=response), response)

    def test_local_civil_midnight_and_dst(self):
        for zone, bought, expected in [('Asia/Kathmandu', '2026-10-20', '2026-10-19T18:15:00Z'),
                                      ('Pacific/Kiritimati', '2026-10-20', '2026-10-19T10:00:00Z'),
                                      ('America/Los_Angeles', '2026-03-08', '2026-03-08T08:00:00Z')]:
            event = copy.deepcopy(self.event); event['headers']['x-trepo-time-zone'] = zone
            response = {**self.response, 'body': json.dumps({'items': [{**self.item, 'acquired_on': bought}]})}
            row = json.loads(self.project(response=response, event=event)['body'])['items'][0]
            self.assertEqual(row['_createdDate'], expected)

    def test_header_case_and_bad_payloads(self):
        event = copy.deepcopy(self.event); event['headers'] = {k.upper(): v for k, v in event['headers'].items()}
        self.assertNotEqual(self.project(event=event), self.response)
        for body in ['oops', 'null', '[]', '{}', '{"items":null}']:
            response = {**self.response, 'body': body}
            self.assertIs(self.project(response=response), response)


if __name__ == '__main__':
    unittest.main()
