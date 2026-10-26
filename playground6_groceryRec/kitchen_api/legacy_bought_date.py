"""Opt-in response-only bridge for old iOS purchase-age rendering.

No database writes. Call only after normal authorization and serialization.
The caller must explicitly supply corrected item IDs and a legacy build ceiling.
Old clients also call this timestamp their check-in date; that UI limitation is
intentional and must be disclosed before any rollout beyond an isolated test.
"""
import json
import re
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def project_response(response, event, *, corrected_item_ids=(), max_build=0, now=None):
    headers = {str(k).lower(): str(v) for k, v in (event.get('headers') or {}).items()}
    method = event.get('httpMethod') or (event.get('requestContext') or {}).get('http', {}).get('method')
    path = event.get('rawPath') or event.get('path') or ''
    build = headers.get('x-trepo-app-build', '')
    if (not corrected_item_ids or method != 'GET' or response.get('statusCode') != 200
            or not re.fullmatch(r'(?:/[^/]+)?/kitchen/[0-9a-fA-F-]{36}/?', path)
            or headers.get('x-trepo-app-platform') != 'ios'
            or not re.fullmatch(r'[1-9][0-9]{0,7}', build)
            or int(build) > max_build
            or headers.get('x-trepo-kitchen-age') == 'bought-date-v1'):
        return response
    try:
        zone = ZoneInfo(headers['x-trepo-time-zone'])
        payload = json.loads(response['body'])
    except (KeyError, TypeError, ValueError, ZoneInfoNotFoundError):
        return response
    if not isinstance(payload, dict) or not isinstance(payload.get('items'), list):
        return response
    today = (now or datetime.now(timezone.utc)).astimezone(zone).date()
    eligible = set(corrected_item_ids)
    changed = False
    for row in payload['items']:
        if not isinstance(row, dict) or row.get('_id') not in eligible:
            continue
        if str(row.get('analysis_stage', '')).lower() not in {'final', 'ready', 'complete', 'completed'}:
            continue  # Backdating must not suppress pending-capture processing.
        bought = row.get('acquired_on')
        try:
            acquired = date.fromisoformat(bought)
            if acquired.isoformat() != bought or acquired > today:
                continue
        except (TypeError, ValueError):
            continue  # Clearing/unknown dates require the new client to display honestly.
        row['checked_in_at'] = row.get('_createdDate')
        row['_createdDate'] = datetime.combine(acquired, time.min, zone).astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        row['days_old'] = (today - acquired).days
        row['age_display_compat'] = 'bought-date-v1'
        changed = True
    if not changed:
        return response
    return {**response, 'headers': {**(response.get('headers') or {}), 'Cache-Control': 'no-store'},
            'body': json.dumps(payload, separators=(',', ':'))}
