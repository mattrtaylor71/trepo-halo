"""Owner-local calendar day from existing device registration evidence.

No scans, writes, global cache, or guessed offsets. Only clock fields are projected;
pagination keys remain in memory. Missing evidence
is unavailable, which callers already render without dropping recipes.
"""
import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo
from meal_reservation_state import ReservationSnapshotUnavailable


def _string(item, name):
    value = item.get(name)
    return value.get('S') if isinstance(value, dict) else None


def _timestamp(value):
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        value = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value.astimezone(datetime.timezone.utc) if value.tzinfo is not None else None
    except ValueError:
        return None


def reported_calendar_day(items, now):
    if not isinstance(now, datetime.datetime) or now.tzinfo is None:
        raise ReservationSnapshotUnavailable('An aware clock is required')
    if not isinstance(items, list):
        raise ReservationSnapshotUnavailable('Invalid clock evidence')
    candidates = []
    for item in items:
        if not isinstance(item, dict):
            raise ReservationSnapshotUnavailable('Invalid clock evidence')
        if item.get('disabled') not in (None, {'BOOL': False}):
            continue
        zone = _string(item, 'time_zone')
        stamp = _timestamp(_string(item, 'time_zone_updated_at') or _string(item, 'updated_at'))
        if not isinstance(zone, str) or not zone or len(zone) > 100 or stamp is None:
            continue
        if stamp > now + datetime.timedelta(minutes=5):
            continue
        try:
            tz = ZoneInfo(zone)
        except (ValueError, KeyError):
            continue
        candidates.append((stamp, zone, tz))
    if not candidates:
        raise ReservationSnapshotUnavailable('Owner calendar clock unavailable')
    newest = max(value[0] for value in candidates)
    latest = [value for value in candidates if value[0] == newest]
    days = {now.astimezone(value[2]).date() for value in latest}
    if len(days) != 1:
        raise ReservationSnapshotUnavailable('Conflicting owner calendar clocks')
    return days.pop()


def owner_calendar_day(ddb, owner, now=None, table_name='TrepoDeviceTokens', max_items=128):
    if not isinstance(owner, str) or not owner or len(owner) > 128:
        raise ReservationSnapshotUnavailable('Invalid calendar owner')
    if type(max_items) is not int or not 1 <= max_items <= 128:
        raise ReservationSnapshotUnavailable('Invalid clock lookup bound')
    now = now or datetime.datetime.now(datetime.timezone.utc)
    query = {'TableName': table_name, 'KeyConditionExpression': 'owner_id = :owner',
             'ExpressionAttributeValues': {':owner': {'S': owner}}, 'ConsistentRead': True,
             'ProjectionExpression': 'owner_id, disabled, time_zone, time_zone_updated_at, updated_at'}
    items, cursors = [], set()
    try:
        while True:
            query['Limit'] = min(32, max_items - len(items) + 1)
            response = ddb.query(**query)
            rows = response.get('Items')
            if not isinstance(rows, list) or any(not isinstance(row, dict) or row.get('owner_id') != {'S': owner} for row in rows):
                raise ReservationSnapshotUnavailable('Invalid clock lookup response')
            items.extend(rows)
            if len(items) > max_items:
                raise ReservationSnapshotUnavailable('Clock lookup incomplete')
            cursor = response.get('LastEvaluatedKey')
            if not cursor:
                return reported_calendar_day(items, now)
            import json
            fingerprint = json.dumps(cursor, sort_keys=True)
            if fingerprint in cursors or len(cursors) >= 7:
                raise ReservationSnapshotUnavailable('Clock lookup incomplete')
            cursors.add(fingerprint)
            query['ExclusiveStartKey'] = cursor
    except ReservationSnapshotUnavailable:
        raise
    except Exception as error:
        raise ReservationSnapshotUnavailable('Owner calendar clock unavailable') from error


def current_owner_day(owner):
    """Bound service lookups; SDK failures remain ordinary unavailable evidence."""
    try:
        import boto3
        from botocore.config import Config
        client = boto3.client('dynamodb', config=Config(
            connect_timeout=2, read_timeout=2, retries={'total_max_attempts': 1}))
        return owner_calendar_day(client, owner)
    except ReservationSnapshotUnavailable:
        raise
    except Exception as error:
        raise ReservationSnapshotUnavailable('Owner calendar clock unavailable') from error
