"""Versioned shelf-life evidence. Pure functions; no model, database or network.

A calendar is pinned when a lifecycle is created. Time passing changes countdowns,
never evidence or the deadline. Recorded labels remain separate from estimates.
Unknown acquisition/opening/freezing history must not be replaced with today.
"""
import copy
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

VERSION = 'shelf-lifecycle-20260924-v1'
IDENTITY = ('product_name', 'brand', 'variant', 'category')


def object_value(value):
    if isinstance(value, dict): return copy.deepcopy(value)
    try:
        parsed = json.loads(value or '{}')
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError): return {}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(',', ':')).encode()).hexdigest()


def civil(value, zone):
    if not value: return None
    try:
        raw = str(value).strip()
        if len(raw) == 10: return date.fromisoformat(raw)
        instant = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        if instant.tzinfo is None: instant = instant.replace(tzinfo=timezone.utc)
        return instant.astimezone(ZoneInfo(zone)).date()
    except (ValueError, TypeError, OverflowError): return None


def truth(value):
    return value is True or value == 1 or (isinstance(value, str) and value.lower() in ('1', 'true'))


def location(value):
    value = str(value or '').strip().lower()
    return {'refrigerated':'fridge', 'frozen':'freezer', 'counter':'pantry'}.get(value, value) or None


def valid_days(value):
    if isinstance(value, bool): return None
    try:
        n = int(value)
        if float(value) != n or not 1 <= n <= 730: return None
        return n
    except (ValueError, TypeError, OverflowError): return None


def duration(guidance, opened, storage):
    scenarios = object_value(guidance.get('scenarios'))
    key = ('opened' if opened else 'sealed') + '_' + str(storage)
    scenario = object_value(scenarios.get(key))
    if scenarios:
        # A missing/unusable scenario is not permission to borrow another one.
        if not scenario: return None
        low, high = scenario.get('min_days'), scenario.get('max_days')
    else:
        if guidance.get('timing_start') == 'after_opening' and not opened: return None
        zone = location(guidance.get('storage_zone'))
        if zone not in (None, 'mixed', storage): return None
        low, high = guidance.get('min_days', guidance.get('minDays')), guidance.get('max_days', guidance.get('maxDays'))
    high = valid_days(high)
    if high is None: return None
    if low is not None:
        try:
            if isinstance(low, bool) or float(low) != int(low) or not 0 <= int(low) <= high: return None
        except (ValueError, TypeError, OverflowError): return None
    return high


def advance(row, previous=None, *, now=None, time_zone='America/Los_Angeles'):
    """Accept an observed item revision. Repeating it is idempotent.

    Human event timestamps come from the existing protected edit sidecar, never
    analysis observation time. Freezer exit without explicit thaw guidance is
    unknown instead of reusing the original check-in deadline.
    """
    now = now or datetime.now(timezone.utc)
    prior = object_value(previous)
    zone = prior.get('time_zone') or time_zone
    ZoneInfo(zone)  # invalid configuration must fail visibly
    today = now.astimezone(ZoneInfo(zone)).date()
    meta = object_value(row.get('_shelf_edits'))
    history = object_value(meta.get('__shelf_history'))
    corrected = bool(meta.get('__shelf_clock_correction') and meta.get('__shelf_clock_correction') != prior.get('clock_correction'))
    identity = fingerprint({k: row.get(k) for k in IDENTITY})
    same_identity = prior.get('identity_hash') == identity
    guidance = object_value(row.get('storage_guidance'))
    stage = row.get('analysis_stage')
    ready = stage in (None, '', 'final') and row.get('analysis_status') not in ('pending', 'processing', 'failed')
    accepted = prior.get('guidance') if same_identity else None
    accepted_hash = prior.get('accepted_guidance_hash') if same_identity else None
    # Identity corrections must wait for genuinely new guidance, not reinterpret
    # the previous product's unchanged analysis as evidence for its new name.
    established_identity = bool(prior.get('accepted_guidance_hash') or prior.get('identity_guidance_pending') or
                                any(k in meta for k in IDENTITY))
    stale_identity_guidance = bool(prior and established_identity and (not same_identity or prior.get('identity_guidance_pending')) and
                                  guidance.get('analysis_identity_hash') != identity)
    if (not accepted and ready and guidance and not stale_identity_guidance
            and any(duration(guidance,opened,place) is not None for opened in (False,True) for place in ('fridge','freezer','pantry'))):
        accepted, accepted_hash = guidance, fingerprint(guidance)
    accepted = object_value(accepted)
    stored = location(row.get('storage_location'))
    # Unknown storage is not silently a refrigerator.
    if stored is None: stored = location(accepted.get('storage_zone'))
    if stored not in ('fridge', 'freezer', 'pantry'): stored = None
    opened_at = meta.get('__opened_at') if '__opened_at' in meta else row.get('opened_at') or history.get('first_opened_at')
    # Once opened remains opened for timing even if a package was resealed.
    if not corrected:opened_at = prior.get('first_opened_at') or opened_at
    opened = truth(row.get('is_opened')) or bool(opened_at) or (bool(prior.get('ever_opened')) and not (corrected and 'opened_on' in meta.get('__shelf_corrected_keys',[])))
    frozen_at = meta.get('__storage_started_at') if '__storage_started_at' in meta else row.get('storage_started_at')
    if stored == 'freezer' and not frozen_at and prior.get('storage') == 'freezer' and not corrected:
        frozen_at = prior.get('freezing_at')
    thawed = bool(prior.get('thawed') or history.get('thawed_at') or
                  (prior.get('storage') == 'freezer' and stored != 'freezer'))
    recorded = civil(row.get('product_expiration'), zone)
    start, clock, days = None, None, None
    why = 'missing_guidance'
    if stored == 'freezer':
        start, clock = civil(frozen_at, zone), 'storage_started_at'
        why = 'missing_freezing_date'
    elif thawed:
        # Ordinary six-scenario guidance has no thawing model. Do not assume
        # freezing reset the item or extrapolate a safe thaw duration from it.
        clock, why = 'thawed_at', 'needs_thaw_guidance'
    elif opened:
        start, clock = civil(opened_at, zone), 'opened_at'
        why = 'missing_opening_date'
    else:
        if '__cooked_on' in meta:
            start, clock = civil(meta.get('__cooked_on'), zone), 'cooked_on'
        elif '__acquired_on' in meta:
            start, clock = civil(meta.get('__acquired_on'), zone), 'acquired_on'
        else:
            start, clock = civil(row.get('_createdDate'), zone), 'check_in'
        why = 'missing_check_in_date'
    if start:
        if start > today: why = 'invalid_event_date'
        elif not accepted: why = 'identity_needs_guidance' if stale_identity_guidance else 'missing_guidance'
        elif not stored: why = 'unknown_storage'
    if start and start <= today and accepted and stored:
        days = duration(accepted, opened, stored)
        # Retain the established guard against implausibly short dry-food model
        # guidance. A fresh item stored on the counter is not a dry-food category.
        category = str(row.get('category') or '').lower()
        if days is not None and days < 7 and any(c in category for c in ('pantry','spice','season','baking')):
            days = None
        if days is None: why = 'unsupported_storage_guidance'
    estimate = start + timedelta(days=days) if days is not None else None
    active = row.get('action', 'IN') == 'IN'
    # A recorded date is preserved as a label. Outside the freezer, an evidenced
    # shorter opened estimate may tighten it. In the freezer the current known
    # freezing estimate governs, while the original recorded label remains visible.
    deadline, basis = estimate, 'estimate' if estimate else 'unknown'
    if stored != 'freezer' and not thawed and recorded:
        if not opened or estimate is None or recorded <= estimate:
            deadline, basis = recorded, 'recorded_date'
    pre_freeze_date = prior.get('pre_freeze_date') if prior.get('storage') == 'freezer' else None
    pre_freeze_basis = prior.get('pre_freeze_basis') if pre_freeze_date else None
    if stored == 'freezer' and prior.get('storage') != 'freezer' and start:
        old_date = civil(prior.get('deadline'),zone)
        if old_date and old_date < start:
            pre_freeze_date, pre_freeze_basis = old_date.isoformat(), prior.get('basis')
    if stored == 'freezer' and pre_freeze_date:
        deadline, basis = civil(pre_freeze_date,zone), pre_freeze_basis
    proof = object_value(meta.get('__expiration_provenance'))
    def instant(value):
        try:
            value = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
            return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
        except (ValueError, TypeError): return None
    edited = instant(proof.get('observed_at'))
    transitions = [instant(v) for v in (opened_at, frozen_at, history.get('thawed_at'))]
    # Bind a manual date to the state it was actually chosen for. Comparing
    # timestamps alone fails when a later correction says "opened yesterday".
    proof_hash = fingerprint(proof) if proof else None
    current_context = fingerprint({'identity':identity,'storage':stored,'opened':opened,
        'opened_at':str(opened_at) if opened_at else None,
        'frozen_at':str(frozen_at) if frozen_at else None,'thawed':thawed})
    proof_context = (prior.get('date_override_context') if proof_hash and prior.get('date_proof_hash') == proof_hash
                     else current_context if proof_hash else None)
    # An explicit date chosen for the CURRENT state is authoritative. A later
    # opening/storage change reevaluates it instead of preserving a stale override.
    if (recorded and type(proof.get('version')) is int and proof.get('version') == 1 and proof.get('source') == 'item_edit'
            and proof.get('value') == recorded.isoformat() and edited
            and proof_context == current_context
            and all(edited >= t for t in transitions if t)):
        deadline, basis = recorded, 'recorded_date'
    if not active: deadline, basis, why = None, 'unknown', 'removed'
    pending = row.get('analysis_status') in ('pending','processing') or (stage in ('fast','preliminary') and row.get('analysis_status') not in ('ready','failed'))
    status = 'known' if deadline else ('pending' if pending else 'needs_information')
    if not active: status = 'removed'
    reason = ('Recorded date' if basis == 'recorded_date' else {
        'check_in':'Estimated from check-in', 'opened_at':'Estimated from opening',
        'acquired_on':'Estimated from acquisition', 'cooked_on':'Estimated from cooking',
        'storage_started_at':'Estimated from freezing'}.get(clock, 'Estimated shelf life')) if deadline else {
            'missing_opening_date':'Opening date unknown. Set a date to use this item by.',
            'missing_freezing_date':'Freezing date unknown. Check the package and set a date.',
            'needs_thaw_guidance':'Check the thawed item’s guidance and set a date to use it by.',
            'identity_needs_guidance':'Item details changed. Check the package and set a date to use it by.',
            'removed':'Item is no longer in the kitchen.',
        }.get(why, 'Shelf-life information is not available yet.')
    if stored == 'freezer' and pre_freeze_date and deadline and deadline.isoformat() == pre_freeze_date:
        reason = 'Already past its date or estimate before freezing'
    state = {
        'version': VERSION, 'time_zone': zone, 'identity_hash': identity,
        'clock_correction':meta.get('__shelf_clock_correction'),
        'date_proof_hash':proof_hash,'date_override_context':proof_context,
        'raw_guidance_hash': fingerprint(guidance), 'accepted_guidance_hash': accepted_hash,
        'identity_guidance_pending': stale_identity_guidance,
        'guidance': accepted, 'storage': stored, 'ever_opened': opened,
        'first_opened_at': str(opened_at) if opened_at else None, 'thawed': thawed,
        'freezing_at': str(frozen_at) if frozen_at and stored == 'freezer' else None,
        'pre_freeze_date':pre_freeze_date if stored == 'freezer' else None,
        'pre_freeze_basis':pre_freeze_basis if stored == 'freezer' else None,
        'clock_source': clock, 'clock_start': start.isoformat() if start else None,
        'duration_days': days, 'recorded_date': recorded.isoformat() if recorded else None,
        'estimated_date': estimate.isoformat() if estimate else None,
        'deadline': deadline.isoformat() if deadline else None, 'basis': basis,
        'status': status, 'reason': reason, 'unknown_reason': why if not deadline else None,
    }
    semantic = {k:v for k,v in state.items() if k not in ('raw_guidance_hash',)}
    changed = fingerprint(semantic) != fingerprint({k:prior.get(k) for k in semantic})
    state['generation'] = int(prior.get('generation', 0)) + int(changed)
    state['changed_at'] = now.isoformat() if changed else prior.get('changed_at')
    return state


def public_snapshot(state, today=None, now=None):
    """Bounded projection used by item detail, Shelf Life, Clean and reminders."""
    zone = state['time_zone']
    today = today or (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(zone)).date()
    deadline = civil(state.get('deadline'), zone)
    left = (deadline - today).days if deadline else None
    # Missing evidence is item-detail metadata, not a date-based alert. Shipped
    # clients turn the literal `needs_information` into a local attention row,
    # so project it as unknown while retaining the original assessment status.
    # Do not rewrite persisted evidence, clocks, generation, or review history.
    status = 'unknown' if state.get('status') == 'needs_information' else state.get('status')
    needs_review = status == 'known' and left is not None and left <= 2
    return {k:state.get(k) for k in ('version','time_zone','generation','changed_at','deadline',
        'recorded_date','estimated_date','basis','status','reason','clock_source','clock_start','unknown_reason')} | {
            'status':status, 'assessment_status':state.get('status'),
            'evaluated_on':today.isoformat(), 'days_left':left,
            'needs_review':needs_review,
            'clean_priority': max(15, 80 - max(-7, left) * 4) if needs_review else 0,
        }


def review_projection(snapshot, edits, now=None):
    """Review affects attention only. It never changes the shelf assessment."""
    now = now or datetime.now(timezone.utc)
    review = object_value(object_value(edits).get('__shelf_review'))
    valid = review.get('generation') == snapshot.get('generation')
    def future(key):
        try:
            value = datetime.fromisoformat(str(review.get(key)).replace('Z', '+00:00'))
            return value.isoformat() if valid and value.tzinfo and value > now else None
        except (ValueError, TypeError):
            return None
    until = future('clean_until')
    snooze = future('remind_after')
    due_snooze = False
    try:
        stamp = datetime.fromisoformat(str(review.get('remind_after')).replace('Z','+00:00'))
        due_snooze = bool(valid and stamp.tzinfo and 0 <= (now-stamp).total_seconds() < 86400)
    except (ValueError,TypeError): pass
    return dict(snapshot, clean_review_until=until, reminder_snoozed_until=snooze, reminder_snooze_due=due_snooze,
                last_reviewed_at=review.get('reviewed_at') if valid else None,
                review_operation=review.get('operation_id') if valid else None,
                clean_eligible=bool(snapshot.get('needs_review') and not until))


def legacy_watch(row, snapshot):
    """Additive response: retain every shipped ItemToWatch field and route."""
    left = snapshot.get('days_left')
    if left is None or left > 2 or snapshot['status'] != 'known': return None
    recorded = snapshot['basis'] == 'recorded_date'
    reason = snapshot['reason']
    created = civil(row.get('_createdDate'), snapshot['time_zone'])
    clock = civil(snapshot.get('clock_start'), snapshot['time_zone'])
    evaluated = date.fromisoformat(snapshot['evaluated_on'])
    return {'item_id':row['_id'], 'product_name':row.get('product_name') or 'Item',
        'index':0, 'reason':reason, 'urgency':'expired' if left < 0 else 'expiring_soon',
        'days_over':max(0, -left), 'days_left':max(0,left) if left >= 0 else None,
        'days_old':max(0,(evaluated-created).days) if created else None, '_created_date':created.isoformat() if created else None,
        'emoji':(row.get('product_image_url') or '')[6:] if str(row.get('product_image_url') or '').startswith('emoji:') else None,
        'quantity':row.get('quantity'), 'confidence':1.0 if recorded else 0.5,
        'confidence_basis':'recorded_date' if recorded else 'uncalibrated_estimate',
        'clock_source':snapshot['clock_source'], 'clock_age_days':max(0,(evaluated-clock).days) if clock else None,
        'expiration_date':snapshot['recorded_date'] if recorded else None, 'shelf_life_date':snapshot['deadline'],
        'recorded_date':snapshot['recorded_date'],
        'evaluated_on':snapshot['evaluated_on'], 'expiry_source':'user' if recorded else 'estimate',
        'expiration_provenance':row.get('expiration_provenance'), 'shelf_life':snapshot}
