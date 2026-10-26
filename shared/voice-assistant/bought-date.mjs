// Calendar dates, not timestamps. Resolve relative dates once, before retaining
// the mutation request; a retry tomorrow must replay the original correction.
export const hasBoughtDate = value => Object.hasOwn(value || {}, 'bought_date');
const invalid = () => Object.assign(new Error('What exact bought date should I use? Choose a past date, such as October 19, 2026, or say yesterday.'), {statusCode:400,details:{type:"invalid_bought_date"}});
const iso = date => date.toISOString().slice(0,10);
export function normalizeBoughtDate(value, {now = new Date(), timeZone = 'America/Los_Angeles'} = {}) {
  if (value === null) return null; // Only an explicit request to forget the date.
  if (typeof value !== 'string' || !value.trim() || value.length > 100) throw invalid();
  const parts = Object.fromEntries(new Intl.DateTimeFormat('en-US', {timeZone,year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(now).map(p=>[p.type,p.value]));
  const today = `${parts.year}-${parts.month}-${parts.day}`;
  const day = new Date(`${today}T12:00:00Z`);
  let text = value.toLowerCase().trim(), result;
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) result = text;
  else if (text === 'today') result = today;
  else if (text === 'yesterday') { day.setUTCDate(day.getUTCDate()-1); result=iso(day); }
  else {
    const ago=text.match(/^(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an) (days?|weeks?) ago$/);
    const weekday=text.match(/^last (sunday|monday|tuesday|wednesday|thursday|friday|saturday)$/);
    const named=text.match(/^(january|february|march|april|may|june|july|august|september|october|november|december) (\d{1,2})(?:st|nd|rd|th)?[,]? (\d{4})$/);
    if (ago) {
      const n=Number(ago[1]) || ({one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10,a:1,an:1}[ago[1]]) || 0;
      if(n>36500) throw invalid();
      day.setUTCDate(day.getUTCDate()-n*(ago[2].startsWith('week')?7:1));result=iso(day);
    } else if (weekday) {
      const index=['sunday','monday','tuesday','wednesday','thursday','friday','saturday'].indexOf(weekday[1]);
      day.setUTCDate(day.getUTCDate()-((day.getUTCDay()-index+7)%7 || 7));result=iso(day);
    } else if (named) {
      const month=['january','february','march','april','may','june','july','august','september','october','november','december'].indexOf(named[1])+1;
      result=`${named[3]}-${String(month).padStart(2,'0')}-${named[2].padStart(2,'0')}`;
    } else throw invalid();
  }
  const parsed=new Date(`${result}T12:00:00Z`);
  if (!Number.isFinite(parsed.getTime()) || iso(parsed)!==result || result>today || result<'1900-01-01') throw invalid();
  return result;
}

export async function householdBoughtDate(connection, household, value) {
  const [rows]=await connection.execute('SELECT time_zone FROM kitchen_shelf_calendars WHERE household_id=?',[String(household)]);
  return normalizeBoughtDate(value,{timeZone:rows[0]?.time_zone || 'America/Los_Angeles'});
}

// An explicitly unknown date stays unknown. Check-in time is never a bought date.
export function boughtDateFields(row) {
  return Object.hasOwn(row,'acquired_on') && row.acquired_on !== undefined
    ? {bought_date:row.acquired_on === 'null' ? null : row.acquired_on} : {};
}

// Re-resolve membership for a correction; model IDs and cached member lists
// cannot widen this scope. This is also used by older deployed Thyme packages.
export async function householdCorrectionRows(connection, context) {
  const actor=String(context?.userId || context?.tableOwnerId || '').trim();
  if (!actor || context?.isFallbackContext) throw Object.assign(new Error('Kitchen membership unavailable.'),{statusCode:403});
  const [rows]=await connection.execute(`SELECT k.*, actor.user_id AS _correction_actor,
    COALESCE(NULLIF(actor.owner_id,''),actor.user_id) AS _correction_household
    FROM new_users actor LEFT JOIN shared_kitchen k ON k.action='IN' AND k.owner_id IN (
      SELECT CONVERT(member.user_id USING utf8mb4) COLLATE utf8mb4_unicode_ci FROM new_users member
      WHERE member.user_id=actor.user_id OR (actor.owner_id IS NOT NULL AND actor.owner_id<>'' AND member.owner_id=actor.owner_id))
    WHERE actor.user_id=? ORDER BY k._createdDate DESC,k._id`,[actor]);
  if (!rows.length || rows[0]._correction_actor !== actor ||
      (context.ownerId && ![actor,String(rows[0]._correction_household)].includes(String(context.ownerId))))
    throw Object.assign(new Error('Kitchen membership changed. Refresh your household.'),{statusCode:403});
  return rows.filter(row=>row._id!=null);
}
