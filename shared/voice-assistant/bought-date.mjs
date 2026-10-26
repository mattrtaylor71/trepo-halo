// Calendar dates, not timestamps. Resolve relative dates once, before retaining
// the mutation request; a retry tomorrow must replay the original correction.
export const hasBoughtDate = value => Object.hasOwn(value || {}, 'bought_date');
const invalid = () => Object.assign(new Error('What exact bought date should I use? Choose a past date, such as October 19, 2026, or say yesterday.'), {statusCode:400,details:{type:"invalid_bought_date"}});
const iso = date => date.toISOString().slice(0,10);
const monthNames = ['january','february','march','april','may','june','july','august','september','october','november','december'];
const monthPattern = '(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)';
const namedDatePattern = new RegExp(`^(${monthPattern})\\.? (\\d{1,2})(?:st|nd|rd|th)?(?:,? (\\d{4}))?$`);
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
    const named=text.match(namedDatePattern);
    if (ago) {
      const n=Number(ago[1]) || ({one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10,a:1,an:1}[ago[1]]) || 0;
      if(n>36500) throw invalid();
      day.setUTCDate(day.getUTCDate()-n*(ago[2].startsWith('week')?7:1));result=iso(day);
    } else if (weekday) {
      const index=['sunday','monday','tuesday','wednesday','thursday','friday','saturday'].indexOf(weekday[1]);
      day.setUTCDate(day.getUTCDate()-((day.getUTCDay()-index+7)%7 || 7));result=iso(day);
    } else if (named) {
      const month=monthNames.findIndex(name=>name.slice(0,3)===named[1].slice(0,3))+1;
      const suffix=`${String(month).padStart(2,'0')}-${named[2].padStart(2,'0')}`;
      // Past purchase + omitted year means its most recent calendar occurrence.
      // Do not invent a future purchase or reinterpret an explicit year.
      let year=Number(named[3] || parts.year);
      if (!named[3] && `${year}-${suffix}`>today) year--;
      result=`${year}-${suffix}`;
      if (!named[3] && suffix==='02-29') {
        while (year>=1900 && iso(new Date(`${result}T12:00:00Z`))!==result) result=`${--year}-${suffix}`;
      }
    } else throw invalid();
  }
  const parsed=new Date(`${result}T12:00:00Z`);
  if (!Number.isFinite(parsed.getTime()) || iso(parsed)!==result || result>today || result<'1900-01-01') throw invalid();
  return result;
}

// Recover only an adjacent, server-loaded year clarification. User words supply
// the item and date; assistant text merely proves it asked for the missing year.
// Never search older turns past a new request, cancellation or topic change.
export function boughtDateYearFollowup(transcript, messages = []) {
  const year=String(transcript).trim().match(/^(\d{4})[.!]?$/)?.[1];
  if (!year) return null;
  const turns=messages.filter(m=>['user','assistant'].includes(m?.role) && typeof m.content==='string');
  const question=turns.at(-1), request=turns.at(-2);
  if (question?.role!=='assistant' || request?.role!=='user' ||
      !/\b(?:what|which) year\b/i.test(question.content)) return null;
  const source=request.content.trim();
  if (!/\b(?:i|we)\s+(?:(?:actually|just|really)\s+)*(?:bought|purchased)\b|\b(?:change|correct|update|set)\b[^.!?]*\b(?:bought|purchase|acquired)\s+date\b/i.test(source) ||
      /\b(?:don['’]t|do not|never|cancel|ignore|instead)\b|[?]/i.test(source)) return null;
  const matches=[...source.matchAll(new RegExp(`\\b(${monthPattern})\\.? (\\d{1,2})(?:st|nd|rd|th)?(?=\\s*[.!]?\\s*$)`, 'gi'))];
  if (matches.length!==1) return null;
  const match=matches[0];
  const month=monthNames.findIndex(name=>name.slice(0,3)===match[1].toLowerCase().slice(0,3))+1;
  const boughtDate=`${year}-${String(month).padStart(2,'0')}-${match[2].padStart(2,'0')}`;
  return {sourceTranscript:source.slice(0,match.index)+`${match[0]}, ${year}`+source.slice(match.index+match[0].length),boughtDate};
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
