'use strict';
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, QueryCommand } = require('@aws-sdk/lib-dynamodb');

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const TABLE = process.env.TABLE_NAME || 'TrepoAnalyticsEvents';
const TOKEN = process.env.FEED_TOKEN || '';

// Known-benign backend_error classes that would drown the "something's wrong" signal:
// master_feed dup-PK race (harmless idempotency), and auth code-send failures (a user
// typing a bad phone number). Everything else is a real problem worth seeing.
// Known-benign / self-recovering backend_error classes that would drown the
// "something's wrong" signal. Everything else is a real problem worth seeing.
function isBenign(e) {
  const err = e.error || '';
  if (e.op === 'master_feed_write' && /Duplicate entry|1062/.test(err)) return true;   // idempotency race
  if (e.service === 'auth' && (e.op === 'send_code' || e.op === 'verify_code')) return true; // bad phone #
  if (e.op === 'generate' && /returned \d+ recipes \(need \d+\)/.test(err)) return true; // mealplan count-retry recovers
  return false;
}

async function recentErrors(want) {
  const r = await ddb.send(new QueryCommand({
    TableName: TABLE,
    IndexName: 'EventNameIndex',
    KeyConditionExpression: 'event_name = :n',
    ExpressionAttributeValues: { ':n': 'backend_error' },
    ScanIndexForward: false, // newest first
    Limit: 500, // over-fetch so filtering + dedupe still fills the feed
  }));
  const all = (r.Items || []).map((it) => {
    const p = it.properties || {};
    return {
      ts: String(it.ts_id || '').split('#')[0],
      owner: it.owner_id || null,
      service: p.service || 'unknown',
      op: p.op || null,
      code: p.code || null,
      error: p.error || null,
      job_id: p.job_id || null,
    };
  });
  const real = all.filter((e) => !isBenign(e));
  // Dedupe: collapse the same problem (service|op|error-prefix) into one card with a
  // count + the newest timestamp, so a repeating error is one line, not a wall.
  const byKey = new Map();
  for (const e of real) {
    const key = (e.service || '') + '|' + (e.op || '') + '|' + String(e.error || '').slice(0, 80);
    const hit = byKey.get(key);
    if (hit) { hit.n += 1; if (e.ts > hit.ts) hit.ts = e.ts; }
    else { byKey.set(key, { ...e, n: 1 }); }
  }
  const deduped = Array.from(byKey.values()).sort((a, b) => (a.ts < b.ts ? 1 : -1));
  return { events: deduped.slice(0, want), filtered: all.length - real.length };
}

exports.handler = async (event) => {
  const qs = (event && event.queryStringParameters) || {};
  if (!TOKEN || qs.k !== TOKEN) {
    return { statusCode: 403, headers: { 'content-type': 'text/plain' }, body: 'Forbidden' };
  }
  if (qs.format === 'json') {
    let events = []; let filtered = 0; let err = null;
    try { const r = await recentErrors(120); events = r.events; filtered = r.filtered; }
    catch (e) { err = String(e && e.message || e); }
    return {
      statusCode: 200,
      headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
      body: JSON.stringify({ events, count: events.length, filtered, now: new Date().toISOString(), error: err }),
    };
  }
  return {
    statusCode: 200,
    headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' },
    body: PAGE(qs.k),
  };
};

function PAGE(token) {
  return `<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0f1512">
<title>Trepo · Errors</title>
<style>
  :root{
    --bg:#0e1411; --panel:#161d19; --panel2:#1b231e; --line:#28322c;
    --ink:#e8efea; --soft:#9aa8a1; --faint:#63706a;
    --accent:#3fae86; --accent-dim:#1f3a30;
    --sev:#e06a5a; --sev-dim:#3a201c;
    --voice:#c9903f; --recipes:#7a9ce0; --capture:#e06a5a; --kitchen:#5fbf9a; --bulk:#d17ab0; --unknown:#8a97910;
    --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
    --sans:-apple-system,system-ui,"Segoe UI",Roboto,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased}
  body{padding:max(env(safe-area-inset-top),10px) 0 calc(env(safe-area-inset-bottom) + 24px)}
  header{position:sticky;top:0;z-index:5;background:linear-gradient(180deg,#0e1411 70%,rgba(14,20,17,.0));
    padding:8px 16px 12px;display:flex;align-items:center;gap:10px}
  .logo{font-weight:800;letter-spacing:-.01em;font-size:19px}
  .logo b{color:var(--accent)}
  .live{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--soft);margin-left:auto;font-variant-numeric:tabular-nums}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 0 var(--accent);animation:pulse 2.4s infinite}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(63,174,134,.5)}70%{box-shadow:0 0 0 7px rgba(63,174,134,0)}100%{box-shadow:0 0 0 0 rgba(63,174,134,0)}}
  @media (prefers-reduced-motion:reduce){.dot{animation:none}}
  .bar{display:flex;gap:8px;padding:0 16px 12px;align-items:center;flex-wrap:wrap}
  .count{font-size:13px;color:var(--soft)} .count b{color:var(--ink);font-variant-numeric:tabular-nums}
  .feed{display:flex;flex-direction:column;gap:8px;padding:0 12px}
  .card{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--unknown);border-radius:13px;
    padding:12px 13px;display:flex;flex-direction:column;gap:6px}
  .card.s-voice{border-left-color:var(--voice)} .card.s-recipes{border-left-color:var(--recipes)}
  .card.s-capture{border-left-color:var(--capture)} .card.s-bulk{border-left-color:var(--capture)}
  .card.s-kitchen{border-left-color:var(--kitchen)} .card.s-grocery{border-left-color:var(--kitchen)}
  .r1{display:flex;align-items:center;gap:8px}
  .svc{font-family:var(--mono);font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
    padding:3px 7px;border-radius:6px;background:var(--panel2);color:var(--soft);white-space:nowrap}
  .op{font-weight:650;font-size:14.5px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
  .mult{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--sev);background:var(--sev-dim);
    padding:2px 6px;border-radius:6px;white-space:nowrap}
  .ago{font-size:12px;color:var(--faint);font-variant-numeric:tabular-nums;white-space:nowrap}
  .msg{font-family:var(--mono);font-size:12px;line-height:1.45;color:var(--soft);
    background:var(--panel2);border-radius:8px;padding:8px 9px;word-break:break-word;max-height:4.6em;overflow:hidden;position:relative}
  .msg.open{max-height:none}
  .meta{display:flex;gap:10px;font-size:11px;color:var(--faint);flex-wrap:wrap}
  .meta code{font-family:var(--mono)}
  .empty{text-align:center;padding:70px 24px;color:var(--soft)}
  .empty .big{font-size:44px;margin-bottom:10px}
  .empty .t{font-size:16px;font-weight:650;color:var(--ink)}
  .empty .s{font-size:13px;margin-top:4px}
  .err{color:var(--sev);font-size:12px;padding:0 16px}
</style></head>
<body>
  <header>
    <div class="logo">Trepo <b>Errors</b></div>
    <div class="live"><span class="dot"></span><span id="upd">live</span></div>
  </header>
  <div class="bar"><div class="count"><b id="n">0</b> errors · newest first<span id="filt"></span></div></div>
  <div id="err" class="err"></div>
  <div id="feed" class="feed"></div>

<script>
  var TOKEN=${JSON.stringify(token)};
  var SEV={};
  function ago(iso){
    var t=Date.parse(iso); if(isNaN(t)) return '';
    var s=Math.max(0,(Date.now()-t)/1000);
    if(s<60) return Math.floor(s)+'s';
    if(s<3600) return Math.floor(s/60)+'m';
    if(s<86400) return Math.floor(s/3600)+'h';
    return Math.floor(s/86400)+'d';
  }
  function esc(s){return String(s==null?'':s).replace(/[&<>]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;'}[c]})}
  function shortOwner(o){ if(!o) return ''; o=String(o); return o.length>12 ? o.slice(0,8) : o; }
  function render(d){
    document.getElementById('err').textContent = d.error ? ('query error: '+d.error) : '';
    var ev=d.events||[];
    document.getElementById('n').textContent = ev.length;
    document.getElementById('filt').textContent = d.filtered ? ('  ·  '+d.filtered+' benign hidden') : '';
    document.getElementById('upd').textContent = 'updated '+new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
    var feed=document.getElementById('feed');
    if(!ev.length){ feed.innerHTML='<div class="empty"><div class="big">✓</div><div class="t">All clear</div><div class="s">No backend errors in the recent window.</div></div>'; return; }
    feed.innerHTML = ev.map(function(e){
      var svc=(e.service||'unknown').toLowerCase();
      var op = e.op || e.code || 'error';
      var owner = e.owner && String(e.owner).indexOf('backend#')!==0 ? '<code>'+esc(shortOwner(e.owner))+'</code>' : '';
      var code = e.code && e.code!==e.op ? '<span>'+esc(e.code)+'</span>' : '';
      var job = e.job_id ? '<span>job '+esc(String(e.job_id).slice(0,10))+'</span>' : '';
      var mult = (e.n && e.n>1) ? '<span class="mult">×'+e.n+'</span>' : '';
      return '<div class="card s-'+esc(svc)+'">'+
        '<div class="r1"><span class="svc">'+esc(svc)+'</span><span class="op">'+esc(op)+'</span>'+mult+'<span class="ago">'+ago(e.ts)+'</span></div>'+
        (e.error?'<div class="msg" onclick="this.classList.toggle(\\'open\\')">'+esc(e.error)+'</div>':'')+
        ((owner||code||job)?'<div class="meta">'+owner+code+job+'</div>':'')+
      '</div>';
    }).join('');
  }
  function load(){
    fetch('?format=json&k='+encodeURIComponent(TOKEN),{cache:'no-store'})
      .then(function(r){return r.json()}).then(render)
      .catch(function(e){document.getElementById('err').textContent='fetch error: '+e});
  }
  load(); setInterval(load, 20000);
  document.addEventListener('visibilitychange',function(){ if(!document.hidden) load(); });
</script>
</body></html>`;
}
