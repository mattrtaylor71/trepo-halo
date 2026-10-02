import fs from 'node:fs';
import assert from 'node:assert/strict';
import { signRequest, id } from '../src/core.mjs';

const root = '/Users/MattTaylor/Library/Caches/trepo-thyme-agent-v2-20261001';
const cfg = JSON.parse(fs.readFileSync(root + '/deployment-private.json'));
const oldSession = JSON.parse(fs.readFileSync(root + '/cloud-session-private.json'));
const secret = fs.readFileSync(root + '/bridge-secret', 'utf8');
const out = process.env.THYME_EVIDENCE;
assert.ok(out, 'Set THYME_EVIDENCE to a private evidence file');
const evidence = [];
async function call(data) {
  const body = JSON.stringify({ ...data, principal: 'qualification-read-only' });
  const timestamp = String(Date.now()), nonce = id();
  const res = await fetch(cfg.url, { method: 'POST', body,
    headers: {'content-type':'application/json', 'x-thyme-time':timestamp, 'x-thyme-nonce':nonce,
      'x-thyme-signature':signRequest({body,timestamp,nonce},secret)},
    signal: AbortSignal.timeout(30000),
  });
  const value = await res.json();
  assert.equal(res.status, 200, JSON.stringify(value));
  return value;
}
async function turn(text, sessionId) {
  const started = Date.now();
  let s = await call({op:'message',text,sessionId,requestId:id(),model:'gpt-6-luna'});
  for (let i=0; i<90 && ['queued','running','verifying'].includes(s.status); i++) {
    await new Promise(r=>setTimeout(r,2000));
    s=await call({op:'session',sessionId:s.id});
  }
  assert.equal(s.status,'completed',JSON.stringify(s.error));
  assert.equal(s.proposals.length,0);
  const answer=s.messages.filter(m=>m.role==='assistant'&&m.phase!=='commentary').at(-1)?.text;
  evidence.push({sessionId:s.id,text,answer,seconds:(Date.now()-started)/1000,agentVersion:s.agentVersion,
    providerId:s.providerId,previousProviderId:s.previousProviderId,recipes:s.recipes.length,proposals:s.proposals.length});
  fs.writeFileSync(out, JSON.stringify(evidence,null,2),{mode:0o600});
  console.log(JSON.stringify({text,answer,seconds:evidence.at(-1).seconds}));
  return s;
}
let fresh=await turn('Can you suggest dinner?');
assert.equal(fresh.recipes.length,0);
assert.ok(evidence.at(-1).answer.includes('?'));
const question=evidence.at(-1).answer;
fresh=await turn('Shopping is fine, for three people, no more than 30 minutes. Just confirm these choices; do not create a recipe or change anything.',fresh.id);
assert.equal(fresh.messages.filter(m=>m.role==='assistant'&&m.text===question).length,1,'Question appeared twice');
// A fresh API read and another durable worker invocation simulate reload/reopen.
fresh=await call({op:'session',sessionId:fresh.id});
fresh=await turn('What choices have I given you for this chat? Just remind me; do not create anything.',fresh.id);
assert.match(evidence.at(-1).answer,/3|three/i);
assert.match(evidence.at(-1).answer,/30|thirty/i);
assert.match(evidence.at(-1).answer,/shop|buy|pick up/i);
const migrated=await turn('Shopping is fine for two people and 25 minutes. Just confirm those choices for this chat. Do not create a recipe or change app data.',oldSession.id);
assert.ok(migrated.agentVersion);
assert.notEqual(migrated.providerId,oldSession.providerId);
assert.equal(migrated.previousProviderId,oldSession.providerId);
assert.ok(migrated.messages.some(m=>m.id===oldSession.messages[0].id));
console.log(JSON.stringify({passed:true,turns:evidence.length,liveAccountMutations:0,oldChatHistoryPreserved:true}));
