import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const root=new URL('../',import.meta.url);
const code=fs.readFileSync(new URL('edge-guard.js',root),'utf8');
const context=vm.createContext({});vm.runInContext(code,context);
const path='/icons/trepo-app-bold-outline-v1/'+'a'.repeat(64)+'/'+'b'.repeat(64)+'.png';
test('immutable icon GET and HEAD preserve exact path',()=>{
 for(const method of ['GET','HEAD']){
   const request={uri:path,method,querystring:{}};
   assert.equal(context.handler({request}),request);
 }
});
test('private media, traversal, wrong hashes, query strings and writes never reach origin',()=>{
 for(const request of [
 {uri:'/images/customer-photo.jpg',method:'GET'}, {uri:path.replace('/icons/','/icons/../'),method:'GET'},
 {uri:path.replace('a'.repeat(64),'a'.repeat(63)),method:'GET'}, {uri:path,method:'POST'},
 {uri:path,method:'GET',querystring:{user:{value:'private'}}}, {uri:'/saved-recipes/person',method:'GET'}
 ]) assert.equal(context.handler({request}).statusCode,404);
});
test('deployed template uses tested guard and never forces private/no-store content into cache',()=>{
 const template=JSON.parse(fs.readFileSync(new URL('template.json',root)));
 assert.equal(template.Resources.Guard.Properties.FunctionCode,code);
 const cfg=template.Resources.Cache.Properties.CachePolicyConfig;
 assert.equal(cfg.MinTTL,0);assert.equal(cfg.DefaultTTL,0);
 assert.equal(cfg.ParametersInCacheKeyAndForwardedToOrigin.HeadersConfig.HeaderBehavior,'none');
});
