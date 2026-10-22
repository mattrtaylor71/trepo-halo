import {test} from 'node:test';
import assert from 'node:assert/strict';
import {buildAppOutput} from '../lib/app-response.mjs';
const saved=(id,name,toolName='log_dish_from_voice')=>({toolName,result:{ok:true,statusCode:200,actionSummary:`Logged ${name}.`,toolResult:{dish:{id,dish_name:name,calories:100,protein:3}}}});
const render=toolEvents=>buildAppOutput({text:'Logged your dishes.',toolEvents,responseSurface:'app'});
test('each saved dish has its own card with its durable ID',()=>{
 const cards=render([saved('banana','Banana'),saved('coffee','Coffee','log_dish_ingredients')]).cards;
 assert.deepEqual(cards.map(c=>c.entity.id),['banana','coffee']);
 assert.deepEqual(cards.map(c=>c.body),['Logged Banana.','Logged Coffee.']);
});
test('a subsequent read does not hide committed dish cards',()=>{
 const cards=render([saved('banana','Banana'),saved('coffee','Coffee'),{toolName:'get_recent_dishes',result:{ok:true,toolResult:{dishes:[]}}}]).cards;
 assert.deepEqual(cards.map(c=>c.entity.id),['banana','coffee']);
});
test('failed or ID-less writes never become saved-dish cards',()=>{
 const cards=render([saved('banana','Banana'),saved('coffee','Coffee'),{...saved('missing','Failed'),result:{ok:false,statusCode:500}},saved(null,'Unverified')]).cards;
 assert.deepEqual(cards.map(c=>c.entity.id),['banana','coffee']);
});
test('duplicate receipts for one dish create only one card',()=>{
 const cards=render([saved('banana','Banana'),saved('banana','Banana'),saved('coffee','Coffee')]).cards;
 assert.deepEqual(cards.map(c=>c.entity.id),['banana','coffee']);
});
