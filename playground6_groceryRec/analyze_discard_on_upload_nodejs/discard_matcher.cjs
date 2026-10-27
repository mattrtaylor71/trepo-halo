'use strict';
// The release assembler copies shared/kitchen-discard/identity.cjs next to this
// file. Local source uses the same canonical module, never a second prompt.
const fs=require('node:fs'),path=require('node:path');
const identity=require(fs.existsSync(path.join(__dirname,'discard_identity.cjs'))
  ? './discard_identity.cjs' : '../../shared/kitchen-discard/identity.cjs');

function createPhotoMatcher({connect,members,match,load,retain}) {
  return async function matchAndRemoveFromKitchen(owner,groceryItem,options={}) {
    if (!options.jobId) return {removed:false,reason:'missing_discard_job'};
    let connection,transaction=false;
    try {
      let plan=await load(options.jobId,owner);
      if (!plan && options.remove!==false) return {removed:false,reason:'missing_frozen_match'};
      connection=await connect();
      const memberIds=(await members(connection,owner)).map(String).sort();
      if (!memberIds.length) return {removed:false,reason:'membership_unavailable'};
      const query=`SELECT k.*, COALESCE(e.revision,0) AS amount_revision,
        JSON_UNQUOTE(JSON_EXTRACT(e.overrides,'$.__acquired_on')) AS acquired_on
        FROM shared_kitchen k LEFT JOIN kitchen_item_edits e
        ON e.item_id=CONVERT(k._id USING utf8mb4) COLLATE utf8mb4_unicode_ci
        WHERE k.owner_id IN (${memberIds.map(()=>'?').join(',')}) AND k.action='IN'`;
      if (!plan) {
        const [rows]=await connection.execute(query,memberIds);
        const result=await match({kind:'photo',target:groceryItem,candidates:rows},options.timeoutMs);
        plan=await retain(options.jobId,owner,{...result,job_id:options.jobId,owner:String(owner),members:memberIds});
      }
      if (plan.version!==identity.VERSION || plan.job_id!==options.jobId || plan.owner!==String(owner)
          || JSON.stringify(plan.members)!==JSON.stringify(memberIds)) return {removed:false,reason:'match_scope_changed'};
      if (!plan.matched) return {removed:false,reason:plan.reason};
      if (!plan.expected || plan.expected._id!==plan.selected_id || identity.fingerprint(plan.expected)!==plan.fingerprint)
        return {removed:false,reason:'invalid_frozen_match'};
      const result={matched:true,removed:false,_id:plan.selected_id,reason:'identity_match',
        product_image_url:plan.expected.product_image_url || null,product_image_key:plan.expected.product_image_key || null};
      if (options.remove===false) return result;
      await connection.beginTransaction();transaction=true;
      // Hold current membership against transfer for the entire removal.
      const [lockedMembers]=await connection.execute(
        `SELECT user_id,owner_id FROM new_users WHERE user_id IN (${memberIds.map(()=>'?').join(',')}) ORDER BY user_id LOCK IN SHARE MODE`,memberIds);
      const currentIds=(await members(connection,owner)).map(String).sort();
      if (lockedMembers.length!==memberIds.length || JSON.stringify(currentIds)!==JSON.stringify(memberIds)) throw Error('membership_changed');
      const reason=`discard_match:${options.jobId}`;
      const [rows]=await connection.execute(query+' AND k._id=? FOR UPDATE',[...memberIds,plan.selected_id]);
      const [receipts]=await connection.execute('SELECT _id,owner_id,archived_reason FROM shared_archive_kitchen WHERE _id=? FOR UPDATE',[plan.selected_id]);
      if (receipts.some(r=>r.archived_reason===reason && r.owner_id===plan.expected.owner_id)) {
        await connection.commit();transaction=false;
        return {...result,removed:true,replayed:true};
      }
      if (rows.length!==1 || identity.fingerprint(rows[0])!==plan.fingerprint) throw Error('kitchen_changed');
      const row=rows[0];
      const [archived]=await connection.execute(`INSERT IGNORE INTO shared_archive_kitchen
        SELECT *,NOW(),?,'shared_kitchen' FROM shared_kitchen WHERE _id=? AND owner_id=? AND action='IN'`,[reason,row._id,row.owner_id]);
      if (Number(archived.affectedRows)!==1) throw Error('archive_not_confirmed');
      const [deleted]=await connection.execute('DELETE FROM shared_kitchen WHERE _id=? AND owner_id=? AND action=\'IN\'',[row._id,row.owner_id]);
      if (Number(deleted.affectedRows)!==1) throw Error('removal_not_confirmed');
      await connection.commit();transaction=false;
      return {...result,removed:true};
    } catch(error) {
      if (transaction) await connection.rollback();
      console.error('[discard_identity] removal_not_applied',error?.name || 'Error');
      return {removed:false,reason:'matching_or_commit_unavailable'};
    } finally {if(connection) await connection.end();}
  };
}
let productionMatcher;
async function matchAndRemoveFromKitchen(owner,item,options) {
  if (!productionMatcher) {
    const mysql=require('mysql2/promise');
    const AWS=require('aws-sdk');
    const dynamo=new AWS.DynamoDB.DocumentClient();
    const {getHouseholdMemberIds}=require('./dist/utils/householdSync');
    const table=process.env.JOBS_TABLE;
    const load=async(jobId,owner)=>{
      const {Item}=await dynamo.get({TableName:table,Key:{job_id:jobId},ConsistentRead:true}).promise();
      if (!Item || String(Item.owner)!==String(owner)) throw Error('job_scope_changed');
      return Item.discard_identity_plan || null;
    };
    productionMatcher=createPhotoMatcher({
      connect:()=>mysql.createConnection({host:process.env.DB_HOST,port:Number(process.env.DB_PORT || 3306),
        user:process.env.DB_USER,password:process.env.DB_PASS,database:process.env.DB_NAME,charset:'utf8mb4',timezone:'Z'}),
      members:getHouseholdMemberIds,load,
      retain:async(jobId,owner,plan)=>{
        try {
          await dynamo.update({TableName:table,Key:{job_id:jobId},
            UpdateExpression:'SET discard_identity_plan=:plan',
            ConditionExpression:'attribute_exists(job_id) AND #owner=:owner AND attribute_not_exists(discard_identity_plan)',
            ExpressionAttributeNames:{'#owner':'owner'},ExpressionAttributeValues:{':owner':owner,':plan':plan}}).promise();
          return plan;
        } catch(error) {
          if(error.code!=='ConditionalCheckFailedException') throw error;
          const winner=await load(jobId,owner);if(!winner) throw error;return winner;
        }
      },
      match:(input,timeoutMs)=>identity.matchIdentity(input,{env:process.env,timeoutMs,create:async(body,opts)=>{
        const OpenAI=require('openai');
        const client=new OpenAI(require('/var/task/ai_consent_guard.cjs').providerOptions({apiKey:process.env.OPENAI_API_KEY,maxRetries:0,timeout:45000}));
        return client.responses.create(body,opts);
      }})
    });
  }
  return productionMatcher(owner,item,options);
}
module.exports={matchAndRemoveFromKitchen,createPhotoMatcher,photoTarget:identity.photoTarget};
