// Verify flags-OFF behavior == current production (no regression on deploy).
const mysql = require('mysql2/promise');
const crypto = require('crypto');
const DB = { host:'database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com', user:'admin', password:'Nbmqyq17!', database:'mysqlTutorial', connectTimeout:15000 };
Object.assign(process.env, { DB_HOST:DB.host, DB_USER:DB.user, DB_PASSWORD:DB.password, DB_NAME:DB.database });
const A='00000000-leg-4aaa-8000-'+crypto.randomBytes(6).toString('hex');
const B='00000000-leg-4bbb-8000-'+crypto.randomBytes(6).toString('hex');
const HH='LEGHH'+crypto.randomBytes(3).toString('hex');
const TPL='7d7df434-d942-4037-b054-2d3005ea6abc_new_list';
let pass=0,fail=0; const check=(n,c)=>{c?(pass++,console.log('  ✓',n)):(fail++,console.log('  ✗ FAIL:',n));};
function load(flags){ for(const k of ['LIST_UNION_READ','LIST_UNION_READ_OWNERS','LIST_STRICT_UUID_MATCH'])delete process.env[k]; Object.assign(process.env,flags||{}); delete require.cache[require.resolve('../listHandler.js')]; delete require.cache[require.resolve('../householdSync.js')]; return require('../listHandler.js'); }
(async()=>{
  const conn=await mysql.createConnection(DB); const tA=`${A}_new_list`,tB=`${B}_new_list`;
  const rows=async t=>(await conn.execute(`SELECT product_name,action,household_item_uuid FROM \`${t}\``))[0];
  try{
    await conn.execute('INSERT INTO new_users (user_id,owner_id,first_name,created_at) VALUES (?,?,?,NOW())',[A,HH,'LegA']);
    await conn.execute('INSERT INTO new_users (user_id,owner_id,first_name,created_at) VALUES (?,?,?,NOW())',[B,HH,'LegB']);
    await conn.execute(`CREATE TABLE \`${tA}\` LIKE \`${TPL}\``); await conn.execute(`CREATE TABLE \`${tB}\` LIKE \`${TPL}\``);
    const h=load({}); // ALL FLAGS OFF = current prod
    console.log('[legacy] all flags OFF');
    // add still fans out (this is existing prod behavior)
    await h.handler({operation:'add',ownerId:A,device:'test',product_name:'milk'});
    check('add fans out to both (unchanged)', (await rows(tA)).some(r=>r.product_name==='milk') && (await rows(tB)).some(r=>r.product_name==='milk'));
    // view reads ONLY own table (no union)
    const uuid=(await rows(tA)).find(r=>r.product_name==='milk').household_item_uuid;
    await conn.execute(`INSERT INTO \`${tB}\` (_owner,_device,product_name,action,household_item_uuid) VALUES (?,?,?,?,?)`,[B,'t','eggsOnlyB','ADDED',crypto.randomUUID()]);
    let body=JSON.parse((await h.handler({operation:'view',ownerId:A})).body);
    check('view reads own table only (no union, 1 item)', body.count===1 && body.items[0].product_name==='milk');
    check('no union flag in legacy response', body.union===undefined);
    // legacy _id fallback STILL works (remove by _id on own table)
    const aId=(await conn.execute(`SELECT _id FROM \`${tA}\` WHERE product_name='milk'`))[0][0]._id;
    let r=JSON.parse((await h.handler({operation:'remove',ownerId:A,id:aId})).body);
    check('legacy: remove by _id works (fallback intact)', r.affectedRows>=1);
    check('A milk removed via _id', !(await rows(tA)).some(r=>r.product_name==='milk'));
    // set_action by uuid still propagates both (existing behavior)
    await h.handler({operation:'add',ownerId:A,device:'test',product_name:'cheese'});
    const cu=(await rows(tA)).find(r=>r.product_name==='cheese').household_item_uuid;
    await h.handler({operation:'set_action',ownerId:A,itemUUID:cu,checked:true});
    check('set_action by uuid propagates both (unchanged)', (await rows(tA)).find(r=>r.product_name==='cheese').action==='CHECKED' && (await rows(tB)).find(r=>r.product_name==='cheese').action==='CHECKED');
  }catch(e){console.error('ERR',e);fail++;}
  finally{ await conn.execute(`DROP TABLE IF EXISTS \`${tA}\``); await conn.execute(`DROP TABLE IF EXISTS \`${tB}\``); await conn.execute('DELETE FROM new_users WHERE owner_id=?',[HH]); console.log(`\n=== LEGACY RESULT: ${pass} passed, ${fail} failed ===`); await conn.end(); process.exit(fail?1:0);}
})();
