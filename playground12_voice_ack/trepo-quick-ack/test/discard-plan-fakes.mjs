// Characterization suites exercise SQL fan-out; provider/journal are separate
// boundaries. The dedicated discard-identity suite tests frozen plans + retries.
import {mock} from 'node:test';
import {randomUUID} from 'node:crypto';
import identity from '../../../shared/kitchen-discard/identity.cjs';
export async function installDiscardPlanFixture(resolveRows) {
  const path=new URL('../lib/kitchen-discard-plan.mjs',import.meta.url).href;
  const real=await import(path);
  mock.module(path,{namedExports:{...real,prepareDiscardPlan:async(context,reference,options,snapshot)=>{
    const {scope,rows}=await snapshot();
    const selected=resolveRows ? resolveRows(rows,reference) : rows.filter(r=>reference.item_id
      ? r._id===reference.item_id : r.product_name.toLowerCase().includes(String(reference.item_name).toLowerCase()));
    if (!selected.length) throw Object.assign(new Error('Kitchen item not found.'),{statusCode:404,
      details:{candidates:rows.map(r=>({item_id:r._id,item_name:r.product_name}))}});
    const row=selected[0];
    return {version:identity.VERSION,household:scope.ownerId,members:scope.householdMemberIds,
      actor:scope.userId,selected_id:row._id,expected:row,fingerprint:identity.fingerprint(row),operation_id:randomUUID()};
  }}});
}
