// Internal prospective provenance. A digest detects mismatch; it is not auth.
// Only build from an already normalized, server-captured dispatch payload.
const {createHash}=require('node:crypto');
function makeSceneSourceManifest(payload,{bucket}) {
  if(!payload||typeof payload.owner!=='string'||!/^[A-Za-z0-9_-]{1,64}$/.test(payload.owner)||
     typeof payload.job_id!=='string'||!/^[A-Za-z0-9-]{1,80}$/.test(payload.job_id)||
     !['bulk_inventory_deep','receipt_inventory_deep'].includes(payload.analysis_mode))return null;
  const raw=payload.receipt_s3_keys||[{bucket:payload.s3_bucket,key:payload.s3_key,sha256:payload.s3_sha256}];
  if(payload.receipt_s3_keys&&payload.analysis_mode!=='receipt_inventory_deep')return null;
  if(!Array.isArray(raw)||raw.length<1||raw.length>30)return null;
  const sources=[];
  for(let i=0;i<raw.length;i++) {
    const r=raw[i],suffix=payload.receipt_s3_keys?'-img'+i:'';
    if(!r||r.bucket!==bucket||typeof bucket!=='string'||!/^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/.test(bucket)||
       typeof r.sha256!=='string'||!/^[a-f0-9]{64}$/.test(r.sha256)||typeof r.key!=='string'||
       !['jpg','png','webp','gif'].some(ext=>r.key==='bulk-identify-uploads/'+payload.job_id+suffix+'.'+r.sha256+'.'+ext))return null;
    sources.push({bucket:r.bucket,key:r.key,sha256:r.sha256});
  }
  const manifest={version:1,owner:payload.owner,job_id:payload.job_id,analysis_mode:payload.analysis_mode,
    layout:payload.receipt_s3_keys?'receipt_pages':'single',sources};
  return {...manifest,digest:createHash('sha256').update(JSON.stringify(manifest)).digest('hex')};
}

function verifiedSceneSourceManifest(job,{owner,bucket}) {
  if(!job || job.meta?.owner!==owner)return null;
  const m=job.scene_source_manifest;
  if(!m||m.version!==1||m.owner!==owner||m.job_id!==job.job_id||m.analysis_mode!==job.analysis_mode||!Array.isArray(m.sources))return null;
  const payload={owner,job_id:job.job_id,analysis_mode:job.analysis_mode};
  if(m.layout==='receipt_pages'&&job.analysis_mode==='receipt_inventory_deep')payload.receipt_s3_keys=m.sources;
  else if(m.layout==='single'&&m.sources.length===1)Object.assign(payload,{s3_bucket:m.sources[0]?.bucket,s3_key:m.sources[0]?.key,s3_sha256:m.sources[0]?.sha256});
  else return null;
  const verified=makeSceneSourceManifest(payload,{bucket});
  return verified?.digest===m.digest ? verified : null;
}

function sceneSourceReference(job,{owner,bucket}) {
  const unset=value=>value===undefined||value===null||value==='';
  if(!job||job.meta?.owner!==owner||job.status!=='completed'||!unset(job.review_dismissed_at)||
     (job.deleted!==undefined&&job.deleted!==null&&job.deleted!==false)||!unset(job.deleted_at))return null;
  const verified=verifiedSceneSourceManifest(job,{owner,bucket});
  // Multi-page attribution is not inferred from a merged receipt item list.
  if(!verified||verified.sources.length!==1)return null;
  return {...verified.sources[0]};
}
module.exports={makeSceneSourceManifest,sceneSourceReference,verifiedSceneSourceManifest};
