import crypto from 'node:crypto';
import {S3Client, PutObjectCommand} from '@aws-sdk/client-s3';
let client;

// Preserve original input before transcription, including silence and failed ASR.
// Async Halo ingestion already archives its input; enable this on direct endpoints.
export async function preserveCapturedAudio(audio, env, metadata = {}) {
  const bucket = String(env?.CAPTURE_AUDIO_ARCHIVE_BUCKET || '').trim();
  if (!bucket || !Buffer.isBuffer(audio) || audio.length === 0) return null;
  const key = `voice-inputs/${new Date().toISOString().slice(0,10)}/${crypto.randomUUID()}.bin`;
  const sha256 = crypto.createHash('sha256').update(audio).digest('hex');
  const allowed = ['owner_id','device_id','session_id','response_surface','sample_rate','audio_format','request_id'];
  const tags = {sha256, captured_at:new Date().toISOString(), source:String(env?.AWS_LAMBDA_FUNCTION_NAME || 'direct-voice')};
  for (const name of allowed) {
    if (metadata[name] != null && String(metadata[name])) tags[name] = encodeURIComponent(String(metadata[name])).slice(0,160);
  }
  client ||= new S3Client({region:env?.AWS_REGION || 'us-east-1',maxAttempts:2});
  try {
    await client.send(new PutObjectCommand({Bucket:bucket,Key:key,Body:audio,
      ContentType:'application/octet-stream',ServerSideEncryption:'AES256',Metadata:tags}),
      {abortSignal:AbortSignal.timeout(5000)});
  } catch {
    console.error(JSON.stringify({evt:'captured_audio',status:'archive_failed',source:tags.source}));
    const error = new Error('Your recording could not be saved. Please try again.');
    error.code='audio_archive_unavailable';error.statusCode=503;throw error;
  }
  console.log(JSON.stringify({evt:'captured_audio',status:'retained',key,sha256,bytes:audio.length}));
  return {bucket,key,sha256};
}
