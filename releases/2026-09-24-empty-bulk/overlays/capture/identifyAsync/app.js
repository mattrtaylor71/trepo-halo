const {EMPTY_BULK_MESSAGE, isEmptyBulkResult, emptyBulkResult} = require('../emptyBulkOutcome');
const {captureOutcome,aggregateCaptureOutcomes,applyRecoveryAvailability} = require('../captureOutcome');
// Lambda handler for async identification - creates job and processes in background
const { randomUUID } = require('crypto');
const { LambdaClient, InvokeCommand } = require('@aws-sdk/client-lambda');
const { S3Client, PutObjectCommand, GetObjectCommand } = require('@aws-sdk/client-s3');
const jobQueue = require('../dist/utils/jobQueue');
const { analyzeProduct } = require('../dist/services/analyzeProduct');
const { identifyItemBulkDeep, identifyItemDeep, identifyItemReceiptDeep } = require('../quickIdentify/service');
const { persistBulkKitchenResults } = require('../bulkKitchenWriter');
const { annotateBulkItemsWithKitchenSimilarity } = require('../bulkKitchenSimilarity');

const lambdaClient = new LambdaClient({});
const s3Client = new S3Client({});
const DEEP_MULTI_ITEM_MODE = 'quick_identify_deep';
const BULK_DEEP_MODE = 'bulk_inventory_deep';
const RECEIPT_DEEP_MODE = 'receipt_inventory_deep';
const MAX_ASYNC_INVOKE_PAYLOAD_BYTES = 950000;
const ASYNC_UPLOAD_PREFIX = 'bulk-identify-uploads/';

function isInventoryAnalysisMode(mode) {
  return mode === BULK_DEEP_MODE || mode === RECEIPT_DEEP_MODE;
}

// --- AI op telemetry (additive logging only; never throws) ---
function logAiOp(rec) {
  try {
    const out = { evt: 'ai_op', ...rec };
    if (typeof out.input === 'string' && out.input.length > 400) out.input = out.input.slice(0, 400);
    if (typeof out.output === 'string' && out.output.length > 1200) out.output = out.output.slice(0, 1200);
    if (typeof out.error === 'string' && out.error.length > 500) out.error = out.error.slice(0, 500);
    console.log(JSON.stringify(out));
  } catch (_) { /* never let telemetry throw */ }
}
// Summarize an identify result as item count + first few item names (no image bytes).
function summarizeItems(result) {
  try {
    const items = Array.isArray(result?.items) ? result.items : [];
    const names = items
      .map((it) => it && (it.item_name || it.check_in_label || it.product_name))
      .filter(Boolean)
      .slice(0, 5);
    return JSON.stringify({ item_count: items.length, first_items: names });
  } catch (_) {
    return null;
  }
}

function createHttpError(statusCode, message, details) {
  const error = new Error(message);
  error.statusCode = statusCode;
  error.exposeDetails = details || null;
  return error;
}

function detectImageContentType(buffer) {
  if (!buffer || buffer.length < 4) {
    return 'image/jpeg';
  }
  if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4e && buffer[3] === 0x47) {
    return 'image/png';
  }
  if (buffer[0] === 0xff && buffer[1] === 0xd8) {
    return 'image/jpeg';
  }
  if (buffer[0] === 0x47 && buffer[1] === 0x49 && buffer[2] === 0x46) {
    return 'image/gif';
  }
  if (
    buffer.length >= 12 &&
    buffer[0] === 0x52 &&
    buffer[1] === 0x49 &&
    buffer[2] === 0x46 &&
    buffer[3] === 0x46 &&
    buffer[8] === 0x57 &&
    buffer[9] === 0x45 &&
    buffer[10] === 0x42 &&
    buffer[11] === 0x50
  ) {
    return 'image/webp';
  }
  return 'image/jpeg';
}

function contentTypeToExtension(contentType) {
  switch (contentType) {
    case 'image/png':
      return 'png';
    case 'image/gif':
      return 'gif';
    case 'image/webp':
      return 'webp';
    default:
      return 'jpg';
  }
}

async function uploadOversizedImageToS3(jobId, imageBase64) {
  const bucketName = process.env.UPLOADS_BUCKET_NAME;
  if (!bucketName) {
    throw createHttpError(
      413,
      'Bulk image is too large for async processing.',
      'Compressed image is still too large and no S3 fallback bucket is configured. Reduce image size further or use image_url.'
    );
  }

  const imageBuffer = Buffer.from(imageBase64, 'base64');
  const contentType = detectImageContentType(imageBuffer);
  const extension = contentTypeToExtension(contentType);
  const key = `${ASYNC_UPLOAD_PREFIX}${jobId}.${extension}`;

  await s3Client.send(new PutObjectCommand({
    Bucket: bucketName,
    Key: key,
    Body: imageBuffer,
    ContentType: contentType,
  }));

  return { bucketName, key };
}

async function loadImageBufferFromS3(bucketName, key) {
  const response = await s3Client.send(new GetObjectCommand({
    Bucket: bucketName,
    Key: key,
  }));
  if (!response.Body) {
    throw new Error(`S3 object ${bucketName}/${key} had no response body`);
  }
  const bytes = await response.Body.transformToByteArray();
  return Buffer.from(bytes);
}

function parseOptionalBoolean(value) {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (value === 1) return true;
    if (value === 0) return false;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['true', 'yes', 'y', '1'].includes(normalized)) return true;
    if (['false', 'no', 'n', '0'].includes(normalized)) return false;
  }
  return null;
}

exports.handler = async (event) => {
  console.log('[IdentifyAsync] Event received');

  try {
    if (event?.deep_async_internal) {
      await processJob(
        event.job_id,
        event.image_url,
        event.image,
        event.analysis_mode,
        event.owner,
        event.user_id,
        event.device_id,
        event.s3_bucket,
        event.s3_key,
        event.session_id,
        event.receipt_s3_keys || null,
        Number(event.retry_count || 0),
        event.processing_handoff_token || undefined
      );
      return {
        statusCode: 202,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({ status: 'processing' }),
      };
    }

    const body = JSON.parse(event.body || '{}');
    if (body.operation === 'recover_capture') return require('../captureRecoveryRuntime').handle(event);
    const { image_url, image, owner, user_id, device_id, session_id } = body;
    // Support multi-image receipt uploads: `images` is an array of base64 strings
    const images = Array.isArray(body.images) ? body.images : null;
    const persistToKitchen = parseOptionalBoolean(body.persist_to_kitchen) === true;
    const analysisMode =
      body.analysis_mode === BULK_DEEP_MODE
        ? BULK_DEEP_MODE
        : body.analysis_mode === RECEIPT_DEEP_MODE
          ? RECEIPT_DEEP_MODE
        : body.analysis_mode === DEEP_MULTI_ITEM_MODE
          ? DEEP_MULTI_ITEM_MODE
          : 'product_analysis';

    if (persistToKitchen && !owner) {
      return {
        statusCode: 400,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({
          error: 'Missing owner. persist_to_kitchen requires owner.',
        }),
      };
    }

    if (!image_url && !image && !images) {
      return {
        statusCode: 400,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify({
          error: 'Missing image_url, image, or images. Provide JSON body with one of these fields.',
        }),
      };
    }

    // Create job
    const jobId = randomUUID();
    const imageCount = images ? images.length : 1;
    await jobQueue.createJob(jobId, {
      image_url: image_url || null,
      has_image: !!image || !!images,
      analysis_mode: analysisMode,
      meta: {
        owner: owner || null,
        user_id: user_id || null,
        device_id: device_id || null,
        persist_to_kitchen: Boolean(owner && isInventoryAnalysisMode(analysisMode) && persistToKitchen),
        session_id: session_id || null,
      },
      receipt_image_count: imageCount,
      stage: 'pending',
      stage_message: imageCount > 1 ? `Hang tight, processing ${imageCount} receipt photos...` : 'Hang tight, getting started...',
      progress: 2,
    });

    console.log(`[IdentifyAsync] Created job: ${jobId} (${imageCount} image${imageCount > 1 ? 's' : ''})`);

    // Register job with session if session_id provided
    if (session_id) {
      try {
        const sessionStore = require('../sessionStore');
        await sessionStore.ensureSession(session_id, owner || user_id || 'anonymous', jobId, analysisMode);
        console.log(`[IdentifyAsync] Registered job ${jobId} with session ${session_id}`);
      } catch (sessionErr) {
        console.warn(`[IdentifyAsync] Failed to register job with session: ${sessionErr.message}`);
      }
    }

    // For multi-image receipts, upload all images to S3 first, then dispatch ONE job
    let receiptS3Keys = null;
    if (process.env.CAPTURE_RECOVERY_ENABLED === 'true' && owner && !persistToKitchen &&
        analysisMode === RECEIPT_DEEP_MODE && (image || images?.length)) {
      try {
        receiptS3Keys = await require('../captureSourceRetention').retain({job_id:jobId,owner,images:images?.length ? images : [image]});
      } catch (error) {
        await jobQueue.updateJobStatus(jobId,'failed',null,'Photo could not be retained. Please try again.').catch(()=>null);
        throw error;
      }
    } else if (images && images.length > 0 && analysisMode === RECEIPT_DEEP_MODE) {
      console.log(`[IdentifyAsync] Uploading ${images.length} receipt images to S3 for job ${jobId}`);
      receiptS3Keys = [];
      for (let i = 0; i < images.length; i++) {
        const uploaded = await uploadOversizedImageToS3(`${jobId}-img${i}`, images[i]);
        receiptS3Keys.push({ bucket: uploaded.bucketName, key: uploaded.key });
        console.log(`[IdentifyAsync] Uploaded receipt image ${i + 1}/${images.length} to S3`);
      }
    }

    try {
      if (receiptS3Keys) {
        // Multi-image receipt: dispatch with S3 keys, no inline image
        await dispatchJob(jobId, null, null, analysisMode, owner, user_id, device_id, session_id, receiptS3Keys);
      } else {
        await dispatchJob(jobId, image_url, image, analysisMode, owner, user_id, device_id, session_id, null);
      }
    } catch (error) {
      await jobQueue.updateJobStatus(jobId, 'failed', null, error instanceof Error ? error.message : 'Job dispatch failed').catch(() => null);
      throw error;
    }

    // Return immediately with job ID
    return {
      statusCode: 202, // Accepted
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
      body: JSON.stringify({
        job_id: jobId,
        status: 'pending',
        analysis_mode: analysisMode,
        receipt_image_count: imageCount,
        persist_to_kitchen: Boolean(owner && isInventoryAnalysisMode(analysisMode) && persistToKitchen),
        session_id: session_id || null,
        message: 'Job created. Poll /job/{job_id} for results.',
      }),
    };
  } catch (error) {
    if (event?.deep_async_internal && error?.retryableProcessing) throw error;
    console.error('[IdentifyAsync] Error:', error);
    const statusCode = Number.isInteger(error?.statusCode) ? error.statusCode : 500;
    return {
      statusCode,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
      body: JSON.stringify({
        error: statusCode >= 500 ? 'Internal server error' : (error instanceof Error ? error.message : 'Request failed'),
        details: error?.exposeDetails || (error instanceof Error ? error.message : 'Unknown error'),
      }),
    };
  }
};

async function dispatchJob(jobId, imageUrl, imageBase64, analysisMode, owner, userId, deviceId, sessionId, receiptS3Keys) {
  if (process.env.AWS_LAMBDA_FUNCTION_NAME) {
    const initialPayload = {
      deep_async_internal: true,
      job_id: jobId,
      image_url: imageUrl || null,
      image: imageBase64 || null,
      analysis_mode: analysisMode,
      owner: owner || null,
      user_id: userId || null,
      device_id: deviceId || null,
      session_id: sessionId || null,
      receipt_s3_keys: receiptS3Keys || null,
    };
    let payload = JSON.stringify(initialPayload);
    let payloadSizeBytes = Buffer.byteLength(payload);
    if (payloadSizeBytes > MAX_ASYNC_INVOKE_PAYLOAD_BYTES) {
      if (!imageBase64) {
        throw createHttpError(
          413,
          'Bulk image is too large for async processing.',
          `Compressed image payload is ${payloadSizeBytes} bytes. Reduce image size further or use image_url so the backend can process it asynchronously.`
        );
      }

      const uploaded = await uploadOversizedImageToS3(jobId, imageBase64);
      payload = JSON.stringify({
        deep_async_internal: true,
        job_id: jobId,
        image_url: imageUrl || null,
        image: null,
        analysis_mode: analysisMode,
        owner: owner || null,
        user_id: userId || null,
        device_id: deviceId || null,
        session_id: sessionId || null,
        s3_bucket: uploaded.bucketName,
        s3_key: uploaded.key,
        receipt_s3_keys: receiptS3Keys || null,
      });
      payloadSizeBytes = Buffer.byteLength(payload);
      if (payloadSizeBytes > MAX_ASYNC_INVOKE_PAYLOAD_BYTES) {
        throw createHttpError(
          413,
          'Bulk image is too large for async processing.',
          `Compressed image payload is ${payloadSizeBytes} bytes even after S3 fallback packaging. Reduce image size further or use image_url.`
        );
      }
    }

    await lambdaClient.send(
      new InvokeCommand({
        FunctionName: process.env.AWS_LAMBDA_FUNCTION_NAME,
        InvocationType: 'Event',
        Payload: Buffer.from(payload),
      })
    );
    return;
  }

  processJob(jobId, imageUrl, imageBase64, analysisMode, owner, userId, deviceId, null, null, sessionId, receiptS3Keys).catch((error) => {
    console.error(`[IdentifyAsync] Background processing error for job ${jobId}:`, error);
    jobQueue.updateJobStatus(jobId, 'failed', null, error.message).catch(console.error);
  });
}

async function processJob(...args) {
  if (process.env.PROCESSING_LEASES_ENABLED !== 'true') return processJobWork(...args.slice(0, 12));
  const workArgs = Array.from({ length: 12 }, (_, index) => args[index]);
  return require('../processingLeaseRuntime').run(args[0], args[4],
    context => processJobWork(...workArgs, context), { handoffToken: args[12] });
}

async function processJobWork(jobId, imageUrl, imageBase64, analysisMode, owner, userId, deviceId, s3Bucket, s3Key, sessionId, receiptS3Keys, retryCount = 0, processingContext) {
  const queue = processingContext?.queue || jobQueue;
  try {
    const started = await queue.updateJobStatus(jobId, 'processing');
    if (started === false) {
      console.log('[IdentifyAsync] Skipped terminal or removed job replay');
      return;
    }
    await queue.updateJobProgress(jobId, 'queued', 'Starting up...', 8);
    console.log(`[IdentifyAsync] Processing job: ${jobId}`);

    // --- Multi-image receipt path ---
    if (Array.isArray(receiptS3Keys) && receiptS3Keys.length > 0 && analysisMode === RECEIPT_DEEP_MODE) {
      console.log(`[IdentifyAsync] Multi-image receipt: ${receiptS3Keys.length} images for job ${jobId}`);
      var allItems = [];
      var allReceiptSummaries = [];
      const receiptResults = [];

      for (let i = 0; i < receiptS3Keys.length; i++) {
        const imgNum = i + 1;
        const total = receiptS3Keys.length;
        const progressBase = Math.round(8 + (i / total) * 80);
        await queue.updateJobProgress(jobId, 'analyzing_receipt', `Analyzing receipt photo ${imgNum} of ${total}...`, progressBase);

        const imgBuffer = await loadImageBufferFromS3(receiptS3Keys[i].bucket, receiptS3Keys[i].key);
        console.log(`[IdentifyAsync] Loaded receipt image ${imgNum}/${total} from S3 (${imgBuffer.length} bytes)`);

        const receiptStart = Date.now();
        let partialResult;
        try {
          await processingContext?.assertActive();
          partialResult = await identifyItemReceiptDeep(
            { imageBuffer: imgBuffer },
            {
              onStage: (stageUpdate) =>
                queue.updateJobProgress(
                  jobId,
                  stageUpdate.stage,
                  `Photo ${imgNum}/${total}: ${stageUpdate.message || ''}`,
                  Math.min(progressBase + Math.round((stageUpdate.progress || 0) / total), 95),
                  stageUpdate
                ),
            }
          );
        } catch (receiptErr) {
          logAiOp({
            service: 'bulk', op: 'identify_receipt',
            model: null, latency_ms: Date.now() - receiptStart, status: 'error',
            owner_id: owner, job_id: jobId,
            input: `${receiptS3Keys[i].key} (photo ${imgNum}/${total})`,
            error: (receiptErr && (receiptErr.message || String(receiptErr))) || 'identify_receipt_failed',
          });
          throw receiptErr;
        }
        logAiOp({
          service: 'bulk', op: 'identify_receipt',
          model: (partialResult && partialResult.debug && partialResult.debug.model) || null,
          latency_ms: Date.now() - receiptStart, status: 'success',
          owner_id: owner, job_id: jobId,
          input: `${receiptS3Keys[i].key} (photo ${imgNum}/${total})`,
          output: summarizeItems(partialResult),
        });

        receiptResults.push(partialResult);
        if (Array.isArray(partialResult?.items)) {
          allItems = allItems.concat(partialResult.items);
          console.log(`[IdentifyAsync] Receipt image ${imgNum}: found ${partialResult.items.length} items (total so far: ${allItems.length})`);
        }
        if (partialResult?.receipt_summary) {
          allReceiptSummaries.push(partialResult.receipt_summary);
        }
      }

      // Deduplicate items across receipt images (border items appear in adjacent photos)
      const beforeDedup = allItems.length;
      const seen = new Map();
      const dedupedItems = [];
      for (const item of allItems) {
        // Normalize: lowercase, trim, collapse whitespace
        const name = (item.item_name || item.check_in_label || item.product_name || '').toLowerCase().trim().replace(/\s+/g, ' ');
        if (!name) { dedupedItems.push(item); continue; }
        if (!seen.has(name)) {
          seen.set(name, true);
          dedupedItems.push(item);
        }
      }
      allItems = dedupedItems;
      if (beforeDedup !== allItems.length) {
        console.log(`[IdentifyAsync] Deduped receipt items: ${beforeDedup} → ${allItems.length} (removed ${beforeDedup - allItems.length} duplicates)`);
      }

      // Merge receipt summaries
      var mergedSummary = null;
      if (allReceiptSummaries.length === 1) {
        mergedSummary = allReceiptSummaries[0];
      } else if (allReceiptSummaries.length > 1) {
        const merchants = allReceiptSummaries.map(s => s.merchant_name).filter(Boolean);
        const totals = allReceiptSummaries.map(s => s.total_amount).filter(v => v != null);
        const dates = allReceiptSummaries.map(s => s.purchase_date).filter(Boolean).sort();
        const counts = allReceiptSummaries.map(s => s.receipt_item_count_hint).filter(v => v != null);
        mergedSummary = {
          merchant_name: merchants.length > 0 ? [...new Set(merchants)].join(' + ') : null,
          purchase_date: dates[0] || null,
          total_amount: totals.length > 0 ? totals.reduce((a, b) => a + b, 0) : null,
          receipt_item_count_hint: counts.length > 0 ? counts.reduce((a, b) => a + b, 0) : null,
        };
      }

      // Build combined analysis result
      var analysis = {
        items: allItems,
        receipt_summary: mergedSummary,
        receipt_image_count: receiptS3Keys.length,
        ...aggregateCaptureOutcomes(receiptResults),
      };
    } else {
      // --- Single-image path (original) ---
      let imageBuffer = null;
      if (imageBase64) {
        imageBuffer = Buffer.from(imageBase64, 'base64');
      } else if (s3Bucket && s3Key) {
        await queue.updateJobProgress(jobId, 'loading_image', 'Loading your photo...', 12, {
          s3_key: s3Key,
        });
        imageBuffer = await loadImageBufferFromS3(s3Bucket, s3Key);
      }

      if (!imageBuffer && !imageUrl) {
        throw new Error('No image provided');
      }

      // Persist source image to S3 for kitchen item display
      let persistedImageUrl = imageUrl || null;
      if (imageBuffer && !persistedImageUrl) {
        const bucketName = process.env.UPLOADS_BUCKET_NAME;
        if (bucketName) {
          try {
            const contentType = detectImageContentType(imageBuffer);
            const extension = contentTypeToExtension(contentType);
            const sourceKey = `source-images/${jobId}.${extension}`;
            await s3Client.send(new PutObjectCommand({
              Bucket: bucketName,
              Key: sourceKey,
              Body: imageBuffer,
              ContentType: contentType,
            }));
            persistedImageUrl = `https://${bucketName}.s3.amazonaws.com/${sourceKey}`;
            // Update DynamoDB job record with source image URL
            if (processingContext) {
              await processingContext.updateImageUrl(persistedImageUrl);
            } else {
              const { UpdateCommand } = require('@aws-sdk/lib-dynamodb');
              const { DynamoDBDocumentClient } = require('@aws-sdk/lib-dynamodb');
              const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
              const docClient = DynamoDBDocumentClient.from(new DynamoDBClient({}));
              await docClient.send(new UpdateCommand({
                TableName: process.env.JOB_TABLE_NAME,
                Key: { job_id: jobId },
                UpdateExpression: 'SET image_url = :url',
                ExpressionAttributeValues: { ':url': persistedImageUrl },
              }));
            }
            console.log(`[IdentifyAsync] Source image persisted to S3: ${sourceKey}`);
          } catch (uploadErr) {
            console.warn(`[IdentifyAsync] Failed to persist source image to S3: ${uploadErr.message}`);
          }
        }
      }

      const inventoryAnalyzer =
        analysisMode === BULK_DEEP_MODE
          ? identifyItemBulkDeep
          : analysisMode === RECEIPT_DEEP_MODE
            ? identifyItemReceiptDeep
            : analysisMode === DEEP_MULTI_ITEM_MODE
              ? identifyItemDeep
              : null;

      const singleStart = Date.now();
      const singleOp = inventoryAnalyzer ? 'identify_items' : 'analyze_product';
      const singleInput = s3Key || imageUrl || 'image';
      var analysis;
      try {
        await processingContext?.assertActive();
        analysis =
          inventoryAnalyzer
            ? imageBuffer
              ? await inventoryAnalyzer(
                  { imageBuffer },
                  {
                    onStage: (stageUpdate) =>
                      queue.updateJobProgress(
                        jobId,
                        stageUpdate.stage,
                        stageUpdate.message,
                        stageUpdate.progress,
                        stageUpdate
                      ),
                  }
                )
              : await inventoryAnalyzer(
                  { imageUrl },
                  {
                    onStage: (stageUpdate) =>
                      queue.updateJobProgress(
                        jobId,
                        stageUpdate.stage,
                        stageUpdate.message,
                        stageUpdate.progress,
                        stageUpdate
                      ),
                  }
                )
            : imageBuffer
              ? await analyzeProduct({ imageBuffer }, { stockImageMode: 'deep' })
              : await analyzeProduct({ imageUrl }, { stockImageMode: 'deep' });
      } catch (singleErr) {
        logAiOp({
          service: 'bulk', op: singleOp, model: null,
          latency_ms: Date.now() - singleStart, status: 'error',
          owner_id: owner, job_id: jobId, input: singleInput,
          error: (singleErr && (singleErr.message || String(singleErr))) || 'identify_failed',
        });
        throw singleErr;
      }
      logAiOp({
        service: 'bulk', op: singleOp,
        model: (analysis && analysis.debug && analysis.debug.model) || null,
        latency_ms: Date.now() - singleStart, status: 'success',
        owner_id: owner, job_id: jobId, input: singleInput,
        output: summarizeItems(analysis),
      });
    }

    if (isInventoryAnalysisMode(analysisMode) && Array.isArray(analysis?.items)) {
      try {
        const similarity = await annotateBulkItemsWithKitchenSimilarity({
          owner,
          items: analysis.items,
        });
        analysis = {
          ...analysis,
          items: similarity.items,
          debug: {
            ...analysis.debug,
            kitchen_similarity: similarity.debug,
          },
        };
      } catch (error) {
        console.warn(`[IdentifyAsync] Kitchen similarity annotation failed for job ${jobId}:`, error);
        analysis = {
          ...analysis,
          debug: {
            ...analysis.debug,
            kitchen_similarity: {
              enabled: false,
              error: error instanceof Error ? error.message : 'Unknown kitchen similarity error',
            },
          },
        };
      }
    }

    // An accepted photo can validly contain no groceries. This is terminal, but
    // status-based installed clients must never offer it as an empty review.
    if (isEmptyBulkResult(analysisMode, analysis)) {
      await queue.updateJobProgress(jobId, 'failed', EMPTY_BULK_MESSAGE, 100, {item_count: 0});
      const stored = await queue.updateJobStatus(jobId, 'failed', emptyBulkResult(analysis), EMPTY_BULK_MESSAGE);
      if (stored === false) return; // A losing delivery must not change session state.
      if (sessionId) {
        try {
          const sessionStore = require('../sessionStore');
          const session = await sessionStore.markJobFailed(sessionId, jobId);
          // A mixed session still has valid photos to review. An all-empty session
          // must not send a ready notification or automatically retry the same photo.
          if (sessionStore.isSessionComplete(session) && (session.completed_job_ids || []).length > 0) {
            if (await sessionStore.tryMarkSessionNotified(sessionId)) await sendSessionCompletionNotification(session);
          }
        } catch (sessionErr) {
          console.warn(`[IdentifyAsync] Empty photo session tracking failed for job ${jobId}: ${sessionErr.message}`);
        }
      }
      return;
    }

    let persistence = null;
    if (isInventoryAnalysisMode(analysisMode) && owner && parseOptionalBoolean((await queue.getJob(jobId))?.meta?.persist_to_kitchen) === true) {
      await queue.updateJobProgress(jobId, 'persisting_kitchen_items', 'Saving to your kitchen...', 95, {
        owner,
      });
      await processingContext?.assertActive();
      persistence = await persistBulkKitchenResults({
        owner,
        userId: userId || owner,
        deviceId: deviceId || 'bulk-scan',
        jobId,
        analysis,
        imageUrl: imageUrl || null,
      });
    }

    if (analysisMode === RECEIPT_DEEP_MODE && !analysis.capture_outcomes) {
      analysis = {...analysis,...aggregateCaptureOutcomes([analysis])};
    }
    if (analysisMode === RECEIPT_DEEP_MODE) {
      analysis = applyRecoveryAvailability(analysis, await queue.getJob(jobId), {
        enabled: process.env.CAPTURE_RECOVERY_ENABLED === 'true' &&
          process.env.PROCESSING_LEASES_ENABLED === 'true' && Boolean(process.env.TOKEN_SIGNING_SECRET),
        owner, bucket: process.env.UPLOADS_BUCKET_NAME,
      });
    }
    // Store result
    await queue.updateJobProgress(jobId, 'completed', analysis?.items?.length ? 'Ready to review' : 'No items identified', 100, {
      item_count: analysis?.items?.length || 0,
      persisted_count: persistence?.persisted_count || 0,
    });
    const stored = await queue.updateJobStatus(jobId, 'completed', {
      ...analysis,
      persistence,
      debug: {
        ...analysis.debug,
        analysis_mode: analysisMode,
        processed_at: new Date().toISOString(),
        owner: owner || null,
        persisted_count: persistence?.persisted_count || 0,
        created_count: persistence?.created_count || 0,
        duplicate_count: persistence?.duplicate_count || 0,
        persistence_errors: persistence?.errors?.length || 0,
      },
    });

    if (stored === false) {
      console.log('[IdentifyAsync] Completion already settled; skipping duplicate effects');
      return;
    }

    console.log(`[IdentifyAsync] Job ${jobId} completed successfully`);

    // Single-job push notification (no session)
    if (!sessionId && owner) {
      const itemCount = analysis?.items?.length || 0;
      const route = analysisMode === RECEIPT_DEEP_MODE
        ? 'receipt_analysis_review'
        : 'bulk_session_review';
      await sendSingleJobNotification(owner, jobId, itemCount, route);
    }

    // Session completion check
    if (sessionId) {
      try {
        const sessionStore = require('../sessionStore');
        const session = await sessionStore.markJobCompleted(sessionId, jobId);
        if (sessionStore.isSessionComplete(session)) {
          const canNotify = await sessionStore.tryMarkSessionNotified(sessionId);
          if (canNotify) {
            await sendSessionCompletionNotification(session);
          }
        }
      } catch (sessionErr) {
        console.warn(`[IdentifyAsync] Session completion check failed for job ${jobId}: ${sessionErr.message}`);
      }
    }
  } catch (error) {
    if (processingContext?.lost) return;
    await processingContext?.assertActive();
    console.error(`[IdentifyAsync] Job ${jobId} failed (attempt ${retryCount + 1}):`, error);

    // We already have the user's uploaded image — rather than surfacing a failure,
    // re-drive the job (a fresh Gemini→OpenAI pass) up to a bounded number of times.
    // Only after retries are exhausted do we mark it failed.
    const maxRedrive = Number(process.env.IDENTIFY_MAX_REDRIVE || 2);
    const canRedrive =
      retryCount < maxRedrive &&
      !!process.env.AWS_LAMBDA_FUNCTION_NAME &&
      !!(imageUrl || s3Key || imageBase64 || (analysisMode === RECEIPT_DEEP_MODE && receiptS3Keys?.length));
    if (canRedrive) {
      try {
        await queue.updateJobProgress(jobId, 'retrying', 'Taking another pass at your photo…', 15).catch(() => null);
        const redrivePayload = {
          deep_async_internal: true,
          job_id: jobId,
          image_url: imageUrl || null,
          image: (imageUrl || s3Key) ? null : (imageBase64 || null), // prefer url/s3 to keep payload small
          analysis_mode: analysisMode,
          owner: owner || null,
          user_id: userId || null,
          device_id: deviceId || null,
          s3_bucket: s3Bucket || null,
          s3_key: s3Key || null,
          session_id: sessionId || null,
          receipt_s3_keys: receiptS3Keys || null,
          retry_count: retryCount + 1,
          processing_handoff_token: processingContext?.lease.token || null,
        };
        await lambdaClient.send(new InvokeCommand({
          FunctionName: process.env.AWS_LAMBDA_FUNCTION_NAME,
          InvocationType: 'Event',
          Payload: Buffer.from(JSON.stringify(redrivePayload)),
        }));
        console.log(`[IdentifyAsync] Re-drove job ${jobId} (retry ${retryCount + 1}/${maxRedrive}) instead of failing`);
        return; // the re-drive owns the outcome now — do NOT mark failed
      } catch (redriveErr) {
        console.error(`[IdentifyAsync] Re-drive dispatch failed for job ${jobId}: ${redriveErr.message}; marking failed`);
      }
    }

    const failed = await queue.updateJobStatus(jobId, 'failed', null, error.message);
    if (failed === false) {
      console.log('[IdentifyAsync] Failure already settled; skipping stale failure effects');
      return;
    }

    // TERMINAL failure (re-drives exhausted). Emit a marker containing "failed:" that the
    // CloudWatch metric filter (bulk-identify-failed alarm) keys on. The per-attempt logs
    // above deliberately say "failed (attempt N):" so recoverable re-drives never alarm —
    // only a genuine give-up reaches here and pages.
    console.error(`[IdentifyAsync] Job ${jobId} identify failed: gave up after ${retryCount + 1} attempt(s) (mode=${analysisMode}): ${(error && error.message) || String(error)}`);


    // Session failure tracking
    if (sessionId) {
      try {
        const sessionStore = require('../sessionStore');
        const session = await sessionStore.markJobFailed(sessionId, jobId);
        if (sessionStore.isSessionComplete(session)) {
          const canNotify = await sessionStore.tryMarkSessionNotified(sessionId);
          if (canNotify) {
            await sendSessionCompletionNotification(session);
          }
        }
      } catch (sessionErr) {
        console.warn(`[IdentifyAsync] Session failure tracking failed for job ${jobId}: ${sessionErr.message}`);
      }
    }
  }
}

async function sendSessionCompletionNotification(session) {
  const baseUrl = process.env.NOTIFICATIONS_API_BASE_URL;
  if (!baseUrl) {
    console.warn('[IdentifyAsync] NOTIFICATIONS_API_BASE_URL not set, skipping notification');
    return;
  }
  const completedCount = (session.completed_job_ids || []).length;
  const failedCount = (session.failed_job_ids || []).length;
  const total = session.total_jobs || 0;

  let body;
  if (failedCount === 0) {
    body = total === 1
      ? 'Your groceries have been analyzed and are ready to review.'
      : `All ${total} photos have been analyzed and are ready to review.`;
  } else {
    body = `${completedCount} of ${total} photos analyzed successfully. Tap to review.`;
  }

  try {
    const route = session.analysis_mode === RECEIPT_DEEP_MODE
      ? 'receipt_analysis_review'
      : 'bulk_session_review';
    const response = await fetch(`${baseUrl}/notifications/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        owner_id: session.owner_id,
        title: 'Check-in Ready',
        body,
        data: {
          route,
          session_id: session.session_id,
        }
      })
    });
    if (!response.ok) {
      console.warn(`[IdentifyAsync] Notification send failed: HTTP ${response.status}`);
    }
  } catch (err) {
    console.warn(`[IdentifyAsync] Notification send error: ${err.message}`);
  }
}

async function sendSingleJobNotification(ownerId, jobId, itemCount, route) {
  const baseUrl = process.env.NOTIFICATIONS_API_BASE_URL;
  if (!baseUrl) {
    console.warn('[IdentifyAsync] NOTIFICATIONS_API_BASE_URL not set, skipping notification');
    return;
  }

  const body = itemCount > 0
    ? `Found ${itemCount} item${itemCount === 1 ? '' : 's'}. Tap to review.`
    : 'Your check-in is ready to review.';

  try {
    const response = await fetch(`${baseUrl}/notifications/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        owner_id: ownerId,
        title: 'Check-in Ready',
        body,
        data: {
          route,
          job_id: jobId,
        }
      })
    });
    if (response.ok) {
      console.log(`[IdentifyAsync] Push notification sent for job ${jobId} (${itemCount} items)`);
    } else {
      console.warn(`[IdentifyAsync] Push notification failed: HTTP ${response.status}`);
    }
  } catch (err) {
    console.warn(`[IdentifyAsync] Push notification error: ${err.message}`);
  }
}

