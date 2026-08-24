# HALO production cutover — plan

Response to `HALOMAIN_rev1p5_modular/docs/BACKEND_HANDOFF.md` (d9d6e51).
Everything below was verified against the live account on 2026-08-24, not taken from the doc.

---

## Does this affect app users? YES, if we are careless. The plan is built around that.

**`POST /presign` is not a device endpoint. It is the iOS app's main photo-upload path.**
`UploadAPIService.swift:171` posts the *same body shape* the device does (`user_id`, `device_id`,
`owner`, `action`, `type`, `content_type`, `camera_meta`), and it is called from every capture
flow: check-in, dish, discard, and the dish result screen.

The two share more than the route:

| Shared thing | Used by |
|---|---|
| `POST /presign` → `PresignFunction` | iOS app **and** HALO |
| `POST /presign/discard` | same lambda again (legacy alias) |
| `trepo-grocery-uploads-dev` | iOS app **and** HALO |
| `trepo-grocery-discards-dev` | iOS app **and** HALO |
| `AnalyzeOnUpload`, `AnalyzeDishOnUpload`, `AnalyzeDiscardOnUpload` | both |

Sampled 5,672 objects in the upload bucket: **73% HALO, 27% app**, across 537 device_ids.

**So the governing rule for this work is: the app path must come out byte-identical.**
Every change below is additive and gated on an explicit allow-list. Default behaviour for anything
not on that list is exactly what happens today.

---

## What the handoff doc gets wrong (tell the firmware side)

**The ~700 byte response limit does not exist.** The doc says a response over ~700 bytes
"silently fails to parse and the capture is lost". Not true for shipping firmware:

- the library is **ArduinoJson 7.2.0**, where `StaticJsonDocument<768>` is a deprecated shim that
  inherits `JsonDocument`; `N` only feeds `capacity()` and does not constrain allocation
- production responses are **already 1,811–2,007 bytes** (measured live, all three modes)
- consistent with their own 294-consecutive-capture record

Designing the prod API around a 700-byte cap would be wasted work. Worth correcting, because the
rest of that doc is careful and someone will otherwise trust this.

**Latency is a non-issue.** Their budget is 30,000 ms. Our presign p99 is **250 ms** — 0.8% of it.
846 invocations in 7 days, 0 errors, 0 throttles.

---

## What the doc misses

**The `resized-images/` twin is not made by the presign path.** It comes from three EventBridge
rules whose event patterns **hardcode the bucket name**:

| Rule | Source bucket | Target |
|---|---|---|
| `…AnalyzeOnUploadS3CreateVi-T3trbVtHIQTK` | `trepo-grocery-uploads-dev` | `AnalyzeOnUpload` |
| `…AnalyzeDishOnUploadS3Crea-RMh57QosG9Vy` | `trepo-grocery-uploads-dev` | `AnalyzeDishOnUpload` |
| `…AnalyzeDiscardOnUploadS3C-uqUT5O71Q3PJ` | `trepo-grocery-discards-dev` | `AnalyzeDiscardOnUpload` |

New buckets need EventBridge enabled **and** these patterns extended. Miss it and photos upload
with a 200 and are never processed — precisely the failure their §8 verification catches.

---

## Sequencing: fix OTA FIRST

The doc treats the two blockers as parallel. They are not.

Until OTA works, **any mistake in the backend cutover is permanent in every field unit.** Their own
§7 says "assume the device cannot be updated". Doing the backend first means shipping with no
recovery path. So Phase 1 is OTA, and it is a decision rather than engineering.

---

## Phase 1 — unblock OTA (no code)

`halo-ota-prod` has all four public-access flags `true`; 6.2.0 is staged correctly
(`sense_6.2.0.bin` 1,650,912 bytes + `manifest_latest.json` 331 bytes).

**I scanned the binary before recommending this.** No AWS keys, no `sk-` tokens, no private key
material. The `BEGIN PRIVATE KEY` strings are mbedTLS PEM label constants packed back-to-back in
rodata (each is followed by another label, not base64); the only real cert body is Amazon Root CA 1.
It does embed API hostnames, which are already in every shipped unit.

1. Turn off `BlockPublicPolicy` + `RestrictPublicBuckets` on `halo-ota-prod`
2. Bucket policy: `s3:GetObject` for `*` on **`halo/ota/prod/*` only**
3. Verify unauthenticated: `curl -s -o /dev/null -w '%{http_code}' <manifest URL>` → 200

**Gate:** a device wakes at 02:00 local and takes the update. Confirm on the bench unit before
touching anything in Phase 2.

**Rejected alternative:** CloudFront keeps the bucket private but changes the URL, which means new
firmware, a full re-test, and hand-reflashing every unit. Not worth it for a public firmware image
with no secrets in it.

---

## Phase 2 — prod buckets, app path untouched

1. Create `trepo-grocery-uploads-prod` and `trepo-grocery-discards-prod`. Match the dev buckets'
   CORS, encryption and lifecycle. **Enable EventBridge notifications on both.**
2. Extend the three EventBridge patterns above to list the prod bucket alongside the dev one. Do
   NOT replace the dev entry — the app still uses it.
3. IAM: `PresignFunction` needs `s3:PutObject` on the new buckets; the three Analyze functions need
   their existing read/write grants extended.
4. `PresignFunction`: choose the bucket from an **explicit allow-list of HALO device_ids**, via env
   var `HALO_PROD_DEVICE_IDS` (comma-separated), default **empty**.

**Why an allow-list and not `device_id.startswith("halo-")`:** the app sends
`userService.deviceName`, a user-editable iPhone name. A prefix rule is one oddly-named phone away
from silently routing a real user's photo into the device bucket. An allow-list cannot do that.

**Gate:** with `HALO_PROD_DEVICE_IDS` empty, presign responses for both app and device are
byte-identical to today. This is the proof the deploy itself is safe, before any routing happens.

---

## Phase 2b — the processors are pinned to one bucket (FOUND 2026-08-24, BLOCKS PHASE 3)

Phase 3 was attempted and **failed its gate**. Rolled back the same minute.

Extending the EventBridge patterns makes the processors *fire* for prod objects, but they then
read and write a **fixed** bucket from an env var:

```js
const BUCKET_NAME = process.env.BUCKET_NAME;   // trepo-grocery-uploads-dev
Bucket: BUCKET_NAME,                            // analyze_on_upload_nodejs/app.js:165, 733, 740
```

`app.js:642` does pull `detail.bucket.name` off the event, but only to log it. So a prod object
produced:

```
[handler] Processing S3 object: { bucket: 'trepo-grocery-uploads-prod', ... }
[s3] Downloading image...
AccessDenied: ... not authorized to perform: s3:ListBucket
             on resource: "arn:aws:s3:::trepo-grocery-uploads-dev"
```

Note the denial names the **dev** bucket for a **prod** event — that mismatch is the whole tell.
It reads the prod key out of the dev bucket, the object is not there, and S3 returns 403 rather
than 404 because the role has no `ListBucket`.

**The fix:** make the three processors use the bucket from the event, falling back to
`BUCKET_NAME` when absent. Small and safe, but it is a code change to three functions on the live
path for ~12.6k iOS users, so it needs its own before/after gate on the dev path.

The alternative — deploying prod copies of all three with a different `BUCKET_NAME` — doubles the
functions and the rules and leaves two codebases to keep in step. Not worth it.

---

## Phase 3 — move ONE bench unit

1. Set `HALO_PROD_DEVICE_IDS=halo-16f8-1a6d` (the bench device).
2. Capture on it. Confirm **both** objects exist in the prod bucket, per their §8:
   `images/<user>/<device>/YYYY/MM/DD/<uuid>.jpg` **and** `resized-images/<user>/<device>/<uuid>.jpg`.
   An HTTP 200 is not sufficient; that measure has been wrong twice this month.
3. In the same window, confirm an **app** capture still lands in the dev bucket and still gets its
   twin. This is the regression that matters.
4. Watch `PresignFunction` errors and duration. p99 must stay far under 30 s.

**Gate:** both pass. Any failure → clear the env var, which reverts instantly with no deploy.

---

## Phase 4 — roll the remaining units, then soak

Add device_ids one at a time. 24h soak with:
- every capture getting its twin
- `PresignFunction` errors at 0
- no app-side `api_error` on `/presign`

---

## Rollback

`HALO_PROD_DEVICE_IDS=""`. One env var, no deploy, instant, and it cannot strand data: anything
already written to a prod bucket stays readable, and the device simply goes back to dev on its next
presign. This is the entire reason for the allow-list design.

---

## Answers to their five open questions

1. **Option 1**, confirmed viable and backend-only.
2. **No auth on the device path.** The device sends empty `x-api-key`/`Authorization` and cannot
   learn a key. Gate by `device_id` server-side instead. Adding auth would brick shipped units.
3. **OTA bucket is Matt's call**, not infra's — it is one bucket policy.
4. **Drop the 4 from 2026-08-18.** Pre-fix, and re-emittable later if ever wanted.
5. **Keep the key layout, and yes it is backend-only** — the device writes to whatever presigned
   URL we hand it, so we control the key entirely.

---

## Effort

Phase 1 is minutes once decided. Phases 2–3 are roughly half a day, mostly configuration:
one small change in `PresignFunction`, two buckets, three EventBridge patterns, IAM. Phase 4 is
elapsed time, not work.

**Nothing here requires a firmware change, and nothing invalidates their verified 6.2.0 build.**
