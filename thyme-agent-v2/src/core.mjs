import {
  createHash,
  createHmac,
  timingSafeEqual,
  randomUUID,
} from "node:crypto";
export class Fault extends Error {
  constructor(code, message, status = 400) {
    super(message);
    this.code = code;
    this.status = status;
  }
}
export const id = () => randomUUID();
export const stable = (x) =>
  JSON.stringify(x, (_, v) =>
    v && typeof v === "object" && !Array.isArray(v)
      ? Object.fromEntries(
          Object.entries(v).sort(([a], [b]) => a.localeCompare(b)),
        )
      : v,
  );
export const hash = (x) =>
  createHash("sha256")
    .update(typeof x === "string" ? x : stable(x))
    .digest("hex");
export const now = () => new Date().toISOString();
export const scope = (a) => `A#${a.actor}#H#${a.household}`;
export const key = (s, type = "", suffix = "") =>
  `S#${s}${type ? "#" + type : ""}${suffix ? "#" + suffix : ""}`;
export function checkId(v, label = "ID") {
  if (typeof v !== "string" || !/^[a-zA-Z0-9_-]{1,100}$/.test(v))
    throw new Fault("invalid_id", `Invalid ${label}.`);
  return v;
}
export function signRequest(
  { method = "POST", path = "/", timestamp, nonce, body },
  secret,
) {
  return createHmac("sha256", secret)
    .update([method, path, timestamp, nonce, hash(body)].join("\n"))
    .digest("hex");
}
export function verifyRequest(event, secret, clock = Date.now()) {
  if (!secret)
    throw new Fault(
      "unavailable",
      "The private connection is unavailable.",
      503,
    );
  const h = Object.fromEntries(
    Object.entries(event.headers || {}).map(([k, v]) => [k.toLowerCase(), v]),
  );
  const timestamp = h["x-thyme-time"],
    nonce = h["x-thyme-nonce"],
    signature = h["x-thyme-signature"];
  if (
    !/^\d{13}$/.test(timestamp || "") ||
    Math.abs(clock - Number(timestamp)) > 90000 ||
    !/^[a-zA-Z0-9_-]{16,80}$/.test(nonce || "") ||
    !/^[a-f0-9]{64}$/.test(signature || "")
  )
    throw new Fault(
      "unauthorized",
      "Sign in to your private test kitchen.",
      401,
    );
  const body = event.isBase64Encoded
    ? Buffer.from(event.body || "", "base64").toString()
    : event.body || "";
  if (Buffer.byteLength(body) > 32000)
    throw new Fault("too_large", "Please send a shorter message.", 413);
  const expected = signRequest(
    {
      method: event.requestContext?.http?.method || "POST",
      path: event.rawPath || "/",
      timestamp,
      nonce,
      body,
    },
    secret,
  );
  if (!timingSafeEqual(Buffer.from(expected), Buffer.from(signature)))
    throw new Fault(
      "unauthorized",
      "Sign in to your private test kitchen.",
      401,
    );
  let data;
  try {
    data = JSON.parse(body);
  } catch {
    throw new Fault("invalid_json", "Invalid request.");
  }
  if (!data || typeof data !== "object" || Array.isArray(data))
    throw new Fault("invalid_json", "Invalid request.");
  return { data, nonce };
}
export function publicError(error) {
  return {
    code: error instanceof Fault ? error.code : "unavailable",
    message:
      error instanceof Fault
        ? error.message
        : "Thyme hit a connection problem. Your saved progress is safe. Please try again.",
  };
}
export function legacyResponse(s) {
  const last = [...(s.messages || [])]
    .reverse()
    .find((m) => m.role === "assistant" && m.phase !== "commentary");
  return {
    ok: s.status === "completed",
    status: s.status,
    response: last?.text || s.progress || "",
    reply: last?.text || "",
    session_id: s.id,
    recipes: s.recipes || [],
    pending_actions: s.proposals || [],
    error: s.error || null,
  };
}
