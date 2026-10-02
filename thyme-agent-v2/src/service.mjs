import {
  Fault,
  id,
  key,
  scope,
  hash,
  now,
  checkId,
  publicError,
} from "./core.mjs";
import { checkRecipe } from "./recipes.mjs";
import { record } from "./store.mjs";
import { MODELS } from "./provider.mjs";
import { collection, fingerprint } from "./gateway.mjs";
export class Service {
  constructor({ store, gateway, provider, clock = Date.now }) {
    Object.assign(this, { store, gateway, provider, clock });
  }
  async session(a, sid) {
    checkId(sid);
    const pk = scope(a);
    let s = await this.store.get(pk, key(sid));
    if (!s || s.type !== "session")
      throw new Fault("not_found", "Conversation not found.", 404);
    if (
      s.status === "completed" &&
      s.lastTurnId &&
      s.metrics?.inputTokens == null &&
      this.provider
    ) {
      try {
        const t = await this.provider.turn(s.providerId, s.lastTurnId);
        if (t.usage)
          s = await this.store.put(
            pk,
            key(sid),
            {
              ...s,
              metrics: {
                ...s.metrics,
                inputTokens: t.usage.input_tokens ?? null,
                outputTokens: t.usage.output_tokens ?? null,
                cachedTokens:
                  t.usage.input_tokens_details?.cached_tokens ?? null,
              },
            },
            s.version,
          );
      } catch {}
    }
    const rows = await this.store.list(pk, key(sid) + "#");
    const prefs = {
      dietary: this.gateway.dietary
        ? await this.gateway.dietary(a)
        : (await this.gateway.read(a, "preferences")).dietary,
    };
    const recipes = rows
      .filter(
        (x) =>
          x.type === "recipe" &&
          !x.blocked &&
          x.preferencesHash === hash(prefs.dietary),
      )
      .filter((x) => {
        try {
          checkRecipe(x.recipe, prefs.dietary);
          return true;
        } catch {
          return false;
        }
      })
      .map((x) => x.recipe);
    return {
      ...s,
      messages: rows
        .filter((x) => x.type === "message")
        .sort((x, y) => x.order - y.order)
        .map(({ id, role, text, phase }) => ({ id, role, text, phase })),
      recipes,
      proposals: rows
        .filter((x) => x.type === "proposal")
        .map(({ id, title, summary, args, status, verification, error }) => ({
          id,
          title,
          summary,
          args,
          status,
          verification,
          error,
        })),
    };
  }
  async bootstrap(a) {
    const [kitchen, shopping, prefs, rows] = await Promise.all([
      this.gateway.read(a, "kitchen"),
      this.gateway.read(a, "shopping"),
      this.gateway.read(a, "preferences"),
      this.store.list(scope(a), "S#"),
    ]);
    return {
      name: a.name,
      kitchen: collection(kitchen),
      shopping: collection(shopping),
      preferences: prefs.dietary,
      memory: prefs.memory,
      sessions: rows
        .filter((x) => x.type === "session")
        .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
        .slice(0, 50)
        .map(({ id, title, status }) => ({ id, title, status })),
      syncedAt: now(),
    };
  }
  async message(a, input) {
    const text = String(input.text || "").trim();
    if (!text || text.length > 12000)
      throw new Fault(
        "invalid_message",
        "Write a message of up to 12,000 characters.",
      );
    const rid = checkId(input.requestId, "request ID"),
      pk = scope(a),
      intent = hash([input.sessionId || null, text, input.model || MODELS[0]]),
      old = await this.store.get(pk, "Q#" + rid);
    if (old) {
      if (old.intent !== intent)
        throw new Fault(
          "request_conflict",
          "That request ID was already used for a different message.",
          409,
        );
      return this.session(a, old.sessionId);
    }
    const sid = input.sessionId ? checkId(input.sessionId) : id(),
      s = input.sessionId ? await this.store.get(pk, key(sid)) : null;
    if (input.sessionId && !s)
      throw new Fault("not_found", "Conversation not found.", 404);
    if (s?.activeRequest)
      throw new Fault(
        "busy",
        "Let Thyme finish this request before sending another.",
        409,
      );
    if (s?.status === "needs_reconciliation")
      throw new Fault(
        "reconcile_first",
        "Please check the uncertain change before starting more work.",
        409,
      );
    const model = s?.model || input.model || MODELS[0];
    if (!MODELS.includes(model))
      throw new Fault("model", "Choose one of the available thinking styles.");
    const day = new Date(this.clock()).toISOString().slice(0, 10),
      quota = await this.store.get(pk, "B#" + day);
    if ((quota?.count || 0) >= 100)
      throw new Fault(
        "daily_limit",
        "This test kitchen has reached today’s 100-request limit. Your chats are saved.",
        429,
      );
    const timestamp = this.clock(),
      session = {
        ...(s || {}),
        type: "session",
        id: sid,
        actor: a.actor,
        household: a.household,
        title: s?.title || text.slice(0, 65),
        model,
        status: "queued",
        activeRequest: rid,
        progress: "Request saved. Thyme is getting started…",
        progressDetails: [],
        error: null,
        updatedAt: now(),
      };
    const request = {
      type: "request",
      id: rid,
      sessionId: sid,
      actor: a.actor,
      household: a.household,
      kind: "message",
      text,
      model,
      intent,
      status: "queued",
      createdAt: timestamp,
      attempts: 0,
    };
    await this.store.transaction([
      {
        item: record(
          pk,
          "B#" + day,
          { type: "budget", count: (quota?.count || 0) + 1 },
          quota?.version ?? null,
        ),
        expected: quota?.version ?? null,
      },
      {
        item: record(pk, key(sid), session, s?.version ?? null),
        expected: s?.version ?? null,
      },
      { item: record(pk, "Q#" + rid, request), expected: null },
      {
        item: record(pk, key(sid, "M", rid), {
          type: "message",
          id: rid,
          role: "user",
          text,
          order: timestamp,
        }),
        expected: null,
      },
    ]);
    return this.session(a, sid);
  }
  async decide(a, input) {
    const pk = scope(a),
      sid = checkId(input.sessionId),
      pid = checkId(input.proposalId),
      rid = checkId(input.requestId),
      decision = input.decision;
    if (!["approve", "reject"].includes(decision))
      throw new Fault("decision", "Choose approve or not now.");
    const s = await this.store.get(pk, key(sid)),
      p = await this.store.get(pk, key(sid, "P", pid));
    if (!s || !p)
      throw new Fault("not_found", "That proposed change was not found.", 404);
    if (p.status !== "pending") return this.session(a, sid);
    if (s.activeRequest || s.status === "needs_reconciliation")
      throw new Fault(
        "busy",
        "Resolve the saved request before approving another change.",
        409,
      );
    if (p.validUntil < this.clock())
      throw new Fault(
        "approval_expired",
        "This proposal expired. Ask Thyme to prepare it again.",
        409,
      );
    if (decision === "reject") {
      await this.store.put(
        pk,
        key(sid, "P", pid),
        { ...p, status: "rejected" },
        p.version,
      );
      return this.session(a, sid);
    }
    const fresh = await this.gateway.read(a, p.resource);
    if (fingerprint(fresh) !== p.beforeHash)
      throw new Fault(
        "stale_approval",
        "Your data changed since this proposal. Ask Thyme to prepare it again.",
        409,
      );
    const request = {
      type: "request",
      id: rid,
      sessionId: sid,
      actor: a.actor,
      household: a.household,
      kind: "approval",
      proposalId: pid,
      status: "queued",
      createdAt: this.clock(),
      attempts: 0,
      intent: hash([sid, pid, "approve"]),
    };
    await this.store.transaction([
      {
        item: record(
          pk,
          key(sid),
          {
            ...s,
            status: "queued",
            progress: "Your approval is saved. Applying the change…",
            activeRequest: rid,
            error: null,
            updatedAt: now(),
          },
          s.version,
        ),
        expected: s.version,
      },
      {
        item: record(
          pk,
          key(sid, "P", pid),
          { ...p, status: "approved", approvedAt: now(), operationId: rid },
          p.version,
        ),
        expected: p.version,
      },
      { item: record(pk, "Q#" + rid, request), expected: null },
    ]);
    return this.session(a, sid);
  }
  async resume(a, input) {
    const pk = scope(a),
      sid = checkId(input.sessionId),
      s = await this.store.get(pk, key(sid));
    if (!s) throw new Fault("not_found", "Conversation not found.", 404);
    const pid = input.proposalId ? checkId(input.proposalId) : null,
      p = pid ? await this.store.get(pk, key(sid, "P", pid)) : null;
    if (pid && (!p || p.status !== "needs_reconciliation"))
      return this.session(a, sid);
    const rid = p?.operationId || s.activeRequest,
      req = rid ? await this.store.get(pk, "Q#" + rid) : null;
    if (s.activeRequest && s.activeRequest !== rid)
      throw new Fault(
        "busy",
        "Another request is still running. Wait for it to finish.",
        409,
      );
    if (req && ["queued", "running", "verifying"].includes(req.status))
      return this.session(a, sid);
    if (!req || (!p && req.status !== "interrupted"))
      return this.session(a, sid);
    await this.store.transaction([
      {
        item: record(
          pk,
          key(sid),
          {
            ...s,
            status: "queued",
            activeRequest: rid,
            error: null,
            progress: p
              ? "Checking the saved result…"
              : "Resuming your saved request…",
            updatedAt: now(),
          },
          s.version,
        ),
        expected: s.version,
      },
      {
        item: record(
          pk,
          "Q#" + rid,
          { ...req, status: "queued", attempts: 0, leaseUntil: 0 },
          req.version,
        ),
        expected: req.version,
      },
    ]);
    return this.session(a, sid);
  }
  async feedback(a, input) {
    const sid = checkId(input.sessionId);
    await this.session(a, sid);
    if (!["helpful", "unhelpful"].includes(input.rating))
      throw new Fault("rating", "Choose helpful or unhelpful.");
    await this.store.put(scope(a), key(sid, "F", id()), {
      type: "feedback",
      rating: input.rating,
      note: String(input.note || "").slice(0, 2000),
      at: now(),
    });
    return { ok: true };
  }
  async dispatch(a, input) {
    await this.gateway.check(a);
    switch (input.op) {
      case "bootstrap":
        return this.bootstrap(a);
      case "session":
        return this.session(a, input.sessionId);
      case "message":
        return this.message(a, input);
      case "decide":
        return this.decide(a, input);
      case "resume":
        return this.resume(a, input);
      case "feedback":
        return this.feedback(a, input);
      default:
        throw new Fault("operation", "Unknown operation.");
    }
  }
}
