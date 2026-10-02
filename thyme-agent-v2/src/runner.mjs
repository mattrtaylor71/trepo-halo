import { Fault, key, scope, hash, now, publicError } from "./core.mjs";
import { record } from "./store.mjs";
import { recipeFromText } from "./recipe-text.mjs";
import { createRecipe, checkRecipe } from "./recipes.mjs";
import { readConversation } from "./conversation.mjs";
import { INSTRUCTIONS } from "./instructions.mjs";
import { fingerprint, collection } from "./gateway.mjs";
import { verifyChange } from "./verification.mjs";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export class Runner {
  constructor({
    store,
    gateway,
    provider,
    tools,
    definitions,
    clock = Date.now,
    pause = sleep,
    budgetMs = 180000,
  }) {
    Object.assign(this, {
      store,
      gateway,
      provider,
      tools,
      definitions,
      clock,
      pause,
      budgetMs,
    });
  }
  async run(pk, rid) {
    let req = await this.store.get(pk, "Q#" + rid);
    if (
      !req ||
      ["completed", "failed", "interrupted", "needs_reconciliation"].includes(
        req.status,
      )
    )
      return;
    let a;
    let s = await this.store.get(pk, key(req.sessionId));
    try {
      a = await this.gateway.actor(req.actor);
      if (scope(a) !== pk)
        throw new Fault("membership", "Household changed.", 403);
      if (s?.activeRequest !== rid)
        throw new Fault("request_conflict", "The active request changed.", 409);
    } catch (e) {
      if (e instanceof Fault && [403, 404, 409].includes(e.status)) {
        if (s?.activeRequest === rid)
          await this.finish(pk, s, req, {
            status: "failed",
            error: e.message,
            progress: null,
          });
        else
          await this.store.put(
            pk,
            "Q#" + rid,
            { ...req, status: "failed", leaseUntil: 0 },
            req.version,
          );
        return;
      }
      throw e;
    }
    // Conditional record leases fence duplicate deliveries from the durable stream.
    if (req.leaseUntil > this.clock())
      throw new Fault(
        "leased",
        "Another worker is handling this request.",
        409,
      );
    req = await this.store.put(
      pk,
      "Q#" + rid,
      {
        ...req,
        status: "running",
        attempts: req.attempts + 1,
        leaseUntil: this.clock() + this.budgetMs + 60000,
      },
      req.version,
    );
    const started = this.clock();
    try {
      if (req.kind === "approval") {
        await this.apply(a, s, req);
        return;
      }
      s = await this.progress(pk, s, "running", "Checking your kitchen and shopping list…");
      const [preferences, kitchen, shopping] = await Promise.all([
        this.gateway.read(a, "preferences"),
        this.gateway.read(a, "kitchen"),
        this.gateway.read(a, "shopping"),
      ]);
      const asOf = now();
      s = await this.progress(pk, s, "running", "Putting your answer together…", [
        `Checked ${collection(kitchen).length} kitchen items`,
        `Checked ${collection(shopping).length} shopping items`,
        "Checked food preferences",
      ]);
      if (!req.preferencesHash) {
        req = await this.store.put(
          pk,
          "Q#" + rid,
          { ...req, preferencesHash: hash(preferences.dietary) },
          req.version,
        );
      }
      const proposals = await this.store.list(pk, key(s.id, "P") + "#");
      const receipts = proposals
        .filter((x) => x.status === "applied")
        .slice(-12)
        .map((x) => ({ action: x.action, args: x.args, status: x.status }));
      const agentVersion = hash([INSTRUCTIONS, this.definitions]);
      let conversationHistory;
      // Instructions/tools are immutable at the provider. Upgrade only between
      // turns, preserving app history, recipes, approvals and explicit preferences.
      // Never rotate a submitted/in-flight request: it must reconcile first.
      if (s.providerId && s.agentVersion !== agentVersion && !req.submitted && !req.baselineTurns) {
        const oldTurns = await this.provider.turns(s.providerId);
        if (oldTurns.some((t) => !["completed", "failed", "cancelled"].includes(t.status)))
          throw new Fault("previous_turn_active", "The earlier answer is still finishing. Resume it before continuing.", 409);
        const messages = (await this.store.list(pk, key(s.id, "M") + "#"))
          .filter((m) => m.id !== req.id).sort((x, y) => x.order - y.order);
        // No silent context loss during upgrades. Oversized histories keep their
        // saved work and ask for a new chat rather than guessing what was omitted.
        conversationHistory = messages.map(({ role, text }) => ({ role, text }));
        if (Buffer.byteLength(JSON.stringify(conversationHistory)) > 180000)
          throw new Fault("history_too_large", "This long chat is saved. Start a new conversation to use Thyme’s latest improvements.", 409);
        req = await this.store.put(pk, "Q#" + rid, { ...req, upgradeHistory: conversationHistory }, req.version);
        s = await this.store.put(pk, key(s.id), { ...s, previousProviderId: s.providerId, providerId: null }, s.version);
      }
      const input = JSON.stringify({
        conversation_preferences: await readConversation(this.store, a, s.id),
        conversation_history: req.upgradeHistory,

        confirmed_changes: receipts,
        current_review_requests: {
          complete: true,
          items: proposals.map((p) => ({
            id: p.id,
            action: p.action,
            status: p.status,
            title: p.title,
            validUntil: p.validUntil,
          })),
        },
        conversation_recipe_versions: (
          await this.store.list(pk, key(s.id, "R") + "#")
        ).map((x) => ({
          id: x.recipe.id,
          revision: x.recipe.revision,
          title: x.recipe.title,
        })),
        current_time: now(),
        timezone: "America/Los_Angeles",
        current_preferences: preferences.dietary,
        food_memory: preferences.memory,
        // A fresh, complete, compact snapshot saves model round trips. It is
        // rebuilt per request and never reuses a previous turn's stock.
        current_inventory: compactInventory(kitchen, shopping, asOf),
        user_message: req.text,
      });
      if (!req.submitted) {
        // Persist the previous turn set BEFORE posting. Never guess ownership from the latest answer.
        if (!req.baselineTurns) {
          const previous = s.providerId
            ? await this.provider.turns(s.providerId)
            : [];
          if (
            previous.some(
              (t) => !["completed", "failed", "cancelled"].includes(t.status),
            )
          )
            throw new Fault(
              "previous_turn_active",
              "The earlier answer is still finishing. Resume it, or start a new conversation.",
              409,
            );
          req = await this.store.put(
            pk,
            "Q#" + rid,
            {
              ...req,
              baselineTurns: previous.map((t) => t.id),
              upgradeHistory: undefined,
              submittedInput: input,
            },
            req.version,
          );
        }
        if (!s.providerId) {
          const created = await this.provider.create({
            model: s.model,
            instructions: INSTRUCTIONS,
            tools: this.definitions,
            input: req.submittedInput,
            requestId: "thyme-create-" + rid,
          });
          await this.store.transaction([
            {
              item: record(
                pk,
                key(s.id),
                { ...s, providerId: created.id, agentVersion },
                s.version,
              ),
              expected: s.version,
            },
            {
              item: record(
                pk,
                "Q#" + rid,
                { ...req, submitted: true },
                req.version,
              ),
              expected: req.version,
            },
          ]);
          s = await this.store.get(pk, key(s.id));
          req = await this.store.get(pk, "Q#" + rid);
        } else
          await this.provider.send(
            s.providerId,
            req.submittedInput,
            "thyme-message-" + rid,
          );
        req = await this.store.put(
          pk,
          "Q#" + rid,
          { ...req, submitted: true },
          req.version,
        );
      }
      let rounds = 0;
      while (this.clock() - started < this.budgetMs) {
        await this.gateway.check(a);
        const [turns, state] = await Promise.all([
          this.provider.turns(s.providerId),
          this.provider.session(s.providerId),
        ]);
        const candidates = turns.filter(
          (t) => !(req.baselineTurns || []).includes(t.id),
        );
        if (!req.turnId && candidates.length > 1)
          throw new Fault(
            "turn_ownership",
            "This conversation needs recovery before more work can be done.",
            409,
          );
        const turn = req.turnId
          ? turns.find((t) => t.id === req.turnId)
          : candidates[0];
        if (turn && !req.turnId)
          req = await this.store.put(
            pk,
            "Q#" + rid,
            { ...req, turnId: turn.id },
            req.version,
          );
        if (state.status === "failed")
          throw new Fault(
            "agent_failed",
            "Thyme could not continue this conversation. Start a new conversation; your prior work is saved.",
            502,
          );
        // Concurrent provider reads can observe the action before its turn is
        // visible in the paginated turn list. Wait for ownership evidence.
        if (!req.turnId && (state.required_actions || []).length) {
          await this.pause(600);
          continue;
        }
        for (const action of state.required_actions || []) {
          if (action.turn_id !== req.turnId)
            throw new Fault(
              "turn_ownership",
              "A previous request is still active. No new changes were made.",
              409,
            );
          if (action.type !== "function_call")
            throw new Fault(
              "unsupported_action",
              "Thyme requested an unsupported action. No change was made.",
              502,
            );
          if (++rounds > 30)
            throw new Fault(
              "tool_limit",
              "This request needs more steps than the pilot allows. Try a smaller part.",
            );
          s = await this.progress(pk, s, "running", progressText(action.name, action.arguments));
          const result = await this.tools.call({
            actor: a,
            session: s,
            action,
          });
          await this.provider.result(s.providerId, action, result);
          s = await this.progress(pk, s, "running", "Putting your answer together…", [
            ...(s.progressDetails || []),
            ...(result.ok ? [completedText(action, result)].filter(Boolean) : []),
          ].slice(-3));
        }
        if (turn) {
          if (["completed", "failed", "cancelled"].includes(turn.status)) {
            if (turn.status !== "completed")
              throw new Fault(
                "turn_failed",
                "Thyme stopped before finishing. Your conversation and proposals are saved.",
                502,
              );
            const latest = await this.gateway.read(a, "preferences");
            if (hash(latest.dietary) !== req.preferencesHash)
              throw new Fault(
                "preferences_changed",
                "Your dietary preferences changed during this answer. Please ask again using the latest preferences.",
                409,
              );
            const items = await this.provider.items(s.providerId, turn.id);
            await this.saveMessages(pk, s, req, items, latest.dietary);
            const detail = this.provider.turn
              ? await this.provider.turn(s.providerId, turn.id)
              : turn;
            const usage = detail.usage;
            const metrics = {
              seconds: (this.clock() - req.createdAt) / 1000,
              model: s.model,
              inputTokens: usage?.input_tokens ?? null,
              outputTokens: usage?.output_tokens ?? null,
              cachedTokens: usage?.input_tokens_details?.cached_tokens ?? null,
              toolCalls: rounds,
            };
            await this.finish(pk, s, req, {
              status: "completed",
              lastTurnId: turn.id,
              metrics,
              progress: null,
            });
            return;
          }
        }
        if (!(state.required_actions || []).length) await this.pause(600);
      }
      throw new Fault(
        "timeout",
        "This request took too long. Your progress is saved; no unapproved changes were applied.",
        504,
      );
    } catch (e) {
      const fresh = await this.store.get(pk, "Q#" + rid),
        state = await this.store.get(pk, key(s.id));
      // Never replay an uncertain mutation. Its journal is deliberately left for reconciliation.
      if (fresh?.status === "needs_reconciliation") return;
      const transient =
        ["provider_error", "timeout", "unavailable"].includes(e.code) ||
        !(e instanceof Fault);
      if (transient && fresh.attempts < 3) {
        await this.store.put(
          pk,
          "Q#" + rid,
          {
            ...fresh,
            status: "retrying",
            leaseUntil: 0,
            lastError: publicError(e).code,
          },
          fresh.version,
        );
        await this.progress(
          pk,
          state,
          "running",
          "Reconnecting… your progress is saved.",
        );
        throw e;
      }
      await this.finish(pk, state, fresh, {
        status:
          transient ||
          ["turn_ownership", "tool_limit", "previous_turn_active"].includes(
            e.code,
          )
            ? "interrupted"
            : "failed",
        error: publicError(e).message,
        progress: null,
      });
    }
  }
  async progress(pk, s, status, progress, progressDetails = s.progressDetails || []) {
    if (s.status === status && s.progress === progress && hash(s.progressDetails || []) === hash(progressDetails)) return s;
    return this.store.put(pk, key(s.id), { ...s, status, progress, progressDetails }, s.version);
  }
  async saveMessages(pk, s, req, items, dietary) {
    let n = 0;
    const question = (await readConversation(this.store, { actor: req.actor, household: req.household }, s.id)).pending_question;
    if (question && question.requestId === req.id) {
      const sk = key(s.id, "M", "clarify-" + req.id);
      if (!(await this.store.get(pk, sk))) await this.store.put(pk, sk, {
        type: "message", id: "clarify-" + req.id, role: "assistant", phase: "final_answer",
        text: question.question, order: req.createdAt + 1,
      });
      return;
    }
    const rows = await this.store.list(pk, key(s.id, "P") + "#"),
      pending = rows.some((x) => x.status === "pending");
    for (const item of items) {
      if (item.type !== "message" || item.role !== "assistant") continue;
      let text = (item.content || [])
        .filter((c) => c.type === "output_text")
        .map((c) => c.text)
        .join("\n");
      if (!text) continue;
      if (item.phase !== "commentary") {
        const parsed = recipeFromText(text);
        if (parsed) {
          const existing = await this.store.list(pk, key(s.id, "R") + "#");
          if (!existing.length) {
            const recipe = checkRecipe(
              createRecipe(parsed.recipe, hash([s.id, item.id]).slice(0, 24)),
              dietary,
            );
            await this.store.put(pk, key(s.id, "R", recipe.id), {
              type: "recipe",
              recipe,
              at: now(),
              preferencesHash: hash(dietary),
              sourceMessageId: item.id,
            });
            text =
              "Your recipe is in the card below." +
              (parsed.tail ? "\n\n" + parsed.tail : "");
          } else {
            const equal = existing.some(
              (x) =>
                x.recipe.title === parsed.recipe.title &&
                x.recipe.servings === parsed.recipe.servings &&
                hash(x.recipe.ingredients.map((i) => i.text)) ===
                  hash(parsed.recipe.ingredients) &&
                hash(x.recipe.steps.map((i) => i.text)) ===
                  hash(parsed.recipe.steps),
            );
            if (!equal)
              text =
                "I kept the recipe card unchanged because this draft did not match the saved conversation version. Tell me the specific change you want and I’ll update that recipe.";
            else
              text =
                "Your current recipe is in the card below." +
                (parsed.tail ? "\n\n" + parsed.tail : "");
          }
        }
      }

      if (
        pending &&
        /\b(?:I(?:'ve| have)? (?:added|removed|saved|updated)|(?:added|removed|saved|updated)[^.!?]{0,50}(?:your (?:list|kitchen|recipes)))\b/i.test(
          text,
        )
      )
        text =
          "I’ve prepared the changes for your review below. Nothing has been applied yet.";
      const sk = key(s.id, "M", item.id);
      if (!(await this.store.get(pk, sk)))
        await this.store.put(pk, sk, {
          type: "message",
          id: item.id,
          role: "assistant",
          text: text.slice(0, 24000),
          phase: item.phase,
          order: req.createdAt + 1 + n++ / 100,
        });
    }
  }
  async finish(pk, s, req, patch) {
    const status = patch.status || "completed";
    await this.store.transaction([
      {
        item: record(
          pk,
          key(s.id),
          {
            ...s,
            ...patch,
            activeRequest: status === "interrupted" ? req.id : null,
            updatedAt: now(),
          },
          s.version,
        ),
        expected: s.version,
      },
      {
        item: record(
          pk,
          "Q#" + req.id,
          { ...req, status, leaseUntil: 0, completedAt: now() },
          req.version,
        ),
        expected: req.version,
      },
    ]);
  }
  async apply(a, s, req) {
    const pk = scope(a),
      sk = key(s.id, "P", req.proposalId);
    let p = await this.store.get(pk, sk);
    if (!p || p.operationId !== req.id)
      throw new Fault("approval", "Approval no longer matches.", 409);
    if (p.status === "applied") {
      await this.finish(pk, s, req, { status: "completed", progress: null });
      return;
    }
    if (p.status === "applying") {
      await this.uncertain(
        pk,
        s,
        req,
        p,
        "The connection ended during this change. Check the result before repeating it.",
      );
      return;
    }
    if (!["approved", "verifying", "needs_reconciliation"].includes(p.status))
      throw new Fault("approval", "This change is not approved.", 409);
    if (p.status === "approved") {
      if (
        p.dietaryFence &&
        hash((await this.gateway.read(a, "preferences")).dietary) !==
          p.dietaryFence
      )
        throw new Fault(
          "preferences_changed",
          "Your food preferences changed. Review the plan again.",
          409,
        );
      if (p.recipeFence) {
        const current = await this.store.get(
          pk,
          key(s.id, "R", p.recipeFence.id),
        );
        if (current?.recipe.revision !== p.recipeFence.revision)
          throw new Fault(
            "recipe_conflict",
            "The recipe changed after you reviewed this save. Ask to save the latest version.",
            409,
          );
        const preferences = await this.gateway.read(a, "preferences");
        if (current.preferencesHash !== hash(preferences.dietary))
          throw new Fault(
            "preferences_changed",
            "Your food preferences changed. Review the recipe again before saving.",
            409,
          );
      }
      const before = await this.gateway.read(a, p.resource);
      if (fingerprint(before) !== p.beforeHash)
        throw new Fault(
          "stale_approval",
          "Your data changed after approval. Ask Thyme to prepare it again.",
          409,
        );
      p = await this.store.put(pk, sk, { ...p, status: "applying" }, p.version);
      s = await this.progress(
        pk,
        s,
        "verifying",
        "Applying your approved change…",
      );
      let result;
      try {
        result = await this.gateway.mutate(
          a,
          p.action,
          p.args,
          p.operationId,
          p.before,
        );
      } catch (e) {
        await this.uncertain(pk, s, req, p, publicError(e).message);
        return;
      }
      // Persist the result before readback. A resumed verification NEVER calls mutate again.
      p = await this.store.put(
        pk,
        sk,
        { ...p, result, status: "verifying" },
        p.version,
      );
    }
    let after;
    try {
      after = await this.gateway.read(a, p.resource);
    } catch {
      await this.uncertain(
        pk,
        s,
        req,
        p,
        "The change returned successfully, but the latest data could not be checked. Please refresh before repeating it.",
      );
      return;
    }
    const changed = verifyChange(
      p.action,
      p.args,
      p.before,
      after,
      p.result,
    ).verified;
    // An unchanged snapshot may be a legitimate no-op, but it is not proof of the requested effect.
    if (!changed) {
      await this.uncertain(
        pk,
        s,
        req,
        p,
        "The server accepted the change, but the expected update is not visible yet. Check your data before retrying.",
      );
      return;
    }
    p = await this.store.put(
      pk,
      sk,
      {
        ...p,
        status: "applied",
        error: null,
        afterHash: fingerprint(after),
        verification: "Confirmed by reading your updated Trepo data.",
        appliedAt: now(),
      },
      p.version,
    );
    await this.store
      .put(pk, key(s.id, "M", "approval-" + p.id), {
        type: "message",
        id: "approval-" + p.id,
        role: "assistant",
        phase: "final_answer",
        text: "Done — " + p.summary,
        order: req.createdAt + 1,
      })
      .catch((e) => {
        if (e.code !== "conflict") throw e;
      });
    await this.finish(pk, s, req, { status: "completed", progress: null });
  }
  async uncertain(pk, s, req, p, message) {
    const current = await this.store.get(pk, key(s.id, "P", p.id));
    await this.store.put(
      pk,
      current.sk,
      { ...current, status: "needs_reconciliation", error: message },
      current.version,
    );
    await this.finish(pk, s, req, {
      status: "needs_reconciliation",
      error: message,
      progress: null,
    });
  }
}
function progressText(name, args = {}) {
  if (name === "read_trepo") return ({
    kitchen: "Checking your kitchen…", shopping: "Checking your shopping list…",
    saved_recipes: "Looking through your saved recipes…", preferences: "Checking your food preferences…",
    suggestions: "Finding recipe ideas…", calendar: "Checking your meal calendar…",
  })[args.resource] || "Checking your current Trepo data…";
  if (name.includes("recipe")) return "Working on your recipe…";
  if (name.startsWith("request_"))
    return "Preparing the changes for your review…";
  return "Working through your request…";
}

function completedText(action, result) {
  if (action.name === "read_trepo") {
    const label = { kitchen: "kitchen items", shopping: "shopping items", saved_recipes: "saved recipes" }[action.arguments.resource];
    return label ? `Checked ${collection(result.data).length} ${label}` : "Checked your Trepo data";
  }
  if (action.name === "create_recipe" || action.name === "edit_recipe") return "Recipe ready";
  if (action.name.startsWith("request_")) return "Change ready for your review";
  return null;
}

export function compactInventory(kitchen, shopping, asOf) {
  const pick = (row, keys) => Object.fromEntries(keys.filter((k) => row[k] !== undefined && row[k] !== null).map((k) => [k, row[k]]));
  const snapshot = {
    asOf, complete: true,
    kitchen: collection(kitchen).map((row) => pick(row, ["id", "item_name", "brand", "variant", "category", "quantity_value", "quantity_unit", "fill_percent", "remaining_quantity", "is_opened", "storage_location", "expiration_date", "created_at", "ingredients"])),
    shopping: collection(shopping).map((row) => pick(row, ["shopping_id", "household_item_uuid", "item_name", "quantity", "store", "action"])),
  };
  // Never silently truncate a kitchen and call it complete.
  return Buffer.byteLength(JSON.stringify(snapshot)) <= 60000 ? snapshot : { asOf, complete: false, use_read_trepo: true };
}
