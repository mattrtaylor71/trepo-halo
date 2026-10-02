import { Fault } from "./core.mjs";
export const MODELS = ["gpt-6.1-sol", "gpt-6-astra", "gpt-6-luna"];
export class AgentProvider {
  constructor(
    apiKey,
    { fetcher = fetch, base = "https://api.openai.com/v1" } = {},
  ) {
    this.apiKey = apiKey;
    this.fetcher = fetcher;
    this.base = base;
  }
  async request(path, body, idem) {
    const r = await this.fetcher(this.base + path, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
        "OpenAI-Beta": "agents=v1",
        ...(idem ? { "Idempotency-Key": idem } : {}),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      signal: AbortSignal.timeout(35000),
    });
    if (!r.ok) {
      const detail = (await r.text()).slice(0, 2000);
      const e = new Fault(
        "provider_error",
        r.status === 429
          ? "Thyme is busy. Your request is saved and will be retried."
          : "Thyme could not finish this answer. Your request is saved.",
        502,
      );
      e.providerStatus = r.status;
      e.providerDetail = detail;
      throw e;
    }
    const raw = await r.text();
    return raw ? JSON.parse(raw) : {};
  }
  create({ model, instructions, tools, input, requestId }) {
    return this.request(
      "/agents/sessions",
      {
        agent: {
          model,
          instructions,
          reasoning: { effort: "low" },
          text: { verbosity: "low" },
          tools,
        },
        environment: { type: "none" },
        input,
        stream: false,
        metadata: { application: "trepo-thyme-v2", request_id: requestId },
      },
      requestId,
    );
  }
  send(sid, text, requestId) {
    return this.request(
      `/agents/sessions/${sid}/events`,
      {
        events: [
          {
            type: "agent.session.input.message",
            input: [{ role: "user", content: [{ type: "input_text", text }] }],
          },
        ],
      },
      requestId,
    );
  }
  session(sid) {
    return this.request(`/agents/sessions/${sid}`);
  }
  async list(sid, type, extra = "") {
    let after,
      out = [];
    do {
      const r = await this.request(
        `/agents/sessions/${sid}/${type}?order=asc&limit=100${extra}${after ? "&after=" + encodeURIComponent(after) : ""}`,
      );
      out.push(...r.data);
      if (!r.has_more) break;
      if (!r.last_id || r.last_id === after)
        throw new Fault(
          "pagination",
          "Could not restore the full conversation.",
          502,
        );
      after = r.last_id;
    } while (true);
    return out;
  }
  turn(sid, tid) {
    return this.request(`/agents/sessions/${sid}/turns/${tid}`);
  }
  turns(sid) {
    return this.list(sid, "turns");
  }
  async items(sid, turn) {
    const items = await this.list(
      sid,
      "items",
      turn ? "&turn_id=" + encodeURIComponent(turn) : "",
    );
    // The provider currently returns earlier items even with turn_id in the
    // query. Enforce ownership locally so old questions/drafts never reappear.
    return turn ? items.filter((item) => item.turn_id === turn) : items;
  }
  result(sid, action, result) {
    return this.request(
      `/agents/sessions/${sid}/events`,
      {
        events: [
          {
            type: "agent.session.input.tool_result",
            turn_id: action.turn_id,
            call_id: action.call_id,
            success: true,
            output: JSON.stringify(result),
          },
        ],
      },
      "thyme-result-" + action.call_id,
    );
  }
}
