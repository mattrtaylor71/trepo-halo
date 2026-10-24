// Shared foreground inference transport. No automatic retry: callers own their
// durable operation IDs and must reconcile uncertain tool writes before retrying.
export class ResponsesClient {
  constructor(apiKey, {fetcher = fetch, base = 'https://api.openai.com/v1', clock = Date.now} = {}) {
    Object.assign(this, {apiKey, fetcher, base, clock});
  }
  async complete({model = 'gpt-6.1-sol', input, tools = [], deadline = this.clock() + 25000,
    requestId, toolChoice = 'auto', maxOutputTokens = 3000}) {
    const remaining = deadline - this.clock();
    if (remaining <= 0) throw Object.assign(new Error('Thyme ran out of time. Check the saved result before retrying.'), {code: 'timeout'});
    const started = this.clock();
    const response = await this.fetcher(this.base + '/responses', {
      method: 'POST', headers: {'Authorization': `Bearer ${this.apiKey}`, 'Content-Type': 'application/json',
        ...(requestId ? {'Idempotency-Key': requestId} : {})},
      signal: AbortSignal.timeout(Math.max(1, Math.min(35000, remaining))),
      body: JSON.stringify({model, input, tools: tools.map(t => {
        const f = t.function || t;
        return {type: 'function', name: f.name, description: f.description, parameters: f.parameters, strict: false};
      }), tool_choice: toolChoice?.function ? {type: 'function', name: toolChoice.function.name} : toolChoice,
      reasoning: {effort: model === 'gpt-6-luna' ? 'none' : 'low'}, text: {verbosity: 'low'},
      max_output_tokens: maxOutputTokens, store: false, include: ['reasoning.encrypted_content']}),
    });
    if (!response.ok) {
      await response.body?.cancel();
      throw Object.assign(new Error('Thyme could not finish this answer. Your request can be checked safely.'),
        {code: 'provider_error', providerStatus: response.status});
    }
    const result = await response.json(); // AbortSignal also covers the response body.
    if (result.status !== 'completed' || !Array.isArray(result.output))
      throw Object.assign(new Error('Thyme stopped before finishing this answer.'), {code: 'incomplete_response'});
    return {...result, elapsedMs: this.clock() - started};
  }
}

export function chatInput(messages, outputByCall = new Map()) {
  return messages.flatMap(m => {
    const original = m.tool_calls?.map(t => outputByCall.get(t.id)).find(Boolean);
    if (original) return original; // Preserve encrypted reasoning and exact call items for tool continuation.
    if (m.role === 'tool') return [{type: 'function_call_output', call_id: m.tool_call_id, output: m.content}];
    const result = [];
    if (m.content) result.push({role: m.role, content: m.content});
    for (const t of m.tool_calls || []) result.push({type: 'function_call', call_id: t.id,
      name: t.function.name, arguments: t.function.arguments});
    return result;
  });
}

export function completionMessage(response) {
  return {role: 'assistant', content: response.output.filter(x => x.type === 'message')
    .flatMap(x => x.content || []).filter(x => x.type === 'output_text').map(x => x.text).join('\n'),
    tool_calls: response.output.filter(x => x.type === 'function_call').map(x => ({id: x.call_id,
      type: 'function', function: {name: x.name, arguments: x.arguments}}))};
}
