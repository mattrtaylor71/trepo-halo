// $/1M tokens. 4.x/4o rates are OpenAI published list prices. gpt-5.x rates are
// ESTIMATES (est:true) scaled by tier — flagged so the report never presents an
// unconfirmed number as fact. TOKEN COUNTS from usage are the hard currency; cost
// is derived. Edit these once real 5.x pricing is known and re-run the cost math.
export const PRICING = {
  "gpt-4.1":             { in: 2.00,  out: 8.00,  est: false },
  "gpt-4.1-mini":        { in: 0.40,  out: 1.60,  est: false },
  "gpt-4.1-nano":        { in: 0.10,  out: 0.40,  est: false },
  "gpt-4o":              { in: 2.50,  out: 10.00, est: false },
  "gpt-4o-mini":         { in: 0.15,  out: 0.60,  est: false },
  // ---- estimates (confirm before quoting) ----
  "gpt-5.4-2026-03-05":  { in: 1.25,  out: 10.00, est: true },
  "gpt-5.4":             { in: 1.25,  out: 10.00, est: true },
  "gpt-5.4-mini":        { in: 0.25,  out: 2.00,  est: true },
  "gpt-5.4-nano":        { in: 0.05,  out: 0.40,  est: true },
  "gpt-5.5":             { in: 1.25,  out: 10.00, est: true },
  "gpt-5.6-sol":         { in: 2.00,  out: 12.00, est: true },
  "gpt-5.6-luna":        { in: 1.50,  out: 10.00, est: true },
  "gpt-5.6-terra":       { in: 2.00,  out: 12.00, est: true },
};

export function costUSD(model, usage) {
  const p = PRICING[model];
  if (!p) return { usd: null, est: null };
  const inTok = usage.prompt_tokens ?? usage.input_tokens ?? 0;
  const outTok = usage.completion_tokens ?? usage.output_tokens ?? 0;
  return { usd: (inTok * p.in + outTok * p.out) / 1e6, est: p.est };
}
export const priceIsEst = (m) => !!PRICING[m]?.est;
