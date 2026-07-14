import mysql from 'mysql2/promise';
import OpenAI from 'openai';
import { GroceryItem } from '../openai/identifyGrocery';
import { getHouseholdMemberIds } from './householdSync';

// Kitchen is shared-primary post-migration: swap candidates must come from
// shared_kitchen (household-scoped), not the frozen per-owner {owner}_prod_kitchen.
const USE_SHARED_TABLES = String(process.env.USE_SHARED_TABLES || 'false').toLowerCase() === 'true';

type SwapGeneratedBy = 'fast_local' | 'deep_openai';

export interface SwapSuggestions {
  candidate_ids: string[];
  generated_by: SwapGeneratedBy;
  generated_at: string;
  reason_summary: string;
  score_metadata?: Array<{
    id: string;
    score: number;
    product_name: string | null;
    brand: string | null;
  }>;
}

export interface KitchenSwapCandidateRow {
  _id: string;
  job_id: string | null;
  product_name: string | null;
  brand: string | null;
  variant: string | null;
  category: string | null;
  barcode: string | null;
  _createdDate: string | Date | null;
}

export interface RankedKitchenSwapCandidate extends KitchenSwapCandidateRow {
  score: number;
}

const LOCAL_MIN_SCORE = Math.max(0, Math.min(1, Number(process.env.SWAP_LOCAL_MIN_SCORE || 0.32)));
const MAX_LOCAL_CANDIDATES = Math.max(1, Math.min(12, Number(process.env.SWAP_MAX_CANDIDATES || 8)));
const OPENAI_SHORTLIST_LIMIT = Math.max(MAX_LOCAL_CANDIDATES, Math.min(30, Number(process.env.SWAP_OPENAI_SHORTLIST_LIMIT || 18)));

const GENERIC_MATCH_TOKENS = new Set([
  'fresh', 'ripe', 'whole', 'raw', 'halved', 'sliced', 'diced',
  'item', 'food', 'grocery', 'product', 'pack', 'package', 'bag', 'box', 'container',
]);

const CONFLICT_TOKEN_GROUPS = [
  ['red', 'white', 'yellow', 'green', 'purple'],
  ['lime', 'lemon', 'grapefruit', 'orange'],
  ['plain', 'vanilla', 'chocolate', 'strawberry', 'blueberry', 'raspberry', 'berry', 'mango', 'pineapple', 'coconut'],
  ['spearmint', 'peppermint', 'wintergreen', 'cinnamon'],
];

let client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!client) {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY is required for deep swap ranking');
    }
    client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }
  return client;
}

function getOpenAIModel(): string {
  return process.env.OPENAI_MODEL || 'gpt-5.4-2026-03-05';
}

function cleanText(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function normalize(value: string | null | undefined): string {
  if (!value) return '';
  return value
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeMatchName(value: string | null | undefined): string {
  if (!value) return '';
  let raw = value.toLowerCase();
  raw = raw.replace(/\([^)]*\d+[^)]*\)/g, ' ');
  raw = raw.replace(/\b\d+(\.\d+)?\s*(oz|ounce|ounces|lb|lbs|pound|pounds|g|kg|ml|l|liter|liters|fl|floz|gal|gallon|gallons|qt|quart|quarts|pt|pint|pints)\b/g, ' ');
  raw = raw.replace(/[^\w\s]/g, ' ');
  raw = raw.replace(/\s+/g, ' ').trim();
  if (!raw) return '';
  return raw
    .split(' ')
    .filter(Boolean)
    .filter((token) => !GENERIC_MATCH_TOKENS.has(token))
    .join(' ');
}

function tokenSet(...values: Array<string | null | undefined>): Set<string> {
  const merged = values
    .map((value) => normalizeMatchName(value))
    .filter(Boolean)
    .join(' ');
  return new Set(merged.split(/\s+/).filter(Boolean));
}

function tokenSimilarity(a: string | null | undefined, b: string | null | undefined): number {
  const setA = tokenSet(a);
  const setB = tokenSet(b);
  if (setA.size === 0 && setB.size === 0) return 1;
  if (setA.size === 0 || setB.size === 0) return 0;
  let overlap = 0;
  for (const token of setA) {
    if (setB.has(token)) overlap += 1;
  }
  return (2 * overlap) / (setA.size + setB.size);
}

function overlappingTokens(a: Set<string>, b: Set<string>): string[] {
  const matches: string[] = [];
  for (const token of a) {
    if (b.has(token)) matches.push(token);
  }
  return matches;
}

function getConflictPenalty(
  left: { product_name?: string | null; variant?: string | null },
  right: { product_name?: string | null; variant?: string | null },
): number {
  const leftTokens = tokenSet(left.product_name, left.variant);
  const rightTokens = tokenSet(right.product_name, right.variant);
  if (leftTokens.size === 0 || rightTokens.size === 0) return 0;

  let penalty = 0;
  for (const group of CONFLICT_TOKEN_GROUPS) {
    const leftGroup = group.filter((token) => leftTokens.has(token));
    const rightGroup = group.filter((token) => rightTokens.has(token));
    if (leftGroup.length > 0 && rightGroup.length > 0 && !leftGroup.some((token) => rightGroup.includes(token))) {
      penalty += 0.28;
    }
  }
  return Math.min(penalty, 0.45);
}

function clampScore(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function rankCandidate(item: GroceryItem, row: KitchenSwapCandidateRow): number {
  const itemBarcode = normalize(item.barcode || '');
  const rowBarcode = normalize(row.barcode || '');
  if (itemBarcode && rowBarcode && itemBarcode === rowBarcode) {
    return 1;
  }

  const itemName = normalizeMatchName(item.product_name || '');
  const rowName = normalizeMatchName(row.product_name || '');
  const nameSim = tokenSimilarity(itemName, rowName);
  const brandSim = tokenSimilarity(item.brand || '', row.brand || '');
  const variantSim = tokenSimilarity(item.variant || '', row.variant || '');
  const categorySim = tokenSimilarity(item.category || '', row.category || '');
  const itemNameTokens = tokenSet(item.product_name, item.variant);
  const rowNameTokens = tokenSet(row.product_name, row.variant);
  const itemTokens = tokenSet(item.product_name, item.variant, item.category);
  const rowTokens = tokenSet(row.product_name, row.variant, row.category);
  const sharedNameTokens = overlappingTokens(itemNameTokens, rowNameTokens);
  const sharedTokens = overlappingTokens(itemTokens, rowTokens);
  const conflictPenalty = getConflictPenalty(item, row);

  if (nameSim < 0.18 && sharedTokens.length === 0 && categorySim < 0.34) {
    return 0;
  }

  let score = 0.48 * nameSim + 0.10 * brandSim + 0.08 * variantSim + 0.14 * categorySim;
  if (itemName && rowName && itemName === rowName) score += 0.14;
  if (sharedNameTokens.length >= 1) score += 0.12;
  if (sharedNameTokens.length >= 1 && item.category && row.category && normalize(item.category) === normalize(row.category)) {
    score += 0.08;
  }
  if (sharedTokens.length >= 2) score += 0.06;
  if (item.brand && row.brand && normalize(item.brand) === normalize(row.brand)) score += 0.04;
  if (item.category && row.category && normalize(item.category) === normalize(row.category)) score += 0.05;
  score -= conflictPenalty;

  return clampScore(score);
}

async function getConnection(): Promise<mysql.Connection> {
  const { DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME } = process.env;
  if (!DB_HOST || !DB_USER || !DB_PASS || !DB_NAME) {
    throw new Error('Missing required database environment variables');
  }
  const port = DB_PORT ? parseInt(DB_PORT, 10) : 3306;
  return mysql.createConnection({
    host: DB_HOST,
    port,
    user: DB_USER,
    password: DB_PASS,
    database: DB_NAME,
    charset: 'utf8mb4',
  });
}

async function tableExists(connection: mysql.Connection, tableName: string): Promise<boolean> {
  const [rows] = await connection.execute<mysql.RowDataPacket[]>(
    `SELECT COUNT(*) as count FROM information_schema.tables
     WHERE table_schema = DATABASE() AND table_name = ?`,
    [tableName],
  );
  return Number(rows?.[0]?.count || 0) > 0;
}

async function getKitchenColumnNames(
  connection: mysql.Connection,
  tableName: string,
): Promise<{ hasAnalysisStage: boolean; hasAnalysisStatus: boolean }> {
  const [rows] = await connection.execute<mysql.RowDataPacket[]>(
    `SELECT COLUMN_NAME as column_name FROM information_schema.columns
     WHERE table_schema = DATABASE() AND table_name = ? AND column_name IN ('analysis_stage', 'analysis_status')`,
    [tableName],
  );
  const set = new Set((rows || []).map((row) => String(row.column_name || '')));
  return {
    hasAnalysisStage: set.has('analysis_stage'),
    hasAnalysisStatus: set.has('analysis_status'),
  };
}

async function loadActiveKitchenRows(
  connection: mysql.Connection,
  owner: string,
  excludedJobId: string,
): Promise<KitchenSwapCandidateRow[]> {
  const escapedOwner = owner.replace(/[^a-zA-Z0-9_-]/g, '');
  const params: string[] = [];
  let tableName: string;
  const filters: string[] = [];
  if (USE_SHARED_TABLES) {
    tableName = 'shared_kitchen';
    const memberIds = await getHouseholdMemberIds(connection, owner);
    const members = memberIds && memberIds.length ? memberIds : [escapedOwner];
    filters.push(`\`owner_id\` IN (${members.map(() => '?').join(',')})`);
    params.push(...members);
  } else {
    tableName = `${escapedOwner}_prod_kitchen`;
    if (!(await tableExists(connection, tableName))) {
      return [];
    }
  }

  const columns = await getKitchenColumnNames(connection, tableName);
  filters.push('`action` = \'IN\'', '(`job_id` IS NULL OR `job_id` <> ?)');
  params.push(excludedJobId);
  if (columns.hasAnalysisStage) {
    filters.push("(`analysis_stage` = 'final' OR `analysis_stage` IS NULL)");
  }
  if (columns.hasAnalysisStatus) {
    filters.push("(`analysis_status` = 'ready' OR `analysis_status` IS NULL)");
  }

  const [rows] = await connection.execute<mysql.RowDataPacket[]>(
    `SELECT \`_id\`, \`job_id\`, \`product_name\`, \`brand\`, \`variant\`, \`category\`, \`barcode\`, \`_createdDate\`
     FROM \`${tableName}\`
     WHERE ${filters.join(' AND ')}
     ORDER BY \`_createdDate\` DESC
     LIMIT 250`,
    params,
  );

  return (rows || []).map((row) => ({
    _id: String(row._id || ''),
    job_id: row.job_id == null ? null : String(row.job_id),
    product_name: cleanText(row.product_name) || null,
    brand: cleanText(row.brand) || null,
    variant: cleanText(row.variant) || null,
    category: cleanText(row.category) || null,
    barcode: cleanText(row.barcode) || null,
    _createdDate: row._createdDate || null,
  })).filter((row) => row._id);
}

function buildLocalSuggestionPayload(
  ranked: RankedKitchenSwapCandidate[],
  reasonSummary: string,
  generatedBy: SwapGeneratedBy = 'fast_local',
): SwapSuggestions {
  return {
    candidate_ids: ranked.map((row) => row._id),
    generated_by: generatedBy,
    generated_at: new Date().toISOString(),
    reason_summary: reasonSummary,
    score_metadata: ranked.map((row) => ({
      id: row._id,
      score: Number(row.score.toFixed(3)),
      product_name: row.product_name,
      brand: row.brand,
    })),
  };
}

export function rankLocalCandidates(item: GroceryItem, rows: KitchenSwapCandidateRow[]): RankedKitchenSwapCandidate[] {
  return rows
    .map((row) => ({ ...row, score: rankCandidate(item, row) }))
    .filter((row) => row.score >= LOCAL_MIN_SCORE)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return new Date(a._createdDate || 0).getTime() - new Date(b._createdDate || 0).getTime();
    })
    .slice(0, OPENAI_SHORTLIST_LIMIT);
}

function defaultReasonSummary(item: GroceryItem, count: number): string {
  if (count === 0) {
    return `No active kitchen rows looked similar enough to swap with ${item.product_name || 'this item'}.`;
  }
  return `Ranked ${count} active kitchen item(s) by product, brand, variant, and category similarity.`;
}

export async function suggestKitchenSwapsFast(params: {
  owner: string;
  jobId: string;
  groceryItem: GroceryItem;
}): Promise<SwapSuggestions> {
  const connection = await getConnection();
  try {
    const rows = await loadActiveKitchenRows(connection, params.owner, params.jobId);
    return buildFastSwapSuggestionsFromRows(params.groceryItem, rows);
  } finally {
    await connection.end();
  }
}

function parseJsonResponse<T>(response: any): T {
  const outputText = response?.output_text;
  if (typeof outputText !== 'string' || !outputText.trim()) {
    throw new Error('No structured output returned from OpenAI');
  }
  return JSON.parse(outputText) as T;
}

const deepSwapResponseJsonSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['candidate_ids', 'reason_summary'],
  properties: {
    candidate_ids: {
      type: 'array',
      items: { type: 'string' },
    },
    reason_summary: { type: 'string' },
  },
};

export function buildFastSwapSuggestionsFromRows(
  groceryItem: GroceryItem,
  rows: KitchenSwapCandidateRow[],
): SwapSuggestions {
  const ranked = rankLocalCandidates(groceryItem, rows).slice(0, MAX_LOCAL_CANDIDATES);
  return buildLocalSuggestionPayload(ranked, defaultReasonSummary(groceryItem, ranked.length));
}

export async function rerankDeepSwapSuggestionsFromRows(
  groceryItem: GroceryItem,
  rows: KitchenSwapCandidateRow[],
): Promise<SwapSuggestions> {
  const ranked = rankLocalCandidates(groceryItem, rows);
  const localFallback = buildLocalSuggestionPayload(
    ranked.slice(0, MAX_LOCAL_CANDIDATES),
    defaultReasonSummary(groceryItem, Math.min(ranked.length, MAX_LOCAL_CANDIDATES)),
  );
  if (ranked.length === 0) {
    return localFallback;
  }

  const shortlist = ranked.slice(0, OPENAI_SHORTLIST_LIMIT);
  const shortlistById = new Map(shortlist.map((candidate) => [candidate._id, candidate]));
  const itemSummary = [
    groceryItem.product_name && `Product: ${groceryItem.product_name}`,
    groceryItem.brand && `Brand: ${groceryItem.brand}`,
    groceryItem.variant && `Variant: ${groceryItem.variant}`,
    groceryItem.category && `Category: ${groceryItem.category}`,
    groceryItem.barcode && `Barcode: ${groceryItem.barcode}`,
    groceryItem.product_description && `Description: ${groceryItem.product_description}`,
  ].filter(Boolean).join('\n');

  const shortlistText = shortlist.map((candidate, index) => (
    `${index + 1}. _id="${candidate._id}" | product=${candidate.product_name || '(none)'} | brand=${candidate.brand || '(none)'} | variant=${candidate.variant || '(none)'} | category=${candidate.category || '(none)'} | barcode=${candidate.barcode || '(none)'} | local_score=${candidate.score.toFixed(3)}`
  )).join('\n');

  const openai = getClient();
  const response = await openai.responses.create({
    model: getOpenAIModel(),
    reasoning: { effort: 'low' },
    max_output_tokens: 300,
    text: {
      format: {
        type: 'json_schema',
        name: 'kitchen_swap_candidates',
        strict: true,
        schema: deepSwapResponseJsonSchema,
      },
    } as any,
    input: [
      {
        role: 'system',
        content: [{
          type: 'input_text',
          text: 'You rank which current kitchen items the user might reasonably swap out for a newly checked-in grocery item. Be fairly loose: include exact duplicates first, but also include same food-family items the user might plausibly treat as a swap even if the brand, flavor, cut, or style differs. Example: turkey should usually match other turkey products; yogurt can match similar yogurts; gum can match other gums. Prefer giving the user reasonable optional matches rather than being overly strict. Still exclude broad category matches that are not genuinely plausible swap targets.',
        }],
      },
      {
        role: 'user',
        content: [{
          type: 'input_text',
          text: `NEWLY CHECKED-IN ITEM:\n${itemSummary || 'No product details.'}\n\nACTIVE KITCHEN SHORTLIST:\n${shortlistText}\n\nReturn strict JSON with:\n- candidate_ids: ordered best-first list of kitchen _id values that are reasonably swappable with the new item\n- reason_summary: one short sentence\n\nLean loose rather than strict when the shortlist contains same-family items. If none are genuinely plausible, return an empty candidate_ids array.`,
        }],
      },
    ],
  } as any);

  const parsed = parseJsonResponse<{ candidate_ids?: string[]; reason_summary?: string }>(response);
  const candidateIds = Array.from(new Set(
    (Array.isArray(parsed.candidate_ids) ? parsed.candidate_ids : [])
      .map((id) => String(id || '').trim())
      .filter((id) => shortlistById.has(id)),
  )).slice(0, MAX_LOCAL_CANDIDATES);

  return {
    candidate_ids: candidateIds,
    generated_by: 'deep_openai',
    generated_at: new Date().toISOString(),
    reason_summary: cleanText(parsed.reason_summary) || `OpenAI reranked ${shortlist.length} local swap candidates.`,
    score_metadata: candidateIds.map((id) => {
      const candidate = shortlistById.get(id)!;
      return {
        id,
        score: Number(candidate.score.toFixed(3)),
        product_name: candidate.product_name,
        brand: candidate.brand,
      };
    }),
  };
}

export async function suggestKitchenSwapsDeep(params: {
  owner: string;
  jobId: string;
  groceryItem: GroceryItem;
}): Promise<SwapSuggestions> {
  const connection = await getConnection();
  try {
    const rows = await loadActiveKitchenRows(connection, params.owner, params.jobId);
    return await rerankDeepSwapSuggestionsFromRows(params.groceryItem, rows);
  } catch (error) {
    console.warn('[swap-suggestions] Deep rerank failed, falling back to local ranking:', error);
    return suggestKitchenSwapsFast(params);
  } finally {
    await connection.end();
  }
}
