import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionBound, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import { estimateTokens } from '../src/state.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  floorPercent: 25,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  compactAtPercent: number;
  floorPercent: number;
  /** Write the per-call decision lines to the transcript; off by default, as they run to thousands of characters. */
  logDecisions: boolean;
  minReductionRatio: number;
  model: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    floorPercent: optionNumber(options, 'floorPercent', HOOK_DEFAULTS.floorPercent),
    logDecisions: options['logDecisions'] === true,
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  if (options['keepUnscored'] === true) config.keepUnscored = true;
  return config;
}

export type Verdict =
  | { kind: 'scored' }
  | { kind: 'nothing_to_prune'; bound: number }
  | { kind: 'capacity'; estimatedPercent: number };

export type ContextUsage = { tokens?: number; window: number };

export function estimatedContextPercent(
  usage: ContextUsage | undefined,
  tokensSaved: number,
): number | undefined {
  if (
    !usage ||
    typeof usage.tokens !== 'number' ||
    !Number.isFinite(usage.tokens) ||
    !Number.isFinite(usage.window) ||
    usage.window <= 0
  ) return undefined;
  return 100 * Math.max(0, usage.tokens - Math.max(0, tokensSaved)) / usage.window;
}

/**
 * What the compaction outcome means. Reduction below `minReductionRatio` is
 * not a failure when the candidates could not have freed that much anyway
 * (`nothing_to_prune`); it is only a problem when the window would still be
 * over the compaction threshold afterwards (`capacity`), which is when the
 * built-in summary is worth its cost.
 */
export function verdict(
  result: CompactResult,
  config: Pick<HookConfig, 'minReductionRatio' | 'compactAtPercent' | 'floorPercent'>,
  usage: ContextUsage | undefined,
  tokensSaved: number,
): Verdict {
  const reduction = reductionRatio(result);
  const bound = reductionBound(result);
  const estimatedPercent = estimatedContextPercent(usage, tokensSaved);
  if (estimatedPercent !== undefined && estimatedPercent >= config.floorPercent) {
    return { kind: 'capacity', estimatedPercent };
  }
  if (reduction >= config.minReductionRatio) return { kind: 'scored' };
  if (bound < config.minReductionRatio) return { kind: 'nothing_to_prune', bound };
  return { kind: 'scored' };
}

function estimateMessageTokens(messages: readonly Message[]): number {
  return messages.reduce((total, message) => {
    total += estimateTokens(message.text);
    for (const tool of message.toolUses) {
      total += estimateTokens(tool.tool);
      total += estimateTokens(JSON.stringify(tool.input));
      total += estimateTokens(tool.text ?? '');
    }
    for (const result of message.toolResults ?? []) total += estimateTokens(result.text);
    return total;
  }, 0);
}

export function estimateTokensSaved(before: readonly Message[], after: readonly Message[]): number {
  return Math.max(0, estimateMessageTokens(before) - estimateMessageTokens(after));
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(fetchFn: HookFetch, apiKey: string, model: string): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * The API response each input message belongs to, as the index of its first
 * assistant message; -1 for a user message. The host hands one content block
 * per message and runs fast tools while the response still streams, so a
 * response is a run of assistant messages that tool-result messages break
 * only while one of its calls is still unanswered (a new response cannot
 * start before every call has its result).
 */
function responseStarts(input: readonly Message[]): number[] {
  let start = -1;
  let afterResults = false;
  const unanswered = new Set<string>();
  return input.map((message, index) => {
    if (message.role === 'user') {
      if (!message.toolResults?.length) start = -1;
      for (const result of message.toolResults ?? []) unanswered.delete(result.tool_use_id);
      afterResults = true;
      return -1;
    }
    if (start < 0 || (afterResults && unanswered.size === 0)) {
      start = index;
      unanswered.clear();
    }
    afterResults = false;
    for (const tool of message.toolUses) unanswered.add(tool.tool_use_id);
    return start;
  });
}

/**
 * The input index each output message came from, or -1 for one the library
 * added or a user message it rewrote. The library keeps the order and never
 * removes an assistant message with text, so a rebuilt message is found by
 * its first tool use or, with none left, as the next assistant with text.
 */
function sourceIndices(input: readonly Message[], output: readonly Message[]): number[] {
  const at = new Map<Message, number>(input.map((message, index) => [message, index]));
  const byCall = new Map<string, number>();
  input.forEach((message, index) => {
    for (const tool of message.toolUses) byCall.set(`use:${tool.tool_use_id}`, index);
    for (const result of message.toolResults ?? []) byCall.set(`result:${result.tool_use_id}`, index);
  });
  let next = 0;
  return output.map((message) => {
    let index = at.get(message);
    if (index === undefined) {
      const use = message.toolUses[0];
      const result = message.toolResults?.[0];
      if (use) index = byCall.get(`use:${use.tool_use_id}`);
      else if (result) index = byCall.get(`result:${result.tool_use_id}`);
      else if (message.role === 'assistant') {
        index = input.findIndex(
          (other, i) =>
            i >= next &&
            other.role === 'assistant' &&
            other.text.trim().length > 0 &&
            message.text.startsWith(other.text),
        );
      }
    }
    if (index === undefined || index < 0) return -1;
    next = index + 1;
    return index;
  });
}

/**
 * Maps the library's output back onto session messages. A user message that
 * came back unchanged and carries no tool results is the engine's own object,
 * handle included; every other message is rebuilt without a handle, so the
 * engine builds it from the content (tool uses and tool results inside stay the
 * engine's own objects where unchanged). The assistant messages kept from one
 * response are rebuilt as one, at the place of the first, so its calls stay
 * one message with their results after it.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  // ponytail: the host writes a rebuilt message as one line with a fresh message.id,
  // so the calls of one response rebuilt one by one become separate API messages, and
  // the host's pairing check replaces the results of all but the last with "missing"
  // (upstream #137). Rebuilding the response as one message keeps them paired.
  const starts = responseStarts(input);
  const response = sourceIndices(input, output).map((index, at) =>
    output[at]!.role === 'assistant' && index >= 0 ? starts[index]! : -1,
  );
  const members = new Map<number, Message[]>();
  output.forEach((message, at) => {
    if (response[at]! >= 0) members.set(response[at]!, [...(members.get(response[at]!) ?? []), message]);
  });
  const out: SessionMessage[] = [];
  output.forEach((message, at) => {
    const own = messages.get(message);
    // ponytail: on resume the host merges every transcript line sharing an assistant
    // message.id, across compact boundaries too, so a kept handle multiplies the
    // message (galaxias #1306). Rebuilding mints a fresh id; thinking blocks are lost.
    // Return the handle again once the host stops merging across boundaries.
    // A kept tool-result message keeps its sourceToolAssistantUUID, the uuid its call
    // had before this compaction, and the host writes that as its parentUuid; a resume
    // reads from the boundary on and loses everything before the result. Rebuilt, it
    // chains to the message before it; its attachments and non-text blocks are lost.
    if (own && own.role === 'user' && !own.toolResults?.length) {
      out.push(own);
      return;
    }
    const group = members.get(response[at]!) ?? [message];
    if (group[0] !== message) return;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: group.map((member) => member.text).filter((text) => text.length > 0).join('\n\n'),
      toolUses: group.flatMap((member) =>
        member.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
      ),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    out.push(rebuilt);
  });
  return out;
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** Runs the library over a session transcript; throws when the key is missing or Jev fails. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
): Promise<SessionCompaction> {
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  const result = await compact(messages, jevAsker(fetchFn, config.apiKey, config.model), config);
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
    stats.unscored > 0 ? `${stats.unscored} unscored` : '',
    stats.noticesTruncated > 0 ? `${stats.noticesTruncated} notices truncated` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

/** The last measured context tokens and model window, or undefined where usage is unavailable. */
async function contextUsage($: {
  session: { usage: () => Promise<{ context: ContextUsage }> };
}): Promise<ContextUsage | undefined> {
  try {
    const usage = await $.session.usage();
    return usage.context;
  } catch {
    return undefined;
  }
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

/**
 * The slash command a delivery consists of, or undefined. A slash command that
 * reaches a busy session from a Remote Control client is not run: the engine
 * hands it to the model as a queued message, and the sender sees nothing.
 */
export function bareSlashCommand(text: string): string | undefined {
  const match = /^\/([A-Za-z][\w:-]*)(?:[ \t][^\n]*)?$/.exec(text.trim());
  return match ? `/${match[1]}` : undefined;
}

/** What the model reads in place of a slash command that arrived mid-turn. */
export function busyCommandNotice(command: string): string {
  return (
    `[fast-jev-compaction: リモートから ${command} が届いたが、作業中だったため実行されていない。` +
    `コマンドは作業が止まっている間しか効かない。作業を続けたまま、送り主に次のように一言伝えること: ` +
    `「${command} は作業中だったので実行されていない。作業が止まってから送り直すか、先に止めてから送って」]`
  );
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;
  let turnRunning = false;

  on('turn.start', async (_$, event, next) => {
    turnRunning = true;
    return next(event);
  });

  on('session.receive', async (_$, event, next) => {
    const command = event.origin.kind === 'bridge' ? bareSlashCommand(event.text) : undefined;
    if (!command || !turnRunning) return next(event);
    return next({ ...event, text: busyCommandNotice(command) });
  });

  on('session.compact', async ($, event, next) => {
    if (event.agentId) return next(event);
    if (event.trigger === 'precompute') return { skip: 'fast-jev-compaction: no precompute' };
    try {
      const config = { ...configured, apiKey: await getApiKey($, configured) };
      const { result, messages } = await compactSession(event.messages, config, async (url, init) => {
        const response = await $.http.fetch(url, init);
        return { status: response.status, ok: response.ok, text: response.text };
      });
      if (config.logDecisions) for (const line of decisionLogLines(result)) $.ui.log(line);
      const outcome = verdict(
        result,
        config,
        await contextUsage($),
        estimateTokensSaved(event.messages, messages),
      );
      if (outcome.kind === 'capacity') {
        notify(
          $,
          `fallback to built-in summary (window would stay at ~${Math.round(outcome.estimatedPercent)}% after ${summarize(result)})`,
        );
        return next(event);
      }
      const note =
        outcome.kind === 'nothing_to_prune'
          ? `; candidates were ${percent(outcome.bound)} of the history, the rest is text`
          : '';
      notify(
        $,
        `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)}${note})`,
      );
      return { messages };
    } catch (error) {
      notify(
        $,
        `fallback to built-in summary (${error instanceof Error ? error.message : String(error)})`,
      );
      return next(event);
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (!event.agentId) turnRunning = false;
    if (compacting || event.agentId || event.reason !== 'answer') return next(event);
    compacting = true;
    try {
      const { context } = await $.session.usage();
      if ((context.percent ?? 0) >= configured.compactAtPercent) await $.session.compact();
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
