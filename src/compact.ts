import { noulAnswer } from './request.js';
import { collectToolCalls, estimateTokens, fitState, isMachineText, isPinned } from './state.js';
import type {
  CallAnswer,
  CallDecision,
  CompactOptions,
  CompactResult,
  CompactionState,
  JevAsker,
  JevQuestions,
  Message,
  ResolvedCompactOptions,
  ToolCall,
  ToolUse,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  keepThreshold: 0.5,
  preserveRecentMessages: 6,
  maxStateTokens: 25_000,
  maxRequestTokens: 30_000,
  truncateHeadChars: 300,
  keepUnscored: false,
};

/** Tokens the request envelope (`model`, key names) adds around state and questions. */
const REQUEST_OVERHEAD_TOKENS = 20;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    keepThreshold: finite(options.keepThreshold, DEFAULT_OPTIONS.keepThreshold),
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    maxStateTokens: Math.max(1, finite(options.maxStateTokens, DEFAULT_OPTIONS.maxStateTokens)),
    maxRequestTokens: Math.max(
      1,
      finite(options.maxRequestTokens, DEFAULT_OPTIONS.maxRequestTokens),
    ),
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
    keepUnscored: options.keepUnscored ?? DEFAULT_OPTIONS.keepUnscored,
  };
}

/**
 * The `noul` questions asked about one call: keep the call, keep its result.
 * A result an earlier round already cut to a head is not asked about again;
 * there is no full output left to keep.
 */
export function questionsFor(call: ToolCall): JevQuestions {
  const questions: JevQuestions = {
    [`call_${call.id}`]: {
      type: 'noul',
      instructions: `Tool call ${call.id} (${call.tool}) should stay in the history: knowing this call was made, with its input, still matters for what the assistant does next`,
    },
  };
  if (call.originalChars === undefined) {
    questions[`result_${call.id}`] = {
      type: 'noul',
      instructions: `The full output of tool call ${call.id} (${call.tool}, ${call.resultChars} chars) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do`,
    };
  }
  return questions;
}

/**
 * Splits the candidate calls into batches whose questions, together with the
 * (always complete) state, fit one request.
 */
export function batchCalls(
  calls: readonly ToolCall[],
  stateTokens: number,
  options: Pick<ResolvedCompactOptions, 'maxRequestTokens'>,
): ToolCall[][] {
  const budget = options.maxRequestTokens - stateTokens - REQUEST_OVERHEAD_TOKENS;
  const batches: ToolCall[][] = [];
  let current: ToolCall[] = [];
  let currentTokens = 0;
  for (const call of calls) {
    const tokens = estimateTokens(JSON.stringify(questionsFor(call)));
    if (current.length > 0 && currentTokens + tokens > budget) {
      batches.push(current);
      current = [];
      currentTokens = 0;
    }
    if (current.length === 0 && tokens > budget) {
      throw new Error(
        `state leaves no room for questions (~${stateTokens} of ${options.maxRequestTokens} tokens)`,
      );
    }
    current.push(call);
    currentTokens += tokens;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

export function decideCall(
  call: Pick<ToolCall, 'id' | 'tool' | 'pinned'>,
  answer: CallAnswer,
  options: Pick<ResolvedCompactOptions, 'keepThreshold'>,
): CallDecision {
  const base = { id: call.id, tool: call.tool, ...answer };
  if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
  if (answer.keepResult >= options.keepThreshold) {
    return { ...base, action: 'keep', reason: 'kept' };
  }
  if (answer.keepCall >= options.keepThreshold) {
    return { ...base, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, action: 'drop_call', reason: 'call_dropped' };
}

async function askBatch(
  asker: JevAsker,
  state: CompactionState,
  batch: readonly ToolCall[],
): Promise<Map<string, CallAnswer>> {
  const questions: JevQuestions = Object.assign({}, ...batch.map(questionsFor));
  const { answers } = await asker.ask(state, questions);
  return new Map(
    batch.map((call) => [
      call.id,
      {
        keepCall: noulAnswer(answers, `call_${call.id}`),
        keepResult:
          call.originalChars === undefined ? noulAnswer(answers, `result_${call.id}`) : 0,
      },
    ]),
  );
}

function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-jev-compaction truncated ${text.length - headChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
}


const NOTICE_NOTE_START = '[fast-jev-compaction truncated ';
const NOTICE_NOTE_END = ' chars of this host notice; the full text is in the session transcript]';

/**
 * Host notices (task notifications, command echoes, reminders) are written
 * under the user role but are not the user's words, and they are re-derivable
 * from what the assistant did next. Outside the pinned messages they keep a
 * head and a note, by rule; no question is asked. A previous compaction's
 * summary is machine text too, but it is the only copy of what it summarised,
 * so it stays whole.
 */
export function truncateHostNotices(
  messages: readonly Message[],
  options: Pick<ResolvedCompactOptions, 'preserveRecentMessages' | 'truncateHeadChars'>,
): { messages: Message[]; truncated: number; charsCut: number } {
  let truncated = 0;
  let charsCut = 0;
  const headChars = options.truncateHeadChars;
  const out = messages.map((message, index) => {
    if (message.role !== 'user' || (message.toolResults ?? []).length > 0) return message;
    if (isPinned(index, messages.length, options.preserveRecentMessages)) return message;
    const start = message.text.trimStart();
    if (
      !isMachineText(message.text) ||
      start.startsWith('This session is being continued') ||
      start.startsWith(BOUNDARY_PREFIX)
    ) {
      return message;
    }
    if (message.text.length <= headChars + 120 || message.text.trimEnd().endsWith(NOTICE_NOTE_END)) {
      return message;
    }
    const cut = message.text.length - headChars;
    truncated += 1;
    charsCut += cut;
    const head = headChars > 0 ? `${message.text.slice(0, headChars)}\n` : '';
    return {
      role: message.role,
      text: `${head}${NOTICE_NOTE_START}${cut}${NOTICE_NOTE_END}`,
      toolUses: message.toolUses,
    };
  });
  return { messages: out, truncated, charsCut };
}

const LABEL_INPUT_CHARS = 60;
const LABELS_PER_NOTE = 3;

/** `Bash(gh pr create --title …)`: the tool and the gist of its input. */
function callLabel(tool: Pick<ToolUse, 'tool' | 'input'>): string {
  const first = Object.values(tool.input).find((value) => typeof value === 'string') as
    | string
    | undefined;
  let gist = first ?? '';
  if (!gist) {
    try {
      gist = JSON.stringify(tool.input);
    } catch {
      gist = '';
    }
  }
  gist = gist.replace(/\s+/g, ' ').trim();
  if (gist.length > LABEL_INPUT_CHARS) gist = `${gist.slice(0, LABEL_INPUT_CHARS - 1)}…`;
  return `${tool.tool}(${gist})`;
}

/**
 * Note appended to an assistant turn whose narration outlived its tool
 * calls. It names the calls so the receipt survives, and says what the text
 * now is: a report, not the record.
 */
function removedCallsNote(removed: readonly Pick<ToolUse, 'tool' | 'input'>[]): string {
  const count = removed.length;
  const labels = removed.slice(0, LABELS_PER_NOTE).map(callLabel);
  if (count > LABELS_PER_NOTE) labels.push(`+${count - LABELS_PER_NOTE} more`);
  return `[fast-jev-compaction removed ${count} tool call${count === 1 ? '' : 's'} from this turn (${labels.join(', ')}); this text is the assistant's report, not the tool records — re-verify with a tool before treating it as current]`;
}

export const BOUNDARY_PREFIX = '[fast-jev-compaction boundary:';
const BOUNDARY_EARLIER = /earlier rounds removed (\d+) tool calls?/;

function countByTool(decisions: readonly CallDecision[]): string {
  const counts = new Map<string, number>();
  for (const decision of decisions) {
    if (decision.action === 'drop_call') counts.set(decision.tool, (counts.get(decision.tool) ?? 0) + 1);
  }
  const top = [...counts].sort((a, b) => b[1] - a[1]);
  const shown = top.slice(0, 4).map(([tool, n]) => `${tool} ${n}`);
  if (top.length > 4) shown.push('…');
  return shown.join(', ');
}

/**
 * One message per compaction, right after the pinned first message, saying
 * what this round removed and how much earlier rounds removed before it, so
 * the assistant reads the survivors as survivors. A boundary from an earlier
 * round is replaced, its count folded into the total.
 */
export function withBoundary(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  stats: { resultsDropped: number; noticesTruncated: number },
): Message[] {
  const dropped = decisions.filter((decision) => decision.action === 'drop_call').length;
  let earlier = 0;
  const rest = messages.filter((message) => {
    if (message.role !== 'user' || !message.text.startsWith(BOUNDARY_PREFIX)) return true;
    const own = /this round removed (\d+) tool calls?/.exec(message.text);
    const before = BOUNDARY_EARLIER.exec(message.text);
    earlier += Number(own?.[1] ?? 0) + Number(before?.[1] ?? 0);
    return false;
  });
  if (dropped === 0 && stats.resultsDropped === 0 && stats.noticesTruncated === 0 && earlier === 0) {
    return [...messages];
  }
  const removed = dropped > 0 ? ` (${countByTool(decisions)})` : '';
  const parts = [
    `this round removed ${dropped} tool call${dropped === 1 ? '' : 's'}${removed}`,
    stats.resultsDropped > 0 ? `cut ${stats.resultsDropped} tool result${stats.resultsDropped === 1 ? '' : 's'} to a head` : '',
    stats.noticesTruncated > 0 ? `cut ${stats.noticesTruncated} host notice${stats.noticesTruncated === 1 ? '' : 's'} to a head` : '',
  ].filter(Boolean);
  const text = `${BOUNDARY_PREFIX} ${parts.join(', ')}; earlier rounds removed ${earlier} tool call${earlier === 1 ? '' : 's'}. Assistant text above that reports work done through removed calls carries a "[fast-jev-compaction removed …]" note: it is the assistant's report, not the tool record. Re-verify with a tool before treating such state as current.]`;
  const boundary: Message = { role: 'user', text, toolUses: [] };
  if (rest.length === 0) return [boundary];
  return [rest[0]!, boundary, ...rest.slice(1)];
}

/**
 * The message that narrates the turn a tool call belongs to: the message
 * itself when it has text, else the nearest earlier assistant message with
 * text before any user prompt (a host may hand one turn as a text message
 * followed by tool-use-only messages and their result carriers).
 */
function narrationIndex(messages: readonly Message[], index: number): number {
  for (let i = index; i >= 0; i -= 1) {
    const message = messages[i]!;
    if (message.role === 'user') {
      if (message.text.trim().length > 0 || !(message.toolResults?.length)) return -1;
      continue;
    }
    if (message.text.trim().length > 0) return i;
  }
  return -1;
}

/**
 * Rebuilds the conversation from the decisions. A dropped call disappears
 * together with its result and leaves a note on the turn's narration so it is
 * not mistaken for a tool-free report; a dropped result keeps a bounded head
 * and note. Messages that lose all their content are removed; untouched
 * messages are returned as the same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const actions = new Map<string, CallDecision['action']>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && decision.action !== 'keep') actions.set(call.tool_use_id, decision.action);
  }
  const kept: (Message | null)[] = [];
  const removedCalls = new Map<number, ToolUse[]>();
  messages.forEach((message, index) => {
    const dropped = message.toolUses.filter(
      (tool) => actions.get(tool.tool_use_id) === 'drop_call',
    );
    if (dropped.length > 0) {
      const at = narrationIndex(messages, index);
      if (at >= 0) removedCalls.set(at, [...(removedCalls.get(at) ?? []), ...dropped]);
    }
  });
  for (const message of messages) {
    const touched =
      message.toolUses.some((tool) => actions.has(tool.tool_use_id)) ||
      (message.toolResults ?? []).some((result) => actions.has(result.tool_use_id));
    if (!touched) {
      kept.push(message);
      continue;
    }
    const toolUses = message.toolUses
      .filter((tool) => actions.get(tool.tool_use_id) !== 'drop_call')
      .map((tool) => {
        if (actions.get(tool.tool_use_id) !== 'drop_result') return tool;
        const text = truncatedResultText(
          tool.text ?? '',
          tool.isError ?? false,
          headChars,
        );
        if ((tool.text ?? '') === text) return tool;
        const copy: ToolUse = {
          tool_use_id: tool.tool_use_id,
          tool: tool.tool,
          input: tool.input,
          text,
        };
        if (tool.isError) copy.isError = true;
        return copy;
      });
    const toolResults = (message.toolResults ?? [])
      .filter((result) => actions.get(result.tool_use_id) !== 'drop_call')
      .map((result) => {
        if (actions.get(result.tool_use_id) !== 'drop_result') return result;
        const text = truncatedResultText(result.text, result.isError ?? false, headChars);
        return text === result.text
          ? result
          : {
              tool_use_id: result.tool_use_id,
              text,
              isError: result.isError,
            };
      });
    if (
      !message.toolUses.some(
        (tool) => actions.get(tool.tool_use_id) === 'drop_call',
      ) &&
      !(message.toolResults ?? []).some(
        (result) => actions.get(result.tool_use_id) === 'drop_call',
      ) &&
      toolUses.every((tool, index) => tool === message.toolUses[index]) &&
      toolResults.every(
        (result, index) => result === message.toolResults?.[index],
      )
    ) {
      kept.push(message);
      continue;
    }
    if (message.text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
      kept.push(null);
      continue;
    }
    const rebuilt: Message = { role: message.role, text: message.text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  }
  for (const [index, removed] of removedCalls) {
    const message = kept[index];
    if (!message) continue;
    const noted: Message = {
      role: message.role,
      text: `${message.text}\n${removedCallsNote(removed)}`,
      toolUses: message.toolUses,
    };
    if (message.toolResults) noted.toolResults = message.toolResults;
    kept[index] = noted;
  }
  return kept.filter((message): message is Message => message !== null);
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

/** The reduction dropping every candidate would give: the most a round can free. */
export function reductionBound(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, candidateChars } = result.stats;
  return charsBefore === 0 ? 0 : candidateChars / charsBefore;
}

function inputChars(call: Pick<ToolCall, 'input'>): number {
  try {
    return JSON.stringify(call.input).length;
  } catch {
    return 20;
  }
}

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

/**
 * Compacts a transcript by asking Jev, for every tool call outside the pinned
 * first and newest messages, whether the call and whether its result must
 * stay. The whole history (results omitted, fitted into `maxStateTokens`) is
 * sent as state with every batch of questions. Throws when Jev fails or the
 * history cannot be fitted; the caller decides whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  asker: JevAsker,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const candidates = calls.filter((call) => !call.pinned);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  let fitted: { tokens: number; stage: string; visible: Set<string> } = {
    tokens: 0,
    stage: '',
    visible: new Set(),
  };
  let batches: ToolCall[][] = [];
  const answers = new Map<string, CallAnswer>();
  if (candidates.length > 0) {
    const state = fitState(messages, calls, resolved);
    fitted = state;
    batches = batchCalls(candidates, state.tokens, resolved);
    const answered = await Promise.all(
      batches.map((batch) => askBatch(asker, state.state, batch)),
    );
    for (const map of answered) for (const [id, answer] of map) answers.set(id, answer);
  }

  const unscored = candidates.filter((call) => !fitted.visible.has(call.id));
  const decisions = calls.map((call) => {
    const answer = answers.get(call.id) ?? { keepCall: 1, keepResult: 1 };
    if (resolved.keepUnscored && !call.pinned && !fitted.visible.has(call.id)) {
      return { id: call.id, tool: call.tool, ...answer, action: 'keep' as const, reason: 'unscored' as const };
    }
    return decideCall(call, answer, resolved);
  });
  const notices = truncateHostNotices(
    applyDecisions(messages, decisions, calls, resolved.truncateHeadChars),
    resolved,
  );
  const kept = withBoundary(notices.messages, decisions, {
    resultsDropped: count(decisions, 'result_dropped'),
    noticesTruncated: notices.truncated,
  });
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      kept: count(decisions, 'kept'),
      resultsDropped: count(decisions, 'result_dropped'),
      callsDropped: count(decisions, 'call_dropped'),
      pinned: count(decisions, 'pinned'),
      candidateChars:
        candidates.reduce((sum, call) => sum + inputChars(call) + call.resultChars, 0) +
        notices.charsCut,
      unscored: unscored.length,
      noticesTruncated: notices.truncated,
      stateTokens: fitted.tokens,
      stateStage: fitted.stage,
      requests: batches.length,
      ms: Date.now() - started,
    },
  };
}
