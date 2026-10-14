import type { Message, ToolResult, ToolUse } from '../src/index.js';

/**
 * How Claude Code (2.1.293) writes a hook's compaction to the session
 * transcript and reads it back on resume, reduced to what decides which
 * messages a resume sees. Read off the bundled source:
 *
 * - `Qne().rowsOf` hands the hook one message per transcript line, its uuid as
 *   the handle. `messagesOf` gives back the line itself for a handle whose
 *   content is unchanged and builds a new line otherwise (`_mt`): an assistant
 *   line with a fresh message.id and timestamp, a user line without
 *   `sourceToolAssistantUUID`.
 * - `mQn` gives every returned line a fresh uuid and keeps the rest.
 * - `insertMessageChain` writes the boundary with no parent, then each line with
 *   the previous one as parent, except a user line whose
 *   `sourceToolAssistantUUID` the session already holds: its parent is that.
 * - On resume, `E1r` drops the lines before the last boundary when a boundary
 *   in the file lists preserved messages (otherwise the lines stay), `Rbt` walks
 *   parentUuid back from the last line, `A1r` stands in for a missing parent
 *   with the closest line at most 5 s older, and `P1r` adds every loaded
 *   assistant line sharing a message.id with one on the walk.
 * - A request merges the assistant lines sharing a message.id, past the
 *   tool-result lines between them, and joins consecutive user lines; then
 *   `ensureToolResultPairing` replaces each call the next user message does
 *   not answer with a "missing" result and strips the orphaned real one.
 */
type Line = {
  uuid: string;
  parentUuid: string | null;
  kind: 'user' | 'assistant' | 'boundary';
  timestamp: number;
  messageId?: string;
  sourceToolAssistantUUID?: string;
  text: string;
  toolUses: ToolUse[];
  toolResults?: ToolResult[];
};

export type HookMessage = Message & { handle?: string };

const FALLBACK_MS = 5000;

function content(message: Message): string {
  return JSON.stringify([message.role, message.text, message.toolUses, message.toolResults ?? []]);
}

export class HostModel {
  readonly lines: Line[] = [];
  /** The conversation the running session holds. */
  private live: Line[] = [];
  private ids = 0;
  private clock = 0;

  constructor(private readonly dropBeforeLastBoundary: boolean) {}

  private id(prefix: string): string {
    this.ids += 1;
    return `${prefix}-${this.ids}`;
  }

  private write(line: Omit<Line, 'uuid' | 'parentUuid' | 'timestamp'>, parentUuid: string | null, timestamp: number): Line {
    const written = { ...line, uuid: this.id('u'), parentUuid, timestamp };
    this.lines.push(written);
    return written;
  }

  private append(line: Omit<Line, 'uuid' | 'parentUuid' | 'timestamp'>, parentUuid?: string): Line {
    this.clock += 1000;
    const written = this.write(line, parentUuid ?? this.live.at(-1)?.uuid ?? null, this.clock);
    this.live.push(written);
    return written;
  }

  prompt(text: string): void {
    this.append({ kind: 'user', text, toolUses: [] });
  }

  /**
   * One API response, written one content block per line as the host does. A
   * `streamed` call is one the host ran while the response still streamed: its
   * result line comes right after its call, before the next call's line.
   */
  respond(
    text: string,
    calls: { tool: string; input: Record<string, unknown>; result: string; streamed?: boolean }[] = [],
  ): void {
    const messageId = this.id('msg');
    if (text) this.append({ kind: 'assistant', messageId, text, toolUses: [] });
    const answer = ({ row, tool_use_id, result }: { row: Line; tool_use_id: string; result: string }) =>
      this.append(
        { kind: 'user', sourceToolAssistantUUID: row.uuid, text: '', toolUses: [], toolResults: [{ tool_use_id, text: result, isError: false }] },
        row.uuid,
      );
    const later = calls.flatMap((call) => {
      const tool_use_id = this.id('toolu');
      const row = this.append({
        kind: 'assistant',
        messageId,
        text: '',
        toolUses: [{ tool_use_id, tool: call.tool, input: call.input, text: call.result }],
      });
      const pending = { row, tool_use_id, result: call.result };
      if (!call.streamed) return [pending];
      answer(pending);
      return [];
    });
    for (const pending of later) answer(pending);
  }

  messages(): HookMessage[] {
    return this.live.map((line) => this.toMessage(line, line.uuid));
  }

  private toMessage(line: Line, handle?: string): HookMessage {
    const message: HookMessage = { role: line.kind === 'assistant' ? 'assistant' : 'user', text: line.text, toolUses: line.toolUses };
    if (line.toolResults) message.toolResults = line.toolResults;
    if (handle) message.handle = handle;
    return message;
  }

  /** Installs what a `session.compact` hook handed back, as the host writes it. */
  compact(returned: readonly HookMessage[]): void {
    this.clock += 3_600_000;
    const own = new Map(this.live.map((line) => [line.uuid, line]));
    const known = new Set(this.lines.map((line) => line.uuid));
    let parent = this.write({ kind: 'boundary', text: '', toolUses: [] }, null, this.clock).uuid;
    this.live = returned.map((message) => {
      const line = message.handle ? own.get(message.handle) : undefined;
      const kept: Omit<Line, 'uuid' | 'parentUuid'> =
        line && content(this.toMessage(line)) === content(message)
          ? line
          : {
              kind: message.role,
              timestamp: this.clock,
              ...(message.role === 'assistant' && { messageId: this.id('msg') }),
              text: message.text,
              toolUses: message.toolUses,
              ...(message.toolResults && { toolResults: message.toolResults }),
            };
      const source = kept.kind === 'user' ? kept.sourceToolAssistantUUID : undefined;
      const written = this.write(kept, source && known.has(source) ? source : parent, kept.timestamp);
      parent = written.uuid;
      return written;
    });
  }

  /** The conversation a resume of the transcript reads. */
  resume(): Message[] {
    return this.resumed().map((step) => this.toMessage(step));
  }

  /**
   * The calls the pairing check finds unanswered in a request built from the
   * running session (`live`) or from a resume of the transcript.
   */
  unpaired(from: 'live' | 'resume'): string[] {
    type Turn = { role: 'user' | 'assistant'; messageId?: string; uses: string[]; results: string[]; text: string };
    const turns: Turn[] = [];
    for (const line of from === 'live' ? this.live : this.resumed()) {
      if (line.kind === 'user') {
        const last = turns.at(-1);
        const results = (line.toolResults ?? []).map((result) => result.tool_use_id);
        if (last?.role === 'user') {
          last.results.push(...results);
          last.text += line.text;
        } else turns.push({ role: 'user', uses: [], results, text: line.text });
        continue;
      }
      if (line.kind !== 'assistant') continue;
      const uses = line.toolUses.map((tool) => tool.tool_use_id);
      let same: Turn | undefined;
      for (let i = turns.length - 1; i >= 0; i -= 1) {
        const turn = turns[i]!;
        if (turn.role === 'user' && (turn.text !== '' || turn.results.length === 0)) break;
        if (turn.role === 'assistant' && turn.messageId === line.messageId) {
          same = turn;
          break;
        }
      }
      if (same) same.uses.push(...uses);
      else turns.push({ role: 'assistant', messageId: line.messageId, uses, results: [], text: line.text });
    }
    return turns.flatMap((turn, i) => {
      const next = turns[i + 1];
      return turn.role === 'assistant'
        ? turn.uses.filter((id) => next?.role !== 'user' || !next.results.includes(id))
        : [];
    });
  }

  private resumed(): Line[] {
    const last = this.lines.findLastIndex((line) => line.kind === 'boundary');
    const loaded = this.dropBeforeLastBoundary && last >= 0 ? this.lines.slice(last) : [...this.lines];
    const byUuid = new Map(loaded.map((line) => [line.uuid, line]));
    const seen = new Set<string>();
    const walk: Line[] = [];
    let line = loaded.findLast((candidate) => candidate.kind !== 'boundary');
    while (line && !seen.has(line.uuid)) {
      seen.add(line.uuid);
      walk.push(line);
      if (!line.parentUuid) break;
      const from: Line = line;
      line = byUuid.get(from.parentUuid!);
      if (!line || seen.has(line.uuid)) {
        line = loaded
          .filter((other) => !seen.has(other.uuid) && other.timestamp <= from.timestamp && from.timestamp - other.timestamp <= FALLBACK_MS)
          .sort((a, b) => b.timestamp - a.timestamp)[0];
      }
    }
    const conversation: Line[] = [];
    const add = (extra: Line) => {
      seen.add(extra.uuid);
      conversation.push(extra);
    };
    for (const step of walk.reverse()) {
      conversation.push(step);
      if (step.kind !== 'assistant') continue;
      const group = loaded.filter((other) => other.messageId === step.messageId);
      for (const sibling of group) if (!seen.has(sibling.uuid)) add(sibling);
      for (const member of group) {
        for (const child of loaded) {
          if (child.parentUuid === member.uuid && child.toolResults && !seen.has(child.uuid)) add(child);
        }
      }
    }
    return conversation.filter((step) => step.kind !== 'boundary');
  }
}
