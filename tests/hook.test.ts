import { describe, expect, it, vi } from 'vitest';
import type { On, SessionCompactInput, TurnCompleteInput } from 'claude-code';
import {
  compactSession,
  decisionLog,
  decisionLogLines,
  register as registerHook,
  resolveHookConfig,
  summarize,
  toSessionMessages,
  verdict,
} from '../hooks/fast-jev.ts';
import {
  applyDecisions,
  collectToolCalls,
  decideCall,
  reductionBound,
  reductionRatio,
  type Message,
} from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({ compactAtPercent: 60, floorPercent: 25, minReductionRatio: 0.25, model: 'jev-latest' });
    expect(
      resolveHookConfig({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g', compactAtPercent: 'no' }),
    ).toEqual({
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      compactAtPercent: 60,
      floorPercent: 25,
      minReductionRatio: 0.25,
    });
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k', model: 'jev-x' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1), bodies),
    );
    expect(bodies).toHaveLength(1);
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', undefined, 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(messages[1]?.text).toMatch(/^\[fast-jev-compaction boundary: this round removed 1 tool call \(Read 1\); earlier rounds removed 0 tool calls\./);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 kept, 1 call_dropped; state ~\d+ tokens \(full\) in 1 request\(s\)$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.10/result=0.10 t2:Bash:keep/call=0.90/result=0.90');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.10/result=0.10',
      'decisions (2/2): t2:Bash:drop_call/call=0.10/result=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('does not call low reduction a failure when the candidates could not have freed more', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const prose = message('assistant', 'ruling: we keep X, not Y, because Z. '.repeat(400), { handle: 'h-p' });
    const history = [transcript()[0]!, prose, ...transcript().slice(1)];
    const { result: output } = await compactSession(history, config, jevFetch(() => 0));
    expect(reductionRatio(output)).toBeLessThan(config.minReductionRatio);
    expect(verdict(output, config, undefined, 0)).toEqual({
      kind: 'nothing_to_prune',
      bound: reductionBound(output),
    });
    expect(verdict(output, config, { tokens: 200, window: 1000 }, 0)).toMatchObject({ kind: 'nothing_to_prune' });
    expect(verdict(output, config, { tokens: 900, window: 1000 }, 0)).toMatchObject({ kind: 'capacity' });
  });

  it('calls a real reduction scored, and Jev keeping everything scored too while the window has room', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const { result: dropped } = await compactSession(transcript(), config, jevFetch(() => 0));
    expect(verdict(dropped, config, { tokens: 200, window: 1000 }, 0)).toEqual({ kind: 'scored' });
    const { result: kept } = await compactSession(transcript(), config, jevFetch(() => 0.95));
    expect(reductionRatio(kept)).toBe(0);
    expect(verdict(kept, config, { tokens: 200, window: 1000 }, 0)).toEqual({ kind: 'scored' });
    expect(verdict(kept, config, { tokens: 700, window: 1000 }, 0)).toMatchObject({ kind: 'capacity', estimatedPercent: 70 });
  });

  it('falls back at the floor despite a 78% reduction, but scores when the estimate is below it', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    const { result: compacted } = await compactSession(transcript(), { ...config, apiKey: 'k' }, jevFetch(() => 0));
    const result = {
      ...compacted,
      stats: { ...compacted.stats, charsBefore: 1000, charsAfter: 220, candidateChars: 780 },
    };
    expect(reductionRatio(result)).toBe(0.78);
    expect(verdict(result, config, { tokens: 520_000, window: 1_000_000 }, 194_000)).toEqual({
      kind: 'capacity',
      estimatedPercent: 32.6,
    });
    expect(verdict(result, config, { tokens: 520_000, window: 1_000_000 }, 300_000)).toEqual({
      kind: 'scored',
    });
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });
});

describe('hook dispatch guards', () => {
  type HookHandler = (...args: any[]) => any;

  function handlers() {
    const registered = new Map<string, HookHandler>();
    const on = ((name: string, ...args: unknown[]) => {
      registered.set(name, args.at(-1) as HookHandler);
    }) as unknown as On;
    registerHook(on, { apiKey: 'k', preserveRecentMessages: 1 });
    return registered;
  }

  it('skips precompute without asking Jev or continuing to core', async () => {
    const hook = handlers().get('session.compact')!;
    const fetch = vi.fn();
    const next = vi.fn();
    const event = { trigger: 'precompute', messages: transcript() } as SessionCompactInput;
    const result = await hook({ http: { fetch } }, event, next);

    expect(result).toEqual({ skip: 'fast-jev-compaction: no precompute' });
    expect(fetch).not.toHaveBeenCalled();
    expect(next).not.toHaveBeenCalled();
  });

  it('leaves a fork transcript to core without asking Jev', async () => {
    const hook = handlers().get('session.compact')!;
    const fetch = vi.fn();
    const next = vi.fn(async () => ({ messages: [] }));
    const event = { trigger: 'manual', agentId: 'fork-1', messages: transcript() } as SessionCompactInput;

    await hook({ http: { fetch } }, event, next);

    expect(fetch).not.toHaveBeenCalled();
    expect(next).toHaveBeenCalledWith(event);
  });

  for (const trigger of ['manual', 'auto'] as const) {
    it(`still asks Jev for ${trigger} compaction`, async () => {
      const hook = handlers().get('session.compact')!;
      const bodies: string[] = [];
      const event = { trigger, messages: transcript() } as SessionCompactInput;
      const next = vi.fn(async () => ({ messages: event.messages }));
      await hook(
        {
          http: { fetch: jevFetch(() => 0.1, bodies) },
          session: { usage: async () => ({ context: { tokens: 0, window: 1_000_000 } }) },
          ui: { log: vi.fn(), toast: vi.fn() },
        },
        event,
        next,
      );

      expect(bodies).toHaveLength(1);
      expect(next).not.toHaveBeenCalled();
    });
  }

  it('does not auto-compact subagent or non-answer turns', async () => {
    const hook = handlers().get('turn.complete')!;
    const usage = vi.fn(async () => ({ context: { percent: 90 } }));
    const compact = vi.fn();
    const next = vi.fn(async () => ({ text: '' }));
    const engine = { session: { usage, compact } };
    const events = [
      { reason: 'answer', agentId: 'agent-1' },
      { reason: 'aborted' },
      { reason: 'refusal', refusal: { category: null, explanation: null } },
      { reason: 'error' },
    ] as TurnCompleteInput[];

    for (const event of events) await hook(engine, event, next);

    expect(usage).not.toHaveBeenCalled();
    expect(compact).not.toHaveBeenCalled();
    expect(next).toHaveBeenCalledTimes(events.length);
  });

  it('claims the auto-compact guard before awaiting usage', async () => {
    const hook = handlers().get('turn.complete')!;
    let releaseUsage!: (usage: { context: { percent: number } }) => void;
    const pendingUsage = new Promise<{ context: { percent: number } }>((resolve) => {
      releaseUsage = resolve;
    });
    const usage = vi.fn(() => pendingUsage);
    const compact = vi.fn(async () => undefined);
    const next = vi.fn(async () => ({ text: '' }));
    const engine = { session: { usage, compact } };
    const event = { reason: 'answer' } as TurnCompleteInput;

    const first = hook(engine, event, next);
    await hook(engine, event, next);
    releaseUsage({ context: { percent: 90 } });
    await first;

    expect(usage).toHaveBeenCalledTimes(1);
    expect(compact).toHaveBeenCalledTimes(1);
    expect(next).toHaveBeenCalledTimes(2);
  });
});
