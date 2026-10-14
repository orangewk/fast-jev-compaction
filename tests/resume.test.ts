import { describe, expect, it } from 'vitest';
import { compactSession, resolveHookConfig } from '../hooks/fast-jev.ts';
import type { Message } from '../src/index.js';
import { HostModel, type HookMessage } from './host-model.ts';

function jevFetch(answer: (name: string) => number) {
  return async (_url: string, init?: { body?: string }) => {
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

const keepAll = () => 0.95;
const dropRead = (name: string) => (name.endsWith('_t1') ? 0.1 : 0.95);

function work(host: HostModel): void {
  host.prompt('Fix the failing test.');
  host.respond('Reading the file first.', [
    { tool: 'Read', input: { file_path: 'src/a.ts' }, result: 'export const a = 1;\n'.repeat(50) },
  ]);
  host.respond('Running both suites.', [
    { tool: 'Bash', input: { command: 'npm test' }, result: 'FAIL b.test.ts: expected 2 to be 3' },
    { tool: 'Bash', input: { command: 'npm run lint' }, result: 'ok' },
  ]);
  host.respond('b.test.ts expects 3; fixing it.', [
    { tool: 'Edit', input: { file_path: 'b.test.ts' }, result: 'updated' },
  ]);
  host.respond('Fixed: the suite passes.');
  host.prompt('thanks, what next?');
  host.respond('Lint is clean too; nothing else is failing.');
}

async function compact(host: HostModel, answer: (name: string) => number): Promise<HookMessage[]> {
  const config = { ...resolveHookConfig({}), apiKey: 'k' };
  const { messages } = await compactSession(host.messages(), config, jevFetch(answer));
  host.compact(messages);
  return messages;
}

function view(messages: readonly Message[]): string[] {
  return messages.map((message) =>
    JSON.stringify([
      message.role,
      message.text,
      message.toolUses.map((tool) => tool.tool_use_id),
      (message.toolResults ?? []).map((result) => result.tool_use_id),
    ]),
  );
}

const prompt = (text: string): Message => ({ role: 'user', text, toolUses: [] });

for (const dropBeforeLastBoundary of [true, false]) {
  const regime = dropBeforeLastBoundary ? 'from the last boundary on' : 'with the lines before it';
  describe(`resume after compaction, transcript read ${regime}`, () => {
    for (const [label, answer] of [['keeping every call', keepAll], ['dropping a call', dropRead]] as const) {
      it(`reads back every message the compaction kept, ${label}`, async () => {
        const host = new HostModel(dropBeforeLastBoundary);
        work(host);
        const kept = await compact(host, answer);
        host.prompt('continue');
        expect(view(host.resume())).toEqual(view([...kept, prompt('continue')]));
      });
    }

    it('leaves one copy of each kept message after two compactions (galaxias #1306)', async () => {
      const host = new HostModel(dropBeforeLastBoundary);
      work(host);
      await compact(host, keepAll);
      host.prompt('first');
      const kept = await compact(host, keepAll);
      host.prompt('second');
      const resumed = host.resume();
      expect(view(resumed)).toEqual(view([...kept, prompt('second')]));
      expect(resumed.filter((message) => message.text === 'Running both suites.')).toHaveLength(1);
    });
  });
}

describe('host model', () => {
  it('reproduces the cut resume when tool results keep their handles (911ae3f)', async () => {
    const host = new HostModel(true);
    work(host);
    const before = host.messages();
    host.compact(before.map((message) => (message.role === 'assistant' ? { ...message, handle: undefined } : message)));
    host.prompt('continue');
    expect(host.resume().length).toBeLessThan(before.length);
  });

  it('reproduces the copies when assistants keep their handles too (before 911ae3f)', async () => {
    const host = new HostModel(false);
    work(host);
    host.compact(host.messages());
    host.prompt('first');
    host.compact(host.messages());
    host.prompt('second');
    const copies = host.resume().filter((message) => message.text === 'Running both suites.');
    expect(copies.length).toBeGreaterThan(1);
  });
});
