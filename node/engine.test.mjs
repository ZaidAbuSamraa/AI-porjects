import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { runWorkflow } from './engine.mjs';

const lesson = {source_id: 'L1', title: 'Greetings', category: 'Conversations',
  level: 'A1', language: 'en', body: 'Hello!'};
async function sandbox(fn) {
  const d = await mkdtemp(path.join(tmpdir(), 'flow-lab-'));
  try { await fn(path.join(d, 'store.json')); } finally { await rm(d, {recursive: true, force: true}); }
}
const readStore = async store => JSON.parse(await readFile(store, 'utf8'));

test('approval required before saving', () => sandbox(async store => {
  const r = await runWorkflow([lesson], { store });
  assert.equal(r.items[0].status, 'review_pending');
  assert.equal(r.mode, 'review');
  assert.equal(existsSync(store), false);
}));
test('blank reviewer is not approval', () => sandbox(async store => {
  const r = await runWorkflow([lesson], { store, approvedBy: '   ' });
  assert.equal(r.items[0].status, 'review_pending');
  assert.equal(r.mode, 'review');
  assert.equal(existsSync(store), false);
}));
test('duplicate is skipped and only draft is saved', () => sandbox(async store => {
  assert.equal((await runWorkflow([lesson], {store, approvedBy: 'Trainer'})).items[0].status, 'draft_saved');
  assert.equal((await runWorkflow([lesson], {store, approvedBy: 'Trainer'})).items[0].status, 'skipped_duplicate');
  const saved = await readStore(store);
  assert.equal(saved.length, 1);
  assert.equal(saved[0].status, 'draft');
}));
test('invalid rejected', () => sandbox(async store => {
  const r = await runWorkflow([{...lesson, title: ''}], {store, approvedBy: 'Trainer'});
  assert.equal(r.items[0].status, 'failed_validation');
  assert.match(r.items[0].error, /title/);
  assert.deepEqual(await readStore(store), []);
}));
test('invalid CEFR level rejected', () => sandbox(async store => {
  const r = await runWorkflow([{...lesson, level: 'A9'}], {store, approvedBy: 'Trainer'});
  assert.equal(r.items[0].status, 'failed_validation');
  assert.equal(r.items[0].error, 'Invalid CEFR level');
  assert.deepEqual(await readStore(store), []);
}));
test('invalid item does not block valid items', () => sandbox(async store => {
  const bad = {...lesson, source_id: 'L2', title: ''};
  const r = await runWorkflow([bad, lesson], {store, approvedBy: 'Trainer'});
  assert.deepEqual(r.summary, {draft_saved: 1, failed_validation: 1});
  assert.deepEqual((await readStore(store)).map(x => x.source_id), ['L1']);
}));
test('dry run previews', () => sandbox(async store => {
  const r = await runWorkflow([lesson], {store, approvedBy: 'Trainer', dryRun: true});
  assert.equal(r.items[0].status, 'dry_run_preview');
  assert.equal(existsSync(store), false);
}));
test('dry run leaves existing store unchanged', () => sandbox(async store => {
  await runWorkflow([lesson], {store, approvedBy: 'Trainer'});
  const before = await readFile(store, 'utf8');
  const r = await runWorkflow([{...lesson, source_id: 'L2', title: 'Farewells'}],
    {store, approvedBy: 'Trainer', dryRun: true});
  assert.equal(r.items[0].status, 'dry_run_preview');
  assert.equal(await readFile(store, 'utf8'), before);
}));
test('word_count in report and draft', () => sandbox(async store => {
  const r = await runWorkflow([{...lesson, body: '  Hello!  My name   is Sara.  '}],
    {store, approvedBy: 'Trainer'});
  assert.equal(r.items[0].word_count, 5);
  assert.ok(r.items[0].steps.includes('count_words'));
  assert.equal((await readStore(store))[0].word_count, 5);
}));
test('word_count shown in dry-run preview', () => sandbox(async store => {
  const r = await runWorkflow([lesson], {store, approvedBy: 'Trainer', dryRun: true});
  assert.equal(r.items[0].word_count, 1);
}));
