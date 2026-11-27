// Fast drop-in for makefun-diff 1.0.0.
//
// Answers exactly as makefun-diff 1.0.0 does, but faster: the same Myers
// search the brief specifies (common prefix, then common suffix of what is
// left, then the greedy search from the first lines, keeping on each
// diagonal the path reaching furthest into the old text, the deletion
// winning when a deletion and an insertion reach the same point, never
// moving past either text's end, no pruning of edge diagonals, every run
// of equal lines followed as far as it goes; the first path to reach both
// ends is the script), with the same patch layout (header line, hunks with
// up to 3 context lines, deletions before insertions in each change run,
// a "%" line after a line that ends its text with no newline).
//
// Speed only, no output change: lines are interned once (a shared map from
// line string to a small integer id, so line equality is one integer
// compare, exact, with no hash-collision risk), the search runs on typed
// arrays with no per-state allocation, the trace is one typed array per
// depth whose edge slots hold a sentinel so no edge guard is needed in the
// hot loop, and the edit script is recovered from the trace instead of a
// linked list of path objects. Deterministic: no network, no randomness,
// no clock.

const FORMAT = 'makefun-diff/1';
const CONTEXT = 3;
const KEEP = 0;
const DEL = 1;
const INS = 2;
/** A stored old position no path reaches: fails both end checks below. */
const NONE = 1000000000;

/** One id per distinct line string, shared by both texts and all cases. */
const lineIds = new Map();

/**
 * A text's lines: each line's span in the text (end includes its "\n"
 * when it has one) and its interned id. The empty text has no lines;
 * "\n" has one, an empty line; a last line with no "\n" is a line too.
 */
function linesOf(text) {
  const starts = [];
  const ends = [];
  const ids = [];
  const len = text.length;
  let s = 0;
  while (s < len) {
    const nl = text.indexOf('\n', s);
    const e = nl === -1 ? len : nl + 1;
    const line = text.slice(s, e);
    let id = lineIds.get(line);
    if (id === undefined) {
      id = lineIds.size;
      lineIds.set(line, id);
    }
    starts.push(s);
    ends.push(e);
    ids.push(id);
    s = e;
  }
  return {
    starts: Int32Array.from(starts),
    ends: Int32Array.from(ends),
    ids: Int32Array.from(ids),
    count: starts.length,
  };
}

/**
 * The edit script of the middle slices as [kind, count] runs, in order.
 * A and B are the full texts' line-id arrays; the slices start at aOff
 * and bOff and run n2 and m2 lines. Runs use KEEP / DEL / INS above.
 */
function myersRuns(A, B, aOff, bOff, n2, m2) {
  const maxD = n2 + m2;
  // The trace, one Int32Array per depth d, length 2d + 5: diagonal k lives
  // at index k + d + 2 (valid 2..2d + 2), the four edge slots hold NONE so
  // a diagonal one past either edge reads as no path, with no guard.
  // The previous depth's trace is read directly, and each step's move is
  // recomputed from it while walking back.
  const traceX = [];
  let finishD = -1;
  let finishK = 0;

  // d = 0: the path with no edits, following equal lines from the start.
  let X = 0;
  let Y = 0;
  while (X < n2 && Y < m2 && A[aOff + X] === B[bOff + Y]) {
    X++;
    Y++;
  }
  {
    const xs = new Int32Array(5);
    xs[0] = NONE;
    xs[1] = NONE;
    xs[2] = X - 1;
    xs[3] = NONE;
    xs[4] = NONE;
    traceX.push(xs);
  }
  if (X >= n2 && Y >= m2) {
    finishD = 0;
    finishK = 0;
  }

  for (let d = 1; d <= maxD && finishD < 0; d++) {
    const prevXs = traceX[d - 1];
    const xs = new Int32Array(2 * d + 5);
    xs[0] = NONE;
    xs[1] = NONE;
    xs[2 * d + 3] = NONE;
    xs[2 * d + 4] = NONE;
    // j = 0..d, k = 2j - d. Diagonal k lives at xs[2j + 2]; its deletion
    // parent (diagonal k-1, depth d-1) at prevXs[2j], its insertion parent
    // (diagonal k+1, depth d-1) at prevXs[2j + 2].
    let pd = 0;
    let pi = 2;
    let ci = 2;
    let k = -d;
    for (let j = 0; j <= d; j++, pd += 2, pi += 2, ci += 2, k += 2) {
      const prevDel = prevXs[pd];
      const prevIns = prevXs[pi];
      // A move past the end of either text is no candidate; NONE fails both.
      const canRemove = prevDel + 1 < n2;
      const canAdd = prevIns - k < m2;
      if (!canRemove && !canAdd) {
        xs[ci] = NONE;
        continue;
      }
      // The candidate further into the old text, the deletion on a tie.
      const byDel = canRemove ? prevDel + 1 : -2;
      const byIns = canAdd ? prevIns : -2;
      let nx;
      if (byDel >= byIns) nx = prevDel + 2;
      else nx = prevIns + 1;
      let ny = nx - k;
      // Follow every run of equal lines in both texts as far as it goes.
      while (nx < n2 && ny < m2 && A[aOff + nx] === B[bOff + ny]) {
        nx++;
        ny++;
      }
      xs[ci] = nx - 1;
      if (nx >= n2 && ny >= m2) {
        finishD = d;
        finishK = k;
        break;
      }
    }
    traceX.push(xs);
  }
  if (finishD < 0) throw new Error('makefun-diff: the search ended without reaching both ends');

  // Walk the finishing path back to the start, recomputing each step's
  // move from the trace exactly as the forward pass chose it.
  const rev = [];
  let d = finishD;
  let k = finishK;
  while (d > 0) {
    const xEnd = traceX[d][k + d + 2];
    const prevXs = traceX[d - 1];
    const j = (k + d) >> 1;
    const prevDel = prevXs[2 * j];
    const prevIns = prevXs[2 * j + 2];
    const canRemove = prevDel + 1 < n2;
    const canAdd = prevIns - k < m2;
    const byDel = canRemove ? prevDel + 1 : -2;
    const byIns = canAdd ? prevIns : -2;
    const mv = byDel >= byIns ? DEL : INS;
    const parentX = mv === DEL ? prevDel : prevIns;
    const xMid = parentX + (mv === DEL ? 1 : 0);
    const snake = xEnd - xMid;
    if (snake > 0) rev.push([KEEP, snake]);
    rev.push([mv, 1]);
    k = mv === DEL ? k - 1 : k + 1;
    d--;
  }
  const x0 = traceX[0][2];
  if (x0 + 1 > 0) rev.push([KEEP, x0 + 1]);
  rev.reverse();
  const runs = [];
  for (const r of rev) {
    const last = runs.length === 0 ? undefined : runs[runs.length - 1];
    if (last !== undefined && last[0] === r[0]) last[1] += r[1];
    else runs.push([r[0], r[1]]);
  }
  return runs;
}

/** make.fun's own patch of input.old into input.new, as one string. */
export function diffPatch(input) {
  if (input === null || typeof input !== 'object' || typeof input.old !== 'string' || typeof input.new !== 'string') {
    throw new TypeError('makefun-diff: the input is { old, new }, two strings');
  }
  const oldText = input.old;
  const newText = input.new;
  const a = linesOf(oldText);
  const b = linesOf(newText);
  const N = a.count;
  const M = b.count;
  const A = a.ids;
  const B = b.ids;

  // The common prefix, then the common suffix of what is left.
  let prefix = 0;
  const minNM = N < M ? N : M;
  while (prefix < minNM && A[prefix] === B[prefix]) prefix++;
  let suffix = 0;
  while (suffix < N - prefix && suffix < M - prefix && A[N - 1 - suffix] === B[M - 1 - suffix]) suffix++;
  const n2 = N - prefix - suffix;
  const m2 = M - prefix - suffix;

  const mid = myersRuns(A, B, prefix, prefix, n2, m2);
  const runs = [];
  const pushRun = (kind, count) => {
    if (count <= 0) return;
    const last = runs.length === 0 ? undefined : runs[runs.length - 1];
    if (last !== undefined && last[0] === kind) last[1] += count;
    else runs.push([kind, count]);
  };
  pushRun(KEEP, prefix);
  for (const r of mid) pushRun(r[0], r[1]);
  pushRun(KEEP, suffix);

  // The change runs: each a stretch of deletions and insertions between
  // two kept lines, at old line o and new line p.
  const changes = [];
  let o = 0;
  let p = 0;
  let deleted = 0;
  let inserted = 0;
  for (const r of runs) {
    const kind = r[0];
    const count = r[1];
    if (kind === KEEP) {
      o += count;
      p += count;
      continue;
    }
    let c = changes.length === 0 ? undefined : changes[changes.length - 1];
    if (c === undefined || c.o + c.dels !== o || c.p + c.ins !== p) {
      c = { o, p, dels: 0, ins: 0 };
      changes.push(c);
    }
    if (kind === DEL) {
      c.dels += count;
      o += count;
      deleted += count;
    } else {
      c.ins += count;
      p += count;
      inserted += count;
    }
  }

  // The hunks: change runs at most 2 * CONTEXT kept lines apart share one.
  const hunks = [];
  for (const c of changes) {
    const h = hunks.length === 0 ? undefined : hunks[hunks.length - 1];
    const prev = h === undefined ? undefined : h[h.length - 1];
    if (prev !== undefined && c.o - (prev.o + prev.dels) <= 2 * CONTEXT) h.push(c);
    else hunks.push([c]);
  }

  // A line as the patch writes it: its op, the line, a newline, and "%"
  // on a line of its own when it had none.
  const aStarts = a.starts;
  const aEnds = a.ends;
  const bStarts = b.starts;
  const bEnds = b.ends;
  const written = (op, text, s, e) => {
    if (e > s && text.charCodeAt(e - 1) === 10) return op + text.slice(s, e);
    return op + text.slice(s, e) + '\n%\n';
  };

  const out = [`${FORMAT} old=${N} new=${M} hunks=${hunks.length} deleted=${deleted} inserted=${inserted}\n`];
  for (const h of hunks) {
    const first = h[0];
    const last = h[h.length - 1];
    const oldStart = Math.max(0, first.o - CONTEXT);
    const newStart = first.p - (first.o - oldStart);
    const after = Math.min(CONTEXT, N - (last.o + last.dels));
    const oldEnd = last.o + last.dels + after;
    const newEnd = last.p + last.ins + after;
    out.push(`@ ${oldStart + 1} ${oldEnd - oldStart} ${newStart + 1} ${newEnd - newStart}\n`);
    let at = oldStart;
    for (const c of h) {
      for (; at < c.o; at++) out.push(written('=', oldText, aStarts[at], aEnds[at]));
      for (let i = 0; i < c.dels; i++) out.push(written('-', oldText, aStarts[c.o + i], aEnds[c.o + i]));
      for (let i = 0; i < c.ins; i++) out.push(written('+', newText, bStarts[c.p + i], bEnds[c.p + i]));
      at = c.o + c.dels;
    }
    for (; at < oldEnd; at++) out.push(written('=', oldText, aStarts[at], aEnds[at]));
  }
  return out.join('');
}

/** makefun-diff 1.0.0's answer for one case's input. */
export function run(input) {
  return diffPatch(input);
}
