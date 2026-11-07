import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ACCESS_CODE_HEADER,
  ACCESS_CODE_STORAGE_KEY,
  AccessCodeError,
  accessCodeHeaders,
  clearCode,
  gradeError,
  isValidCode,
  loadCode,
  saveCode,
  withAccessCode,
  type CodeStorage,
} from "./accessCode.ts";

const memoryStorage = (initial: Record<string, string> = {}) => {
  const items = new Map(Object.entries(initial));
  const storage: CodeStorage = {
    getItem: (k) => items.get(k) ?? null,
    setItem: (k, v) => void items.set(k, v),
    removeItem: (k) => void items.delete(k),
  };
  return { storage, items };
};

// localStorage can throw on every access (blocked site data, some private modes).
const throwingStorage: CodeStorage = {
  getItem: () => {
    throw new Error("SecurityError");
  },
  setItem: () => {
    throw new Error("QuotaExceededError");
  },
  removeItem: () => {
    throw new Error("SecurityError");
  },
};

// A fake /grade: 401 unless it gets `valid`; records the code of each call.
const fakeGrade = (valid: string) => {
  const calls: (string | null)[] = [];
  const run = async (code: string | null) => {
    calls.push(code);
    if (code !== valid) throw new AccessCodeError();
    return { grade_int: 9 };
  };
  return { run, calls };
};

test("the header is sent only when there is a code", () => {
  assert.deepEqual(accessCodeHeaders(null), {});
  assert.deepEqual(accessCodeHeaders(""), {});
  assert.deepEqual(accessCodeHeaders("abc"), { "X-Access-Code": "abc" });
  assert.equal(ACCESS_CODE_HEADER, "X-Access-Code");
});

test("storage helpers save, load and clear the code", () => {
  const { storage, items } = memoryStorage();
  assert.equal(loadCode(storage), null);
  saveCode(storage, "abc");
  assert.equal(items.get(ACCESS_CODE_STORAGE_KEY), "abc");
  assert.equal(loadCode(storage), "abc");
  clearCode(storage);
  assert.equal(loadCode(storage), null);
});

test("storage helpers never throw, and work with no storage", () => {
  assert.equal(loadCode(throwingStorage), null);
  assert.doesNotThrow(() => saveCode(throwingStorage, "abc"));
  assert.doesNotThrow(() => clearCode(throwingStorage));
  assert.equal(loadCode(null), null);
  assert.doesNotThrow(() => saveCode(null, "abc"));
  assert.doesNotThrow(() => clearCode(null));
});

test("an open API grades without asking for a code", async () => {
  const { storage } = memoryStorage();
  const calls: (string | null)[] = [];
  const out = await withAccessCode(async (code) => {
    calls.push(code);
    return 42;
  }, storage);
  assert.deepEqual(out, { kind: "ok", value: 42 });
  assert.deepEqual(calls, [null]);
});

test("a 401 with no code asks for one, not as a wrong code", async () => {
  const { storage } = memoryStorage();
  const grade = fakeGrade("right");
  const out = await withAccessCode(grade.run, storage);
  assert.deepEqual(out, { kind: "needs-code", wrongCode: false });
  assert.deepEqual(grade.calls, [null]);
});

test("an entered code is sent, retried once and remembered", async () => {
  const { storage, items } = memoryStorage();
  const grade = fakeGrade("right");
  const out = await withAccessCode(grade.run, storage, "  right ");
  assert.deepEqual(out, { kind: "ok", value: { grade_int: 9 } });
  assert.deepEqual(grade.calls, ["right"]);
  assert.equal(items.get(ACCESS_CODE_STORAGE_KEY), "right");

  // The next grade uses the stored code without asking.
  const again = await withAccessCode(grade.run, storage);
  assert.equal(again.kind, "ok");
  assert.deepEqual(grade.calls, ["right", "right"]);
});

test("a wrong entered code is reported and cleared", async () => {
  const { storage, items } = memoryStorage();
  const grade = fakeGrade("right");
  const out = await withAccessCode(grade.run, storage, "wrong");
  assert.deepEqual(out, { kind: "needs-code", wrongCode: true });
  assert.deepEqual(grade.calls, ["wrong"]);
  assert.equal(items.has(ACCESS_CODE_STORAGE_KEY), false);
});

test("a stored code that stops working is cleared", async () => {
  const { storage, items } = memoryStorage({ [ACCESS_CODE_STORAGE_KEY]: "old" });
  const grade = fakeGrade("new");
  const out = await withAccessCode(grade.run, storage);
  assert.deepEqual(out, { kind: "needs-code", wrongCode: true });
  assert.deepEqual(grade.calls, ["old"]);
  assert.equal(items.has(ACCESS_CODE_STORAGE_KEY), false);
});

test("an entered code replaces a stored one", async () => {
  const { storage, items } = memoryStorage({ [ACCESS_CODE_STORAGE_KEY]: "old" });
  const grade = fakeGrade("new");
  const out = await withAccessCode(grade.run, storage, "new");
  assert.equal(out.kind, "ok");
  assert.deepEqual(grade.calls, ["new"]);
  assert.equal(items.get(ACCESS_CODE_STORAGE_KEY), "new");
});

test("a blank entered code falls back to the stored one", async () => {
  const { storage } = memoryStorage({ [ACCESS_CODE_STORAGE_KEY]: "right" });
  const grade = fakeGrade("right");
  const out = await withAccessCode(grade.run, storage, "   ");
  assert.equal(out.kind, "ok");
  assert.deepEqual(grade.calls, ["right"]);
});

test("a code still works when storage throws", async () => {
  const grade = fakeGrade("right");
  const out = await withAccessCode(grade.run, throwingStorage, "right");
  assert.equal(out.kind, "ok");
  const wrong = await withAccessCode(grade.run, throwingStorage, "wrong");
  assert.deepEqual(wrong, { kind: "needs-code", wrongCode: true });
});

test("other errors are not taken for a missing code", async () => {
  const { storage, items } = memoryStorage({ [ACCESS_CODE_STORAGE_KEY]: "right" });
  await assert.rejects(
    withAccessCode(async () => {
      throw new Error("server busy, try again");
    }, storage),
    /server busy/,
  );
  // The stored code is kept: only a 401 clears it.
  assert.equal(items.get(ACCESS_CODE_STORAGE_KEY), "right");
});

test("gradeError: a 401 is a code error, a 429 is a plain error with the detail", () => {
  const unauthorized = gradeError(401, "access code required");
  assert.ok(unauthorized instanceof AccessCodeError);
  assert.equal(unauthorized.message, "access code required");

  for (const detail of [
    "too many attempts, wait a minute",
    "grade limit reached (30 per hour), try again later",
  ]) {
    const limited = gradeError(429, detail);
    assert.equal(limited instanceof AccessCodeError, false);
    assert.equal(limited.message, detail);
  }
  assert.equal(gradeError(429, undefined).message, "grade request failed (429)");
  assert.equal(gradeError(503, "server busy, try again").message, "server busy, try again");
});

for (const detail of [
  "too many attempts, wait a minute",
  "grade limit reached (30 per hour), try again later",
]) {
  test(`a 429 (${detail}) keeps the stored code and doesn't ask for one`, async () => {
    const { storage, items } = memoryStorage({ [ACCESS_CODE_STORAGE_KEY]: "right" });
    const calls: (string | null)[] = [];
    await assert.rejects(
      withAccessCode(async (code) => {
        calls.push(code);
        throw gradeError(429, detail);
      }, storage),
      (err: unknown) => err instanceof Error && !(err instanceof AccessCodeError) && err.message === detail,
    );
    assert.deepEqual(calls, ["right"]);
    assert.equal(items.get(ACCESS_CODE_STORAGE_KEY), "right");
  });
}

test("a 429 on a just-entered code keeps that code", async () => {
  const { storage, items } = memoryStorage();
  await assert.rejects(
    withAccessCode(async () => {
      throw gradeError(429, "too many attempts, wait a minute");
    }, storage, "right"),
    /too many attempts/,
  );
  assert.equal(items.get(ACCESS_CODE_STORAGE_KEY), "right");
});

test("AccessCodeError carries the API's detail", () => {
  assert.equal(new AccessCodeError().message, "access code required");
  assert.equal(new AccessCodeError("nope").message, "nope");
  assert.ok(new AccessCodeError() instanceof Error);
});

test("only printable ASCII codes are valid", () => {
  for (const ok of ["pikachu-teal-river", "Abc123", "a_b.c!~"]) assert.equal(isValidCode(ok), true, ok);
  for (const bad of ["", "two words", "pokémon", "tab\there", "emoji🙂"]) {
    assert.equal(isValidCode(bad), false, bad);
  }
});
