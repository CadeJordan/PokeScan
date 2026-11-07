// The shared access code (intent/archive/cloud-run-deploy). When the API has ACCESS_CODE
// set, /grade answers 401 without the right `X-Access-Code` header; the page then
// asks for the code, keeps it in localStorage and retries the grade once.
//
// Plain TypeScript with no `@/` imports: accessCode.test.ts runs it directly
// under `node --test`.

export const ACCESS_CODE_HEADER = "X-Access-Code";
export const ACCESS_CODE_STORAGE_KEY = "pokescan.accessCode";
export const WRONG_CODE_MESSAGE = "That code didn't work";

/** /grade answered 401: the access code is missing or wrong. */
export class AccessCodeError extends Error {
  constructor(message = "access code required") {
    super(message);
    this.name = "AccessCodeError";
  }
}

/**
 * The error for a failed /grade response: a 401 is an `AccessCodeError` (ask for
 * the code); anything else, including a 429 rate limit ("too many attempts, wait
 * a minute" or "grade limit reached ..."), is a plain `Error` with the API's
 * detail, so it's shown as is and never taken for a wrong code.
 */
export function gradeError(status: number, detail: string | undefined): Error {
  if (status === 401) return new AccessCodeError(detail);
  return new Error(detail ?? `grade request failed (${status})`);
}

export type CodeStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

/** The browser's localStorage, or null where it's unavailable (SSR, blocked). */
export function browserStorage(): CodeStorage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

export function loadCode(storage: CodeStorage | null): string | null {
  try {
    const code = storage?.getItem(ACCESS_CODE_STORAGE_KEY);
    return code ? code : null;
  } catch {
    return null;
  }
}

export function saveCode(storage: CodeStorage | null, code: string): void {
  try {
    storage?.setItem(ACCESS_CODE_STORAGE_KEY, code);
  } catch {
    // Private mode or a full quota: the code still works for this request.
  }
}

export function clearCode(storage: CodeStorage | null): void {
  try {
    storage?.removeItem(ACCESS_CODE_STORAGE_KEY);
  } catch {
    // Nothing to clear.
  }
}

/** A code an HTTP header can carry: printable ASCII, no spaces. Anything else
 * could never match (or makes `fetch` throw), so it's refused up front. */
export function isValidCode(code: string): boolean {
  return /^[!-~]+$/.test(code);
}

/** Request headers carrying the code; none when there's no code. */
export function accessCodeHeaders(code: string | null): Record<string, string> {
  return code ? { [ACCESS_CODE_HEADER]: code } : {};
}

export type AccessOutcome<T> =
  | { kind: "ok"; value: T }
  // The API wants a code: none was sent (wrongCode false), or the one sent was
  // refused (wrongCode true; the stored code has been cleared).
  | { kind: "needs-code"; wrongCode: boolean };

/**
 * Run one request with the access code: `enteredCode` (just typed; saved first)
 * or else the stored one. A 401 (`AccessCodeError`) becomes "needs-code"; any
 * other error is rethrown. `run` is called exactly once, so typing a code and
 * submitting it retries the grade once.
 */
export async function withAccessCode<T>(
  run: (code: string | null) => Promise<T>,
  storage: CodeStorage | null,
  enteredCode?: string,
): Promise<AccessOutcome<T>> {
  const entered = enteredCode?.trim();
  if (entered) saveCode(storage, entered);
  const code = entered || loadCode(storage);
  try {
    return { kind: "ok", value: await run(code) };
  } catch (err) {
    if (!(err instanceof AccessCodeError)) throw err;
    if (code) clearCode(storage);
    return { kind: "needs-code", wrongCode: code !== null };
  }
}
