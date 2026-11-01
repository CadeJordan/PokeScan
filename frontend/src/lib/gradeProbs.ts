// Plain TypeScript with no `@/` imports: gradeProbs.test.ts runs it directly
// under `node --test --experimental-strip-types`.

// Per-grade probabilities from the API's cumulative `rank_probs`, where
// rankProbs[k - 1] is P(grade > k) for k = 1..9. Returns P(grade = k) for
// k = 1..10 at index k - 1. Like backend/app/ml/grade_head.py::decode, negative
// differences (a non-monotone input) are clipped to 0 and the rest renormalised,
// so the entry for `grade_int` is the response's `confidence`.
export const perGradeProbs = (rankProbs: readonly number[]): number[] => {
  const above = [1, ...rankProbs];
  const below = [...rankProbs, 0];
  const clipped = above.map((p, i) => Math.max(0, p - below[i]));
  const total = clipped.reduce((sum, p) => sum + p, 0);
  return clipped.map((p) => p / Math.max(total, 1e-12));
};
