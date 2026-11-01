import assert from "node:assert/strict";
import { test } from "node:test";
import { perGradeProbs } from "./gradeProbs.ts";

// The API's grade_int: the median grade, 1 + the number of P(grade > k) above 0.5.
const medianGrade = (rankProbs: number[]) =>
  1 + rankProbs.filter((p) => p > 0.5).length;

const assertClose = (actual: number[], expected: number[]) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((p, i) =>
    assert.ok(
      Math.abs(p - expected[i]) < 1e-9,
      `grade ${i + 1}: got ${p}, expected ${expected[i]}`,
    ),
  );
};

const sum = (ps: number[]) => ps.reduce((a, b) => a + b, 0);

test("a typical distribution becomes 10 per-grade probabilities", () => {
  const rank = [0.99, 0.97, 0.94, 0.9, 0.85, 0.75, 0.6, 0.35, 0.1];
  const probs = perGradeProbs(rank);
  assertClose(probs, [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.25, 0.25, 0.1]);
  assert.ok(Math.abs(sum(probs) - 1) < 1e-9);
  assert.equal(medianGrade(rank), 8);
});

test("a predicted 10 is P(grade > 9)", () => {
  const rank = [0.999, 0.998, 0.997, 0.995, 0.99, 0.98, 0.95, 0.9, 0.7];
  const probs = perGradeProbs(rank);
  assert.equal(medianGrade(rank), 10);
  assert.ok(Math.abs(probs[9] - 0.7) < 1e-9);
  assert.ok(Math.abs(sum(probs) - 1) < 1e-9);
});

test("a predicted 1 is 1 - P(grade > 1)", () => {
  const rank = [0.4, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001];
  const probs = perGradeProbs(rank);
  assert.equal(medianGrade(rank), 1);
  assert.ok(Math.abs(probs[0] - 0.6) < 1e-9);
  assert.ok(Math.abs(sum(probs) - 1) < 1e-9);
});

test("a non-monotone input clips the negative difference and renormalises", () => {
  // P(grade > 2) > P(grade > 1): grade 2's difference is -0.05, the rest sum to 1.05.
  const rank = [0.9, 0.95, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05];
  const probs = perGradeProbs(rank);
  const raw = [0.1, 0, 0.15, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05];
  assertClose(probs, raw.map((p) => p / 1.05));
  assert.equal(probs[1], 0);
  assert.ok(Math.abs(sum(probs) - 1) < 1e-9);
});

test("inputs just outside [0, 1] are differenced as given, like decode", () => {
  // decode doesn't clamp rank_probs: P(>1) = 1.02 makes grade 1's difference -0.02
  // and P(>9) = -0.01 makes grade 10's -0.01; both clip to 0, the rest sum to 1.03.
  const rank = [1.02, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, -0.01];
  const raw = [0, 0.07, 0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.11, 0];
  assertClose(perGradeProbs(rank), raw.map((p) => p / 1.03));
});

test("degenerate inputs put all the mass on one end", () => {
  assertClose(perGradeProbs(Array(9).fill(0)), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  assertClose(perGradeProbs(Array(9).fill(1)), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
});

test("the borderline 9/10 example highlights P(9), not P(grade >= 9)", () => {
  const rank = [1, 1, 1, 1, 1, 1, 1, 0.85, 0.4];
  const probs = perGradeProbs(rank);
  const grade = medianGrade(rank);
  assert.equal(grade, 9);
  assertClose(probs, [0, 0, 0, 0, 0, 0, 0, 0.15, 0.45, 0.4]);
  // The old chart highlighted 0.85 here, next to a confidence of 0.45.
  assert.ok(Math.abs(probs[grade - 1] - 0.45) < 1e-9);
});

test("on a close 9/10 call the highlighted grade can sit next to a taller bar", () => {
  // P(grade > 9) = 0.45 is under 0.5, so the median is 9 while 10 is likelier.
  const rank = [1, 1, 1, 1, 1, 1, 1, 0.85, 0.45];
  const probs = perGradeProbs(rank);
  const grade = medianGrade(rank);
  assert.equal(grade, 9);
  assertClose(probs, [0, 0, 0, 0, 0, 0, 0, 0.15, 0.4, 0.45]);
  assert.ok(probs[grade - 1] < probs[grade]);
});

test("a real /api/grade response: the highlighted value is the confidence", () => {
  // Cert 76000632 (PSA 7) from data/images, graded by the probe model on 2026-10-31.
  const rank = [0.9947, 0.9838, 0.9309, 0.8688, 0.5615, 0.3389, 0.171, 0.0434, 0.0054];
  const response = { grade_int: 6, confidence: 0.2227 };
  const probs = perGradeProbs(rank);
  assert.equal(medianGrade(rank), response.grade_int);
  // Within 0.1 percentage points; the API rounds to 4 decimals.
  assert.ok(Math.abs(probs[response.grade_int - 1] - response.confidence) < 0.001);
  // A close call: grade 5 is likelier than the median grade 6.
  assert.ok(probs[4] > probs[5]);
});
