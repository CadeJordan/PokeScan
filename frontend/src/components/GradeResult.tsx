"use client";

import type { GradeResponse } from "@/app/api-client";

type Props = { result: GradeResponse };

const formatPct = (n: number) => `${(n * 100).toFixed(1)}%`;

const confidenceLabel = (c: number): string => {
  if (c >= 0.7) return "High";
  if (c >= 0.4) return "Medium";
  return "Low";
};

const GradeResult = ({ result }: Props) => {
  const { grade, grade_int, confidence, sub_grades, notes, rank_probs } = result;

  return (
    <section className="overflow-hidden rounded-lg border border-white/[0.08] bg-white/[0.02]">
      {/* Header strip */}
      <div className="flex items-center justify-between border-b border-white/[0.08] px-5 py-3">
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">
          Prediction
        </span>
        <span className="font-mono text-[11px] tracking-wider text-zinc-600">
          generated · {new Date().toISOString().split("T")[0]}
        </span>
      </div>

      {/* Grade + confidence */}
      <div className="grid divide-y divide-white/[0.08] sm:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] sm:divide-x sm:divide-y-0">
        <Cell label="PSA grade">
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-[72px] font-semibold leading-none tracking-tight text-amber-300">
              {grade_int}
            </span>
            <span className="font-mono text-sm tabular-nums text-zinc-500">
              soft {grade.toFixed(2)}
            </span>
          </div>
        </Cell>

        <Cell label="Confidence">
          <div className="flex flex-col gap-2">
            <div className="flex items-baseline gap-3">
              <span className="font-mono text-[40px] font-semibold leading-none tracking-tight text-zinc-100">
                {formatPct(confidence)}
              </span>
              <span className="font-mono text-xs uppercase tracking-[0.16em] text-zinc-500">
                {confidenceLabel(confidence)}
              </span>
            </div>
            <ConfidenceBar value={confidence} />
            <p className="font-mono text-[11px] text-zinc-500">
              temperature-calibrated on validation
            </p>
          </div>
        </Cell>
      </div>

      {/* Sub-grades */}
      <div className="border-t border-white/[0.08]">
        <div className="border-b border-white/[0.08] px-5 py-2">
          <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">
            Sub-grades
          </span>
        </div>
        <div className="grid grid-cols-2 divide-x divide-y divide-white/[0.08] sm:grid-cols-4 sm:divide-y-0">
          <SubGradeCell
            label="Centering"
            value={sub_grades.centering?.grade ?? null}
            hint={
              sub_grades.centering?.left_right
                ? `${sub_grades.centering.left_right} L/R · ${sub_grades.centering.top_bottom} T/B`
                : undefined
            }
          />
          <SubGradeCell label="Corners" value={sub_grades.corners} pending />
          <SubGradeCell label="Edges" value={sub_grades.edges} pending />
          <SubGradeCell label="Surface" value={sub_grades.surface} pending />
        </div>
      </div>

      {/* Distribution */}
      <details className="group border-t border-white/[0.08]">
        <summary className="flex cursor-pointer select-none items-center justify-between px-5 py-3 transition hover:bg-white/[0.02]">
          <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">
            Distribution over grades
          </span>
          <span className="text-xs text-zinc-600 transition group-open:rotate-180">
            ▾
          </span>
        </summary>
        <div className="border-t border-white/[0.06] px-5 py-4">
          <ol className="grid grid-cols-9 gap-1.5">
            {rank_probs.map((p, i) => {
              const pct = Math.max(0, Math.min(1, p));
              const isPick = i + 2 === grade_int;
              return (
                <li
                  key={i}
                  className="flex flex-col items-center gap-1.5"
                >
                  <div className="flex h-16 w-full items-end justify-center">
                    <div
                      className={`w-full origin-bottom rounded-sm transition-all ${
                        isPick ? "bg-amber-300" : "bg-zinc-700"
                      }`}
                      style={{
                        height: `${Math.max(pct * 100, 2)}%`,
                      }}
                    />
                  </div>
                  <span
                    className={`font-mono text-[10px] tracking-wider ${
                      isPick ? "text-amber-300" : "text-zinc-500"
                    }`}
                  >
                    &gt;{i + 1}
                  </span>
                  <span
                    className={`font-mono text-[10px] tabular-nums ${
                      isPick ? "text-zinc-100" : "text-zinc-400"
                    }`}
                  >
                    {(p * 100).toFixed(0)}
                  </span>
                </li>
              );
            })}
          </ol>
          <p className="mt-3 font-mono text-[11px] text-zinc-600">
            P(grade &gt; N) — cumulative output of the CORN ordinal head
          </p>
        </div>
      </details>

      {notes.length > 0 && (
        <div className="border-t border-white/[0.08] px-5 py-4">
          <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">
            Notes
          </span>
          <ul className="mt-2 space-y-1.5">
            {notes.map((n) => (
              <li
                key={n}
                className="flex items-start gap-2 text-[13px] text-zinc-400"
              >
                <span className="mt-1.5 inline-block h-1 w-1 rounded-full bg-zinc-600" />
                <span>{n}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
};

const Cell = ({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) => (
  <div className="flex flex-col gap-3 px-5 py-5">
    <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">
      {label}
    </span>
    {children}
  </div>
);

const ConfidenceBar = ({ value }: { value: number }) => {
  const pct = Math.min(100, Math.max(2, value * 100));
  return (
    <div className="relative h-1 w-full overflow-hidden rounded-full bg-white/[0.06]">
      <div
        className="h-full bg-zinc-200 transition-all"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
};

const SubGradeCell = ({
  label,
  value,
  hint,
  pending = false,
}: {
  label: string;
  value: number | null;
  hint?: string;
  pending?: boolean;
}) => {
  const has = value !== null && value !== undefined;
  return (
    <div className="flex flex-col gap-1 px-5 py-4">
      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500">
        {label}
      </span>
      <span
        className={`font-mono text-2xl font-semibold tabular-nums ${
          has ? "text-zinc-100" : "text-zinc-700"
        }`}
      >
        {has ? (value as number).toFixed(1) : "—"}
      </span>
      {hint && (
        <span className="font-mono text-[10px] text-zinc-500">{hint}</span>
      )}
      {pending && !has && (
        <span className="font-mono text-[10px] tracking-wider text-zinc-600">
          phase c
        </span>
      )}
    </div>
  );
};

export default GradeResult;
