"use client";

import type { GradeResponse } from "@/app/api-client";

type Props = { result: GradeResponse };

const formatPct = (n: number) => `${(n * 100).toFixed(1)}%`;

const GradeResult = ({ result }: Props) => {
  const { grade, grade_int, confidence, sub_grades, notes } = result;
  const confidenceColor =
    confidence >= 0.7
      ? "bg-emerald-500"
      : confidence >= 0.4
        ? "bg-amber-500"
        : "bg-rose-500";

  return (
    <section className="flex flex-col gap-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <header className="flex items-end justify-between">
        <div>
          <h2 className="text-sm font-medium uppercase tracking-wide text-slate-500">
            Predicted PSA grade
          </h2>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-5xl font-bold tabular-nums text-slate-900">
              {grade_int}
            </span>
            <span className="text-lg text-slate-500 tabular-nums">
              (soft: {grade.toFixed(2)})
            </span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs uppercase tracking-wide text-slate-500">
            Confidence
          </div>
          <div className="mt-1 text-2xl font-semibold tabular-nums text-slate-900">
            {formatPct(confidence)}
          </div>
        </div>
      </header>

      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full ${confidenceColor} transition-all`}
          style={{ width: `${Math.min(100, Math.max(2, confidence * 100))}%` }}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
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

      <details className="text-xs text-slate-500">
        <summary className="cursor-pointer select-none text-slate-600 hover:text-slate-800">
          Distribution over grades
        </summary>
        <ol className="mt-2 grid grid-cols-9 gap-1 tabular-nums">
          {result.rank_probs.map((p, i) => (
            <li
              key={i}
              className="flex flex-col items-center rounded bg-slate-100 px-1 py-1.5"
            >
              <span className="text-[10px] text-slate-500">{`>${i + 1}`}</span>
              <span className="font-medium text-slate-800">
                {(p * 100).toFixed(0)}
              </span>
            </li>
          ))}
        </ol>
      </details>

      {notes.length > 0 && (
        <ul className="space-y-1 text-xs text-amber-700">
          {notes.map((n) => (
            <li key={n}>· {n}</li>
          ))}
        </ul>
      )}
    </section>
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
  const display = value !== null && value !== undefined ? value.toFixed(1) : "—";
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-slate-900">
        {display}
      </div>
      {hint && <div className="mt-0.5 text-[10px] text-slate-500">{hint}</div>}
      {pending && value === null && (
        <div className="mt-0.5 text-[10px] text-slate-400">phase C</div>
      )}
    </div>
  );
};

export default GradeResult;
