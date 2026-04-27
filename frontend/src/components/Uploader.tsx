"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import ImageDropzone from "./ImageDropzone";
import GradeResult from "./GradeResult";
import { gradeCard, type GradeResponse } from "@/app/api-client";

const Uploader = () => {
  const [front, setFront] = useState<File | null>(null);
  const [back, setBack] = useState<File | null>(null);

  const mutation = useMutation<
    GradeResponse,
    Error,
    { front: File; back: File }
  >({
    mutationFn: ({ front, back }) => gradeCard(front, back),
  });

  const ready = front !== null && back !== null;
  const canSubmit = ready && !mutation.isPending;

  const reset = () => {
    setFront(null);
    setBack(null);
    mutation.reset();
  };

  return (
    <section className="flex flex-col gap-6">
      <div className="grid gap-5 sm:grid-cols-2">
        <ImageDropzone label="Front" file={front} onFileChange={setFront} />
        <ImageDropzone label="Back" file={back} onFileChange={setBack} />
      </div>

      <div className="flex flex-col gap-4 border-t border-white/[0.06] pt-6 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-[0.16em]">
          <Step active={front !== null} label="Front" />
          <Divider />
          <Step active={back !== null} label="Back" />
          <Divider />
          <Step active={ready} label="Ready" />
        </div>

        <div className="flex flex-col-reverse gap-2 sm:flex-row sm:items-center">
          {(front || back) && (
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center justify-center rounded-md border border-white/[0.08] bg-transparent px-4 py-2 text-sm font-medium text-zinc-300 transition hover:border-white/20 hover:text-zinc-100"
            >
              Clear
            </button>
          )}
          <button
            type="button"
            disabled={!canSubmit}
            onClick={() => {
              if (front && back) mutation.mutate({ front, back });
            }}
            className="group inline-flex min-w-[160px] items-center justify-center gap-2 rounded-md bg-zinc-100 px-5 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-white disabled:cursor-not-allowed disabled:bg-zinc-700 disabled:text-zinc-400"
          >
            {mutation.isPending ? (
              <>
                <Spinner />
                Grading
              </>
            ) : (
              <>
                Predict grade
                <ArrowIcon />
              </>
            )}
          </button>
        </div>
      </div>

      {mutation.isError && (
        <div className="flex items-start gap-3 border border-rose-400/20 bg-rose-400/[0.06] px-4 py-3 text-sm">
          <span className="mt-1.5 inline-block h-1.5 w-1.5 rounded-full bg-rose-400" />
          <div>
            <div className="font-medium text-rose-200">
              Couldn&apos;t grade that pair
            </div>
            <div className="text-rose-300/80">{mutation.error.message}</div>
          </div>
        </div>
      )}

      {mutation.data && (
        <div className="animate-grade-rise">
          <GradeResult result={mutation.data} />
        </div>
      )}
    </section>
  );
};

const Step = ({ active, label }: { active: boolean; label: string }) => (
  <span className="flex items-center gap-1.5">
    <span
      className={`inline-block h-1.5 w-1.5 rounded-full ${
        active ? "bg-zinc-100" : "bg-zinc-700"
      }`}
    />
    <span className={active ? "text-zinc-300" : "text-zinc-500"}>{label}</span>
  </span>
);

const Divider = () => <span className="h-px w-5 bg-white/10" />;

const ArrowIcon = () => (
  <svg
    viewBox="0 0 20 20"
    className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5 group-disabled:translate-x-0"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
    <path d="M4 10h12M11 5l5 5-5 5" />
  </svg>
);

const Spinner = () => (
  <svg
    className="h-3.5 w-3.5 animate-spin"
    viewBox="0 0 24 24"
    fill="none"
    aria-hidden
  >
    <circle
      cx="12"
      cy="12"
      r="9"
      stroke="currentColor"
      strokeOpacity="0.25"
      strokeWidth="3"
    />
    <path
      d="M21 12a9 9 0 0 0-9-9"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
    />
  </svg>
);

export default Uploader;
