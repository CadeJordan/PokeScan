"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import ImageDropzone from "./ImageDropzone";
import GradeResult from "./GradeResult";
import { gradeCard, type GradeResponse } from "@/app/api-client";

const Uploader = () => {
  const [front, setFront] = useState<File | null>(null);
  const [back, setBack] = useState<File | null>(null);

  const mutation = useMutation<GradeResponse, Error, { front: File; back: File }>({
    mutationFn: ({ front, back }) => gradeCard(front, back),
  });

  const canSubmit = front !== null && back !== null && !mutation.isPending;

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-4">
        <ImageDropzone label="Front" file={front} onFileChange={setFront} />
        <ImageDropzone label="Back" file={back} onFileChange={setBack} />
      </div>

      <div className="flex flex-col items-stretch gap-2">
        <button
          type="button"
          disabled={!canSubmit}
          onClick={() => {
            if (front && back) mutation.mutate({ front, back });
          }}
          className="rounded-xl bg-indigo-600 px-6 py-3 text-base font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {mutation.isPending ? "Grading…" : "Predict grade"}
        </button>
        {mutation.isError && (
          <p className="text-sm text-rose-600">{mutation.error.message}</p>
        )}
      </div>

      {mutation.data && <GradeResult result={mutation.data} />}
    </div>
  );
};

export default Uploader;
