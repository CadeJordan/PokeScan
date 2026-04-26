"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  label: string;
  file: File | null;
  onFileChange: (file: File | null) => void;
  accept?: string;
};

const ALLOWED = new Set(["image/jpeg", "image/png", "image/webp"]);

const ImageDropzone = ({ label, file, onFileChange, accept = "image/*" }: Props) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept_file = useCallback(
    (f: File | undefined | null) => {
      setError(null);
      if (!f) return;
      if (!ALLOWED.has(f.type)) {
        setError("Use JPEG, PNG, or WebP");
        return;
      }
      if (f.size > 12 * 1024 * 1024) {
        setError("Image too large (max 12 MB)");
        return;
      }
      onFileChange(f);
    },
    [onFileChange],
  );

  return (
    <div className="flex flex-col gap-2">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          accept_file(e.dataTransfer.files?.[0]);
        }}
        className={`relative flex aspect-[5/7] cursor-pointer items-center justify-center overflow-hidden rounded-xl border-2 border-dashed transition ${
          isDragging
            ? "border-indigo-400 bg-indigo-50"
            : preview
              ? "border-slate-300 bg-slate-50"
              : "border-slate-300 bg-white hover:border-indigo-400 hover:bg-indigo-50/40"
        }`}
      >
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={preview}
            alt={`${label} preview`}
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex flex-col items-center gap-1 px-4 text-center text-sm text-slate-500">
            <span className="font-medium text-slate-700">Drop image here</span>
            <span className="text-xs">or click to upload / use camera</span>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          capture="environment"
          className="hidden"
          onChange={(e) => accept_file(e.target.files?.[0])}
        />
      </div>
      {error ? (
        <span className="text-xs text-rose-600">{error}</span>
      ) : file ? (
        <button
          type="button"
          onClick={() => onFileChange(null)}
          className="self-start text-xs text-slate-500 underline-offset-2 hover:underline"
        >
          remove
        </button>
      ) : null}
    </div>
  );
};

export default ImageDropzone;
