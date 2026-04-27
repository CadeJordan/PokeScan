"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  label: string;
  file: File | null;
  onFileChange: (file: File | null) => void;
  accept?: string;
};

const ALLOWED = new Set(["image/jpeg", "image/png", "image/webp"]);

const ImageDropzone = ({
  label,
  file,
  onFileChange,
  accept = "image/*",
}: Props) => {
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

  const acceptFile = useCallback(
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
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-400">
          {label}
        </span>
        <span className="font-mono text-[10px] tracking-wider text-zinc-600">
          5 : 7 · jpg/png/webp
        </span>
      </div>

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
          acceptFile(e.dataTransfer.files?.[0]);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        className={`group relative flex aspect-[5/7] cursor-pointer items-center justify-center overflow-hidden rounded-lg border bg-white/[0.02] transition-colors ${
          isDragging
            ? "border-zinc-300 bg-white/[0.04]"
            : preview
              ? "border-white/[0.10]"
              : "border-dashed border-white/[0.10] hover:border-white/25 hover:bg-white/[0.04]"
        }`}
      >
        <CornerMarks />

        {preview ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={preview}
              alt={`${label} preview`}
              className="relative h-full w-full object-contain"
            />
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onFileChange(null);
              }}
              className="absolute right-2 top-2 z-10 rounded-md border border-white/[0.10] bg-zinc-950/80 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-zinc-300 backdrop-blur transition hover:border-white/20 hover:text-zinc-100"
            >
              Remove
            </button>
          </>
        ) : (
          <div className="relative flex flex-col items-center gap-2 px-6 text-center">
            <UploadIcon />
            <span className="text-sm font-medium text-zinc-200">
              Drop {label.toLowerCase()} photo
            </span>
            <span className="text-xs text-zinc-500">
              or click to upload / use camera
            </span>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept={accept}
          capture="environment"
          className="hidden"
          onChange={(e) => acceptFile(e.target.files?.[0])}
        />
      </div>

      {error && <span className="font-mono text-[11px] text-rose-300">{error}</span>}
    </div>
  );
};

const CornerMarks = () => {
  // Thin L-shaped registration marks at each interior corner.
  const base = "absolute h-3.5 w-3.5 border-white/30";
  return (
    <>
      <span aria-hidden className={`${base} left-2.5 top-2.5 border-l border-t`} />
      <span aria-hidden className={`${base} right-2.5 top-2.5 border-r border-t`} />
      <span aria-hidden className={`${base} bottom-2.5 left-2.5 border-l border-b`} />
      <span aria-hidden className={`${base} bottom-2.5 right-2.5 border-r border-b`} />
    </>
  );
};

const UploadIcon = () => (
  <svg
    viewBox="0 0 24 24"
    className="h-7 w-7 text-zinc-400 transition-colors group-hover:text-zinc-200"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
    <path d="M12 16V4" />
    <path d="M7 9l5-5 5 5" />
    <path d="M5 17v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2" />
  </svg>
);

export default ImageDropzone;
