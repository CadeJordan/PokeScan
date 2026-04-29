"use client";
import Link from "next/link";

const NotFound = () => {
  return (
    <div className="flex min-h-[60vh] flex-col items-start justify-center gap-6">
      <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">
        404 · not found
      </span>
      <h1 className="text-4xl font-semibold tracking-tight text-zinc-50 sm:text-5xl">
        That card hasn&apos;t been graded yet.
      </h1>
      <p className="max-w-md text-[15px] text-zinc-400">
        The page you were looking for either moved or never existed.
      </p>
      <Link
        href="/"
        className="inline-flex items-center gap-2 rounded-md bg-zinc-100 px-5 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-white"
      >
        Back home
        <svg
          viewBox="0 0 20 20"
          className="h-3.5 w-3.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <path d="M4 10h12M11 5l5 5-5 5" />
        </svg>
      </Link>
    </div>
  );
};

export default NotFound;
