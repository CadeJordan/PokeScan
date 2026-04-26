"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { getHealth } from "@/app/api-client";

const Navbar = () => {
  const { data } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: false,
  });

  const ready = data?.model_loaded === true;
  const statusLabel = data
    ? ready
      ? "model: ready"
      : "model: not loaded"
    : "model: ?";
  const statusColor = data
    ? ready
      ? "bg-emerald-500"
      : "bg-amber-500"
    : "bg-slate-300";

  return (
    <div className="flex w-full items-center justify-between">
      <Link
        href="/"
        className="text-xl font-bold tracking-tight text-slate-900"
      >
        PokeScan
      </Link>
      <div className="flex items-center gap-2 text-xs text-slate-500">
        <span className={`inline-block h-2 w-2 rounded-full ${statusColor}`} />
        <span>{statusLabel}</span>
      </div>
    </div>
  );
};

export default Navbar;
