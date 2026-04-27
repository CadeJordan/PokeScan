"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { getHealth } from "@/app/api-client";
import PokeballIcon from "./PokeballIcon";

const Navbar = () => {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: false,
  });

  let statusLabel = "checking";
  let dotColor = "bg-zinc-600";
  let textColor = "text-zinc-500";
  let pulse = false;
  if (isError) {
    statusLabel = "API offline";
    dotColor = "bg-rose-400";
    textColor = "text-zinc-400";
  } else if (data) {
    if (data.model_loaded) {
      statusLabel = "model ready";
      dotColor = "bg-emerald-400";
      textColor = "text-zinc-300";
      pulse = true;
    } else {
      statusLabel = "warming up";
      dotColor = "bg-amber-400";
      textColor = "text-zinc-400";
    }
  } else if (isLoading) {
    textColor = "text-zinc-500";
  }

  return (
    <nav className="flex w-full items-center justify-between gap-4">
      <Link
        href="/"
        className="group flex items-center gap-2.5"
        aria-label="PokeScan home"
      >
        <PokeballIcon className="h-5 w-5 text-zinc-300 transition-colors group-hover:text-white" />
        <span className="text-[15px] font-semibold tracking-tight text-zinc-100">
          PokeScan
        </span>
      </Link>

      <div
        className={`flex items-center gap-2 font-mono text-[11px] ${textColor}`}
      >
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${dotColor} ${
            pulse ? "animate-status-pulse" : ""
          }`}
        />
        <span>{statusLabel}</span>
      </div>
    </nav>
  );
};

export default Navbar;
