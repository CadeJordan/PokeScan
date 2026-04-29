import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { ReactQueryClientProvider } from "@/utils/react-query";
import Navbar from "@/components/Navbar";
import "./globals.css";

const inter = Inter({
	variable: "--font-geist-sans",
	subsets: ["latin"],
	weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
	title: "PokeScan — predict your PSA grade",
	description:
		"Upload a photo of your Pokemon card and get a predicted PSA grade plus a calibrated confidence score.",
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang="en">
			<body
				className={`${inter.className} min-h-screen bg-[#08090a] text-zinc-100 antialiased`}
			>
				<ReactQueryClientProvider>
					<header className="sticky top-0 z-30 border-b border-white/[0.06] bg-[#08090a]/80 backdrop-blur-md">
						<div className="mx-auto max-w-4xl px-6 py-4">
							<Navbar />
						</div>
					</header>
					<main className="mx-auto w-full max-w-4xl px-6 py-12 sm:py-16">
						{children}
					</main>
					<footer className="mx-auto w-full max-w-4xl px-6 pb-12 pt-8">
						<div className="border-t border-white/[0.06] pt-6 text-center text-[11px] text-zinc-600">
							PokeScan is an independent project. Not affiliated with PSA,
							Nintendo, Game Freak, or The Pokemon Company.
						</div>
					</footer>
				</ReactQueryClientProvider>
			</body>
		</html>
	);
}
