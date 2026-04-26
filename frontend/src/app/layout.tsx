import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { ReactQueryClientProvider } from "@/utils/react-query";
import Navbar from "@/components/Navbar";
import "./globals.css";

const inter = Inter({
	variable: "--font-geist-sans",
	subsets: ["latin"],
});

export const metadata: Metadata = {
	title: "PokeScan",
	description: "Predict the PSA grade your Pokemon card will receive.",
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang="en">
			<body className={`${inter.className} bg-slate-50 min-h-screen`}>
				<ReactQueryClientProvider>
					<header className="w-full border-b border-slate-200 bg-white">
						<div className="mx-auto max-w-3xl px-6 py-4">
							<Navbar />
						</div>
					</header>
					<main className="mx-auto w-full max-w-3xl px-6 py-8">
						{children}
					</main>
				</ReactQueryClientProvider>
			</body>
		</html>
	);
}
