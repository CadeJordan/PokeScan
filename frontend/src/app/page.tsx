import Uploader from "@/components/Uploader";

const Home = () => {
	return (
		<div className="flex flex-col gap-6">
			<section className="flex flex-col gap-2">
				<h1 className="text-3xl font-bold tracking-tight text-slate-900">
					Grade your Pokemon card
				</h1>
				<p className="text-sm text-slate-600">
					Upload sharp, well-lit photos of the front and back. We&apos;ll predict
					the PSA grade and give a calibrated confidence score.
				</p>
			</section>

			<Uploader />

			<section className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-600 shadow-sm">
				<h2 className="text-base font-semibold text-slate-900">Tips for best results</h2>
				<ul className="mt-2 list-disc space-y-1 pl-5">
					<li>Shoot on a plain dark background, head-on, no glare.</li>
					<li>Fill the frame with the card; cropping is automatic.</li>
					<li>Both front and back are required — back contributes ~30% of the grade.</li>
					<li>This is a screening tool, not a substitute for a real PSA submission.</li>
				</ul>
			</section>
		</div>
	);
};

export default Home;
