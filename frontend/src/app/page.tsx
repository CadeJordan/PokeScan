import Uploader from "@/components/Uploader";

const Home = () => {
	return (
		<div className="flex flex-col gap-16">
			<section className="flex flex-col items-start gap-5">
				<span className="font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">
					grade predictor · v0.1
				</span>
				<h1 className="max-w-2xl text-balance text-4xl font-semibold leading-[1.05] tracking-tight text-zinc-50 sm:text-5xl">
					Predict your PSA grade before you ship.
				</h1>
				<p className="max-w-xl text-balance text-[15px] leading-relaxed text-zinc-400">
					Upload a photo of the front and back of your Pokemon card. The model
					reads its corners, edges and surface and returns a predicted grade with
					its confidence; centering is measured separately.
				</p>
			</section>

			<Uploader />

			<section>
				<SectionHeading
					eyebrow="Pipeline"
					title="How a card is scored"
				/>
				<div className="grid border-t border-white/[0.06] sm:grid-cols-3 sm:divide-x sm:divide-white/[0.06]">
					<HowCard
						step="01"
						title="Detect & dewarp"
						body="Finds the card outline (slabs too), corrects the perspective and warps it to 600 × 840. No outline, no grade: you're asked to retake the photo."
					/>
					<HowCard
						step="02"
						title="Read the regions"
						body="A frozen DINOv2 vision model describes 11 regions per face: corners, edges, centre and the whole card. Four small probes trained on PSA-graded cards turn the regions into grade signals."
					/>
					<HowCard
						step="03"
						title="Check and combine"
						body="A guard checks the front is a front, the back a back, and the front is the right way up. A final probe combines the region signals into the grade distribution. Centering is measured separately from the borders."
					/>
				</div>
			</section>

			<section>
				<SectionHeading
					eyebrow="Photo guide"
					title="Tips for the cleanest scan"
				/>
				<dl className="grid gap-px overflow-hidden border border-white/[0.06] bg-white/[0.06] sm:grid-cols-2">
					<TipRow
						label="Background"
						body="Plain dark surface, no glare or flash."
					/>
					<TipRow
						label="Framing"
						body="Show the whole card with a little background around it; cropping is automatic."
					/>
					<TipRow
						label="Both sides"
						body="Front and back are both required; the model reads both."
					/>
					<TipRow
						label="Limitations"
						body="Screening tool only. Not a substitute for a real PSA submission."
					/>
				</dl>
			</section>
		</div>
	);
};

const SectionHeading = ({
	eyebrow,
	title,
}: {
	eyebrow: string;
	title: string;
}) => (
	<div className="mb-5 flex flex-col gap-1">
		<span className="font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">
			{eyebrow}
		</span>
		<h2 className="text-xl font-semibold tracking-tight text-zinc-100">
			{title}
		</h2>
	</div>
);

const HowCard = ({
	step,
	title,
	body,
}: {
	step: string;
	title: string;
	body: string;
}) => (
	<div className="flex flex-col gap-2 px-5 py-6">
		<span className="font-mono text-[11px] tracking-[0.18em] text-zinc-500">
			{step}
		</span>
		<h3 className="text-base font-semibold text-zinc-100">{title}</h3>
		<p className="text-[13.5px] leading-relaxed text-zinc-400">{body}</p>
	</div>
);

const TipRow = ({ label, body }: { label: string; body: string }) => (
	<div className="bg-[#08090a] px-5 py-4">
		<dt className="font-mono text-[11px] uppercase tracking-[0.16em] text-zinc-500">
			{label}
		</dt>
		<dd className="mt-1 text-[13.5px] leading-relaxed text-zinc-300">
			{body}
		</dd>
	</div>
);

export default Home;
