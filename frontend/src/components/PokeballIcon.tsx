type Props = {
  className?: string;
};

/**
 * Minimal outline Pokeball mark. Stroke-only, no fill, takes its color
 * from `currentColor`. Reads as a brand mark without competing with
 * the rest of the UI.
 */
const PokeballIcon = ({ className }: Props) => (
  <svg
    viewBox="0 0 64 64"
    className={className}
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    role="img"
    aria-label="PokeScan"
  >
    <circle cx="32" cy="32" r="28" />
    <line x1="4" y1="32" x2="22" y2="32" />
    <line x1="42" y1="32" x2="60" y2="32" />
    <circle cx="32" cy="32" r="10" />
    <circle cx="32" cy="32" r="3" fill="currentColor" stroke="none" />
  </svg>
);

export default PokeballIcon;
