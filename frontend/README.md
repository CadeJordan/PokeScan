# PokeScan frontend

Next.js 16 (App Router) + React Query + Tailwind v4. A single page that uploads a card's front and back photos to the PokeScan API and shows the predicted PSA grade. It's Next 16, which breaks older APIs: read `AGENTS.md` before changing Next code.

## Develop

```powershell
copy .env.example .env.local   # NEXT_PUBLIC_API_BASE, including /api
npm install
npm run dev                    # http://localhost:3000
```

Run the API from `PokeScan/` (see the root `README.md`) with `CORS_ORIGINS` allowing `http://localhost:3000` (the default).

## Checks

```powershell
npm run lint
npx tsc --noEmit
npm test        # node --test on src/**/*.test.ts (plain-TS modules only)
npm run build
```

CI runs the same four (`.github/workflows/ci.yml`).

## How it talks to the API

- `src/app/api-client.ts` calls `${NEXT_PUBLIC_API_BASE}/health` and `/grade` from the browser, and mirrors the API's `GradeResponse` schema: keep the two in sync.
- **Access code.** When the API has `ACCESS_CODE` set, `/grade` answers 401 without the right `X-Access-Code` header. The page then asks for the code inline, keeps it in `localStorage` and retries the grade once; a refused code shows "That code didn't work" and is forgotten (`src/lib/accessCode.ts`).

## Deploy

Vercel (Hobby), importing the repo with **Root Directory** `frontend` and `NEXT_PUBLIC_API_BASE=<Cloud Run URL>/api`. Step by step: [`docs/deploy.md`](../docs/deploy.md).
