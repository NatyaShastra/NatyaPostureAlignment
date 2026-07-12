# Dance Coach AI — Frontend

Next.js frontend for the Dance Coach AI system. Connects to the HuggingFace Spaces FastAPI backend and renders adavu classification results, joint angle analysis, coaching feedback, and skeleton overlay image.

---

## File structure

```
dance-coach-frontend/
│
├── package.json              — dependencies and scripts
├── next.config.js            — Next.js config
├── postcss.config.js         — required for Tailwind
├── tailwind.config.js        — Tailwind theme with custom colours
├── tsconfig.json             — TypeScript config
├── .env.example              — copy this to .env.local for local dev
├── .gitignore
│
├── app/
│   ├── layout.tsx            — root layout, loads fonts, sets metadata
│   ├── globals.css           — global styles, design tokens, animations
│   └── page.tsx              — main page with all four states: idle, analysing, results, error
│
├── components/
│   ├── UploadZone.tsx        — drag and drop or click to upload video
│   ├── AnalysingState.tsx    — loading screen with spinning mandala and progress steps
│   ├── ScoreCard.tsx         — overall score, region breakdown, grade badge, top-k predictions
│   ├── FlaggedJoints.tsx     — joint deviation table with sigma bars per joint
│   ├── CoachingFeedback.tsx  — three section LLM feedback with colour coded panels
│   └── SkeletonOverlay.tsx   — base64 image display with click to enlarge lightbox
│
└── lib/
    └── api.ts                — typed API client, analyseVideo() and healthCheck() functions
```

---

## Local development

```bash
# 1. Install dependencies
npm install

# 2. Create your local env file
cp .env.example .env.local
# Edit .env.local and set NEXT_PUBLIC_API_URL to your HF Space URL

# 3. Run dev server
npm run dev
# Opens at http://localhost:3000
```

---

## Deploy to Vercel

1. Push this repo to GitHub (make sure the repo root is dance-coach-frontend, not a parent folder)

2. Go to vercel.com, click Add New Project, import the GitHub repo

3. Vercel will auto-detect Next.js. No build settings need to be changed.

4. Under Environment Variables add:
   - Key: NEXT_PUBLIC_API_URL
   - Value: https://theusefulnerd-dance-coach-ai.hf.space

5. Deploy

6. After deployment, copy your Vercel URL and go to the HuggingFace Space settings. Update the CORS_ORIGINS secret to include your Vercel URL:
   CORS_ORIGINS = https://your-app.vercel.app,http://localhost:3000

   Then restart the Space so it picks up the new CORS setting.

---

## Environment variables

| Variable | Description | Example |
|---|---|---|
| NEXT_PUBLIC_API_URL | URL of the HF Spaces FastAPI backend | https://theusefulnerd-dance-coach-ai.hf.space |

---

## Backend

The backend is a separate repo deployed to HuggingFace Spaces. This frontend only calls two endpoints:

- GET /health — liveness check
- POST /analyse — upload video, receive full coaching report as JSON

The overlay image comes back as a base64 JPEG string in the overlay_image_b64 field. The SkeletonOverlay component decodes and displays it as an img tag.

---

## Design

Cultural warm aesthetic inspired by classical Indian arts. Colour palette is saffron, gold, crimson on a deep ink background. Headings use Playfair Display, body text uses Crimson Pro. Results are shown in a three column layout designed for desktop teacher demo use.
