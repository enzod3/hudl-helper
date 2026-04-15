# DB Scouting Dashboard

Apriori-based route tendency mining from Hudl film data. Built for DB coaches and defensive coordinators.

## Quick Start

```bash
npm install
npm run dev
```

Opens at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

Output goes to `dist/` — deploy that folder to any static host.

## Deploy Options

**Vercel** (easiest):
```bash
npm i -g vercel
vercel
```

**Netlify**:
- Connect your repo, set build command to `npm run build` and publish directory to `dist`

**GitHub Pages**:
- Add `base: '/your-repo-name/'` to `vite.config.js`
- Build and push `dist/` to `gh-pages` branch

**Any static host** (S3, Cloudflare Pages, etc.):
- Upload the `dist/` folder

## How It Works

1. **Create games** in the sidebar (e.g. "vs Lincoln Wk3", "vs Central Wk5")
2. **Upload Hudl JSON exports** to each game (multiple files per game)
3. **Toggle games on/off** to combine data across matchups
4. **Adjust filters** — view mode (F/B/Full), min plays, down, distance, formation, backfield
5. **Field position slider** — drag handles to limit by field zone, or use presets (20 & In, Backed Up, etc.)
6. **AI Scouting Report** — paste your OpenAI API key and hit ⚡ to generate a full written breakdown

## Field Zones

| Zone | Yards to Go (Own GL) | Description |
|------|---------------------|-------------|
| Backed Up | 1-15 | Pinned deep |
| Open Field | 16-60 | Between the 20s |
| The Fringe | 61-80 | Approaching scoring territory |
| Red Zone | 81-90 | Inside the 20 |
| Gold Zone | 91-99 | Inside the 10 |
| 20 & In | 81-99 | Composite: Red + Gold |

## Tech Stack

- React 18 + Vite
- Recharts for visualization
- OpenAI GPT-4o for AI scouting reports
- Zero backend — everything runs client-side
