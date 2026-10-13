# NetRoute Intelligence landing page

This Next.js application is the public presentation layer for the Predictive Network Degradation & Route Optimizer. It explains the system, reports validated evidence, previews the eight Streamlit workflows, and includes a guided route-decision demo at `/demo`.

The browser demo uses deterministic scenarios and client-side path calculations for explanation. Interactive topology editing, LTE uploads and scoring, complete failure controls, six-algorithm benchmarks, Mininet evidence, and observability belong to the Python Streamlit application.

Production: [predictive-network-route-optimizer.vercel.app](https://predictive-network-route-optimizer.vercel.app)

## Local development

```bash
npm ci
npm run dev
```

Open `http://localhost:3000`.

## Verification

```bash
npm run typecheck
npm test
npm run build
```

## Vercel configuration

Use `web` as the Vercel project root. The committed `vercel.json` selects the Next.js framework.

Set `NEXT_PUBLIC_STREAMLIT_URL` only after the Streamlit application is hosted on a suitable Python platform. Without it, dashboard calls to action link to the root README’s local setup section, so the site never implies that a live dashboard exists.
