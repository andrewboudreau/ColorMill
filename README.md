# ColorMill

A browser-based color milling and mixing simulator.

This project starts as a small TypeScript + Canvas app with a raylib-inspired shape/input loop. The goal is to model how two-roll mills fold, stretch, cut, and blend colored material.

## Scripts

```bash
npm install
npm test
npm run build
npm run dev
```

## Deployment

Pushes to `main` run tests, build the app, and publish the generated `dist` folder to GitHub Pages using GitHub Actions.
