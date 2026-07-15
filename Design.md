# Design.md — Sentiment, Issue & Topic Analysis Dashboard

## Overview
Analytics dashboard for the AI Customer Review Analytics Platform (see `PROJECT.md`) — surfaces sentiment trends, issue-tag breakdowns, and per-category review intelligence from the shared BGE-embedding pipeline (sentiment classification + unsupervised issue clustering, one embedding pass, one Lambda).

**Source of truth for this merge:**
- **Structure, components, density, interaction patterns** → `design-2.md` (the actual dashboard spec)
- **Color, typography, spacing** → `DESIGN-coinbase.md`, wholesale, not cherry-picked
- **Border radius** → kept from `design-2.md` — not part of the requested swap, and Coinbase's pill/24px marketing radius doesn't survive contact with a dense data screen

The result: Coinbase's quiet, single-accent-color, institutional-calm visual language, applied to a high-density SaaS dashboard grammar instead of a marketing site.

---

## Colors

| Token | Value | Role | Migrated from |
|---|---|---|---|
| `primary` | `#0052FF` | CTAs, active states, links, focus rings, interactive highlights — Coinbase Blue | design-2 primary (indigo `#6366F1`) |
| `primary-active` | `#003ECC` | Hover/press on primary elements | design-2 primary-hover |
| `primary-disabled` | `#A8B8CC` | Disabled primary buttons | new — not in design-2 |
| `ink` (text-primary) | `#0A0B0D` | Headings, body text, primary labels | design-2 text-primary (`#0A0A0A`) |
| `body` (text-secondary) | `#5B616E` | Descriptions, metadata, secondary labels | design-2 text-secondary (`#6B6B6B`) |
| `muted` | `#7C828A` | Placeholders, timestamps, disabled states | design-2 neutral (`#9C9C9C`) |
| `hairline` (border) | `#DEE1E6` | Card borders, dividers, input borders | design-2 border (`#E8E8EC`) |
| `canvas` (surface) | `#FFFFFF` | Cards, panels, modals, nav backdrop | unchanged |
| `surface-soft` (background) | `#F7F7F7` | Page background | design-2 background (`#FAFAFA`) |
| `surface-strong` | `#EEF0F3` | Chip backgrounds, secondary buttons, icon plates | design-2 gray-100 |
| `semantic-up` (success) | `#05B169` | Positive sentiment, published status, confirmations | design-2 success (`#10B981`) |
| `semantic-down` (error) | `#CF202F` | Negative sentiment, rejected status, destructive actions | design-2 error (`#EF4444`) |
| `warning` *(repurposed accent-yellow)* | `#F4B000` | Neutral sentiment, pending status, low-confidence flags | design-2 warning (`#F59E0B`) |

**Dropped:** design-2's secondary green (`#20970B`, "reserved exclusively for the DESIGN.md brand highlight") — that token existed for a homepage brand moment this project doesn't have. Not carried forward.

**Note on `warning`:** Coinbase documents `accent-yellow` as illustrative-only, used inside Bitcoin glyph art, explicitly *not* an action or semantic color. This dashboard needs a third sentiment state and a pending-status color, and Coinbase's system doesn't otherwise provide one. Repurposing it here is a deliberate deviation from the source doc's own rule — done because there's no alternative, not because it's a clean fit.

---

## Typography

Coinbase's licensed typefaces (CoinbaseDisplay, CoinbaseSans, CoinbaseMono) aren't available — using Coinbase's own documented substitutes:

- **Display + Body**: `Inter` — one family, not two. Coinbase differentiates display from body through **weight**, not typeface pairing. This replaces design-2's General Sans / DM Sans contrast entirely. You lose the "editorial magazine" tension between a geometric display font and a humanist body font — that's the cost of matching Coinbase's calmer, flatter voice. If you want that contrast back later, that's a typography decision to make deliberately, not something to quietly reintroduce.
- **Code / numeric**: `JetBrains Mono` — this one was already shared between both source docs, no conflict. **Every number on this dashboard renders in JetBrains Mono**: sentiment percentages, confidence margins, review counts, deltas. No exceptions — this is Coinbase's hardest typographic rule and it maps cleanly onto an analytics product.

### Type scale (condensed from Coinbase's marketing scale to dashboard needs — no 80px hero mega, nobody needs that on a filter screen)

| Token | Size | Weight | Tracking | Use |
|---|---|---|---|---|
| `title-lg` | 32px | Inter 400 | −0.4px | Page-level section headings |
| `title-md` | 18px | Inter 600 | 0 | Card titles, component titles |
| `title-sm` | 16px | Inter 600 | 0 | List labels, table headers |
| `body-md` | 16px | Inter 400 | 0 | Default body text |
| `body-sm` | 14px | Inter 400 | 0 | Secondary/dense body text |
| `caption` | 13px | Inter 400 | 0 | Timestamps, helper text |
| `caption-strong` | 12px | Inter 600, uppercase | 0 | Overlines, badge labels |
| `number-display` | 18px | JetBrains Mono 500 | 0 | Every stat, percentage, count |
| `button` | 14px | Inter 600 | 0 | Button labels |
| `nav-link` | 14px | Inter 500 | 0 | Top-nav items |

**Note:** design-2's body size was 15px; Coinbase's documented body-md is 16px. Going with 16px — it's Coinbase's spec and the 1px difference isn't worth fragmenting the scale over.

---

## Spacing

Coinbase's tokens, applied directly — this was a clean swap, both systems already used a 4px base unit:

| Token | Value |
|---|---|
| `xxs` | 4px |
| `xs` | 8px |
| `sm` | 12px |
| `base` | 16px |
| `md` | 20px |
| `lg` | 24px |
| `xl` | 32px |
| `xxl` | 48px |
| `section` | 96px *(marketing-band spacing — realistically unused inside the dashboard shell; keep for landing/onboarding screens only)* |

- **Container max width:** 1280px, 24px horizontal padding (kept from design-2 — Coinbase's ~1200px is close enough not to matter)
- **Card grid gap:** 20–24px (`md`–`lg`)
- **Component padding:** small 8×12 (`xs`×`sm`), medium 12×16 (`sm`×`base`), large 16×24 (`base`×`lg`)

---

## Elevation

Kept design-2's flat-card-plus-hover-lift model — Coinbase's own elevation system (hairline border / single soft-drop shadow / photographic depth) is marketing-hero-specific and doesn't map onto a card grid of stat panels. Only the *color* of the interactive glow shadow changed:

- Cards: flat, 1px `hairline` border, no default shadow. Hover: `0 8px 30px rgba(0,0,0,0.08)` + 2px lift, 200ms transition.
- Primary buttons hover: `0 4px 12px rgba(0,82,255,0.35)` — Coinbase Blue tint, was indigo.
- Focus ring: `0 0 0 3px rgba(0,82,255,0.12)` — Coinbase Blue tint, was indigo.
- Nav: backdrop-blur, no shadow.
- Dropdowns/popovers: shadow-lg (unchanged).

---

## Border Radius *(unchanged from design-2 — not requested to swap, and shouldn't be)*

| Value | Use |
|---|---|
| 4px | Tags, chips, badges, inline code |
| 6px | Buttons, inputs, selects |
| 8px | Metadata cards, dropdowns, panels |
| 12px | Stat/report cards, search bar, featured sections |
| 9999px | Avatars, status dots, pill badges |

Coinbase's actual scale — 100px pill CTAs, 24px card corners — was built for full-bleed marketing heroes and pricing tiers. Applied to a filter bar with 15 issue-tag chips and a dense review table, it reads as bulbous and undersells the information density. Not porting it.

---

## Components

### Buttons
Primary: `primary` fill, white text, 6px radius, `button` typography (Inter 600, 14px). Secondary: transparent bg, 1px `hairline` border, same radius. Ghost: no border/bg, text-color change on hover. Destructive: `semantic-down` text + border. All buttons shift up 1px on hover. Sizes: small 32px / medium 38px / large 44px.

### Cards
`canvas` surface, 1px `hairline` border, 12px radius, overflow hidden. Report cards: chart/sparkline area on top (fixed height), stat block below. Hover: 2px lift + shadow per Elevation above. 200ms transition.

### Inputs
1px `hairline` border, `canvas` background, 6px radius, 10×14px padding, 14px font. Focus: border turns `primary`, 3px ring `rgba(0,82,255,0.12)`. Error: border turns `semantic-down`. Placeholder text: `muted`.

### Chips
**Tag chips** (issue categories — neutral, not evaluative): pill shape, `surface-strong` bg, `body` text, 4×12px padding, 12px font. Active: `primary` bg, white text.
**Status chips** (sentiment/state — evaluative, semantic color required): same pill shape.
- Positive / Published → `semantic-up` tint bg, `semantic-up` text
- Neutral / Pending → `warning` tint bg, `warning` text
- Negative / Rejected → `semantic-down` tint bg, `semantic-down` text

### Lists
Stacked rows, 1px `hairline` dividers, flex space-between, 12×16px padding. Hover: `surface-soft` background.

### Toggles
20px, rounded-full, unchecked `hairline` gray, checked `primary` with white check. Used for dashboard preference switches.

### Navigation
Sticky top nav, backdrop-blur, 56px height, 1px `hairline` bottom border. Logo left, nav links center/hamburger on mobile, avatar dropdown right. Links: `nav-link` typography, hover shows `surface-strong` background, active state uses `primary` underline (not `primary` text-fill — keeps blue scarce per Coinbase's own "one or two blue moments per view" rule).

### Search
⌘K global search, rounded-xl (12px, not Coinbase's 100px pill) bar with icon + shortcut badge — internal radius scale stays consistent at 12px rather than jumping to Coinbase's pill geometry for one component.

### Tooltips
Native browser tooltips via `title` attribute — no custom component, unchanged from design-2.

---

## Dashboard-Specific Adaptations (tied to PROJECT.md)

- **Sentiment badge** — 3-state status chip per the Chips spec above (positive/neutral/negative), used in the review feed and category breakdown cards.
- **Issue tag chip** — renders the 15 KMeans cluster names (`sizing_and_fit`, `audio_and_music_quality`, `general_dissatisfaction`, etc., plus `other` for noise clusters 6/11) as neutral tag chips, never semantic-colored — they're categorical, not evaluative, and coloring them would visually imply a good/bad judgment that isn't there.
- **Confidence margin** — always `number-display` (JetBrains Mono). Default `muted` text color; switches to `warning` color when margin falls below the 0.30 asymmetric threshold documented in PROJECT.md, as a "low-confidence, consider excluding" visual flag — this reuses the repurposed warning token functionally, not decoratively.
- **Category breakdown card** — stat block per category: sentiment score, negative-trend delta (with up/down semantic color on the delta arrow), top issue tag (neutral chip), review volume — all numeric values in mono.
- **Trend chart** — positive line `semantic-up`, negative line `semantic-down`, neutral line `muted` gray. `primary` (Coinbase Blue) is never used as a data-series color — it's reserved for interactive UI only, straight from Coinbase's own hardest rule.
- **Filter bar** — date range, category, sentiment, issue tag, confidence-margin threshold. Built from the Inputs + Chips components above, 12px radius throughout, no pill geometry.

---

## Motion & Animation

Neither source doc specified this in usable depth — design-2 gave one number (200ms card-hover transition), Coinbase explicitly scoped animation out entirely. That's not enough for a dashboard where charts redraw on every filter change. Filled in below, split into general UI motion and chart-specific motion, built on Tailwind v4.1's real `@theme`/`--animate-*` syntax.

**Not verified against your actual charting library** (Recharts / Chart.js / D3 / other) — that lives in whatever your real architecture.md specifies, and I don't have it. The chart section below is written library-agnostic; the exact implementation of "morph old value into new value" differs by library and needs checking once you send that file over.

### Timing tokens
- `fast` — 150ms — micro-interactions (button press, checkbox toggle)
- `base` — 200ms — card hover/lift (matches design-2's existing number)
- `slow` — 300ms — modal/dropdown enter-exit, filter-triggered chart updates
- `chart` — 400–600ms — initial chart render, number count-up

### General UI animations
- **Hover/press micro-interactions** — transitions, not keyframe animations. `transform`/`opacity` only, `fast`–`base` duration, `ease-out`. This is just formalizing the card-lift and button-lift already defined under Elevation.
- **Modal/dropdown enter-exit** — fade + scale: opacity 0→1, scale 0.95→1, `base` duration.
- **Toast/inline alerts** (e.g. "low-confidence predictions excluded") — slide-in + fade, `base` duration, auto-dismiss.
- **Loading states** — skeleton pulse for cards/tables, spinner for inline actions. Tailwind's built-in `animate-pulse` / `animate-spin` — no custom keyframes needed here.
- **Route/page transitions — explicitly none.** Don't animate between dashboard views. On a data-dense screen this fights perceived load speed instead of helping it; it's a marketing-site instinct, not a dashboard one.

### Chart animations
- **Initial render** — data draws in over `chart` duration (400–600ms). If multiple series (positive/negative/neutral trend lines), stagger the draw-in rather than animating all three at once — reduces first-load visual noise.
- **Filter/data update** — existing values should morph to new values (bar height, line path) over `slow` duration, not fade-out-then-fade-in the whole chart. Confirm against your charting library — this is trivial in Recharts/D3, more constrained in some Chart.js setups.
- **Stat card count-up** — numbers (review volume, sentiment score, deltas) count up from previous value over `chart` duration on data change. Reinforces "this number just moved" on a dashboard that's full of deltas.
- **Sparkline draw-in** — `stroke-dashoffset` animation, 500ms `ease-out`, **mount only** — don't replay on every re-render, that's distracting rather than delightful.

### Reduced motion
`prefers-reduced-motion: reduce` disables count-up, draw-in, and stagger entirely — leaves only opacity fades under 150ms. Not optional.

### Tailwind v4.1 implementation
v4.1 has no `tailwind.config.js` by default — tokens and keyframes live directly in your CSS entry file via `@theme`, and every token is auto-exposed as a real CSS variable at runtime. Mapping this doc's tokens directly:

```css
@import "tailwindcss";

@theme {
  /* colors */
  --color-primary: #0052FF;
  --color-primary-active: #003ECC;
  --color-ink: #0A0B0D;
  --color-body: #5B616E;
  --color-muted: #7C828A;
  --color-hairline: #DEE1E6;
  --color-canvas: #FFFFFF;
  --color-surface-soft: #F7F7F7;
  --color-surface-strong: #EEF0F3;
  --color-semantic-up: #05B169;
  --color-semantic-down: #CF202F;
  --color-warning: #F4B000;

  /* typography */
  --font-sans: "Inter", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", monospace;

  /* spacing */
  --spacing-xxs: 4px;  --spacing-xs: 8px;   --spacing-sm: 12px;
  --spacing-base: 16px; --spacing-md: 20px; --spacing-lg: 24px;
  --spacing-xl: 32px;  --spacing-xxl: 48px;

  /* radius — design-2's scale, not Coinbase's */
  --radius-chip: 4px;
  --radius-control: 6px;
  --radius-panel: 8px;
  --radius-card: 12px;

  /* animation */
  --animate-fade-in-scale: fade-in-scale 200ms ease-out;
  @keyframes fade-in-scale {
    0% { opacity: 0; transform: scale(0.95); }
    100% { opacity: 1; transform: scale(1); }
  }

  --animate-sparkline-draw: sparkline-draw 500ms ease-out forwards;
  @keyframes sparkline-draw {
    from { stroke-dashoffset: var(--sparkline-length); }
    to { stroke-dashoffset: 0; }
  }
}
```

Count-up and chart-morph animations aren't pure-CSS `@theme` candidates — they need JS driving the interpolated value (framer-motion/Motion, or a small custom hook), since CSS can't tween a text node or a chart library's internal data state. Flagging that now so nobody tries to force it into a `--animate-*` variable and burns an afternoon on it.

---

## Do's and Don'ts

**Do**
- Keep `primary` (Coinbase Blue) exclusively for interactive elements — CTAs, links, focus rings, active nav state. Never decorative, never a data-series color.
- Render every numeric value — percentages, counts, confidence margins, deltas — in JetBrains Mono.
- Maintain the 4px spacing grid throughout.
- Use `semantic-up`/`semantic-down` only as text/tint color for sentiment and trend indicators, never as a solid button background (inherited directly from Coinbase's trading-semantics rule).

**Don't**
- Don't reintroduce Coinbase's pill (100px) or 24px card radius — wrong grammar for this density of screen.
- Don't add a second brand accent color beyond `primary` + the three semantic tokens.
- Don't color issue-tag chips semantically — they're categories, not judgments.
- Don't use pure black/white for text — stick to `ink` (#0A0B0D) and `canvas` (#FFFFFF) as defined.
- Don't forget the `warning` token is a repurposed, off-label use of Coinbase's illustrative yellow — if this ever needs to scale to a public marketing surface using the real Coinbase system, that reuse needs to be revisited, not copy-pasted in.
- Don't animate page/route transitions between dashboard views — kills perceived performance on a data-dense product.
- Don't replay sparkline draw-in or stagger animations on every re-render — mount-only, or it reads as glitchy rather than polished.
