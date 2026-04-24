# Clouds view: proper cloud physics + smooth rendering

## Problem
The Clouds view shows a noisy latitude gradient that looks like rendering artifacts, not clouds. Real cloud maps have smooth fields with sharp boundaries (ITCZ, stratocumulus decks off California/Peru/Namibia).

## Current Implementation
Cloud fraction is a simple diagnostic formula:
```
cloudFrac = 0.25 + 0.15*cos(2*lat) + 0.15*clamp((SST-15)/15) + 0.10*clamp((|lat|-50)/30)
```
- No moisture budget
- No vertical motion / convection dynamics
- No marine stratocumulus (which forms over COLD water with warm air above — our formula does the opposite)
- Per-pixel fillRect rendering is slow and noisy
- Cloud haze overlay on Sea Surface view is distracting — should be removed or made toggle-able

## What Good Looks Like
- Subtropical stratocumulus decks (off California, Peru, Namibia) — low clouds over cold upwelling water
- ITCZ cloud band — deep convection near the equator
- Storm tracks at 40-60° — midlatitude cyclone cloud bands
- Clear skies over deserts (Sahara, Arabian, Australian interior)
- Smooth field, not pixelated noise

## Proposed Fix
1. **Smooth the cloud field** — apply a box filter before rendering, or compute at lower resolution and upscale
2. **Add subsidence-driven stratocumulus** — high cloud fraction where SST is cold but air above is warm
3. **Add moisture proxy** — evaporation ~ wind * (SST_sat - humidity), clouds form where moisture converges
4. **Render as smooth overlay** — use ImageData with bilinear interpolation instead of per-pixel fillRect
5. **Remove cloud haze from Sea Surface view** — or make it a toggle
