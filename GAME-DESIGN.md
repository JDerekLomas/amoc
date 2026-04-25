# HELM
### A planetary‑ocean simulator, played as a research vessel

> *The bridge of a science ship. The viewport is the sea. Outside the
> windows: an entire ocean, turning. You are not looking at a model. You
> are aboard the model. Every dial you touch, the planet feels.*

This is the design for the game layer that goes on top of `v4-physics/`.
The simulator already exists; its physics are real. What's missing is a
reason to come back, a reason to care about a particular gyre on a
particular Tuesday. That is what this document is for.

---

## 1. The pitch in one paragraph

You are the bridge officer of a science vessel that can travel through
deep time and counterfactual Earths. Each voyage drops you on a planet
in some specific climatic situation — pre‑Drake Antarctica, +5°C 22nd
century, Cryogenian snowball — with a small set of dials, a question to
answer, and a few model‑decades to answer it in. The ocean simulates
itself in real time on your console. The planet responds. The logbook
fills with the discoveries you make. Some voyages take three minutes;
some, three sittings. None of them are "won" with a high score; they're
won with an instrument reading that tells the planet's story.

---

## 2. The four pillars

### I. The planet is a character.
The ocean is not background; it is the protagonist. The game never tells
you "the AMOC has collapsed." It shows you that the deep‑water formation
gauge has gone slack, that the Greenland melt indicator is climbing,
that the North Atlantic ice tongue is creeping further south every
season. The planet narrates itself through instruments. The player's
job is to read it.

### II. The physics is real, the framing is dramatic.
We never lie about the model. The barotropic vorticity equation is
visible on the bridge. But we surround it with the language of voyages,
mission logs, captain's reports. The drama is in the *stakes* — your
North Atlantic just froze, what now — not in fictional flourishes
glued on top of the math.

### III. Failure is a finding.
A run that doesn't hit the goal still produces a logbook entry. *"Ship
log, Year 47: AMOC failed to recover. Hypothesis: friction was too low,
the gyre overshot. Next attempt: try r = 0.06."* The planet is not your
adversary; the equations are not optional rules; the wrong answer is
still data. We never bonk‑sound a player.

### IV. Every dial teaches a thing.
The campaign is structured so each mission introduces or interrogates
exactly one physical concept. Friction. β‑plane. Buoyancy. Albedo
feedback. Bifurcation. By the end of Act II the player has, without
ever being lectured, internalized the variables that drive Earth's
climate. The game *is* the curriculum.

---

## 3. The core loop

```
       ┌──────────────────────────────────────────┐
       ▼                                          │
  BRIEFING ──→ HELM ──→ OBSERVATION ──→ REPORT ───┘
   (read     (turn     (watch the      (logbook
   the       dials,    planet          entry +
   stakes)   pause/    respond over    next
             paint)    model years)    voyage)
```

Each lap takes 3–10 minutes. The Briefing is a single page. The Helm
is the existing console. Observation is the live sim. The Report is a
sketched logbook spread that summarizes the run, awards stars, and
cues the next voyage.

---

## 4. Three modes

The product offers three doors in. Choose one, all use the same engine
and the same console:

### MODE A — VOYAGES (the campaign)
A linear‑ish progression of curated missions. ~24 voyages across four
acts. New tools and views unlock as the player progresses. This is the
default front door. **Sole source of truth for what the game teaches.**

### MODE B — SANDBOX (the existing app, dressed for guests)
No goals, every dial unlocked, paint anywhere. The current
`console.html` is already this. The only additions are *Save & Share*
(serialize a planet to a URL) and *Photo Mode* (export an annotated
screenshot — geometry, params, season, key diagnostics).

### MODE C — DAILY ANOMALY (small ritual + leaderboard)
Once a day, a generated planet with one of its baseline parameters
mutated. The player has 3 minutes to identify which dial moved and by
how much, by reading the instruments. Submission goes to a leaderboard.
This is the "Wordle of climate."

The campaign is the heart. The sandbox is the lobby. The daily is the
hook for return visits.

---

## 5. The campaign — Voyages

24 voyages in 4 acts of 6 each. Designed so a curious adult can clear
Act I in a single sitting, and the full campaign in 4–8 hours. Three
star ratings per voyage:

| ★      | meaning                                                      |
|--------|--------------------------------------------------------------|
| ☆      | Goal hit. Voyage complete.                                   |
| ★★     | Hit within an "elegant" margin (e.g. <30 model years, fewer scenarios used). |
| ★★★    | Bonus condition discovered (a hidden criterion described in the post‑report). |

Three‑star runs unlock the *Deep Logbook* entries — long‑form
explanatory essays that go well beyond the surface mission text.

### Act I — Foundations *(the engine teaches itself)*

Tools unlocked at start: **Wind, Friction, Viscosity, Reset, Pause.**
Views: **Streamfunction, Speed.**

**1. First Light**
> *A featureless ocean. A breath of wind. Before anything else, the
> ocean wants to spin.*

- **Setup:** Empty rectangular basin (no real continents). Single
  hemisphere wind pattern. T = 0.
- **Goal:** Achieve Kinetic Energy ≥ 1×10⁴ within 50 model years.
- **Bonus:** Achieve a single‑gyre circulation pattern (hidden test:
  ψ has only one extremum).
- **Concept introduced:** wind‑driven gyres.
- **Logbook entry on win:** *"Sverdrup, 1947 — the first great
  diagnostic of the wind‑driven ocean."*

**2. Western Wall**
> *The current is fast on one side. The other side is just... slow.
> Why?*

- **Setup:** Same basin, same wind. Friction slider only.
- **Goal:** Make the boundary current on the western side at least
  3× faster than the eastern side.
- **Bonus:** Match a target Stommel boundary‑layer width within 5%.
- **Concept:** β effect → western intensification.
- **Reveal:** *Stommel, 1948.*

**3. Two Speeds**
> *The ocean has memory. Push it, then let it drift.*

- **Setup:** Realistic wind. Year‑speed dial unlocked.
- **Goal:** Run for 100 model years without intervention; show the
  player that gyres pulse with seasons.
- **Bonus:** Identify the lag (ocean response to wind: months, not
  days).
- **Concept:** seasonality, thermal inertia.

**4. The Atlantic, Sketched**
> *Every continental edge bends the water.*

- **Setup:** Real continents, only the Atlantic basin is "lit" — rest
  of ocean masked. Player paints a single coastline correction.
- **Goal:** Reproduce the Gulf Stream's western tongue (validated by
  comparing the simulated streamfunction to a stored reference).
- **Concept:** geometry as boundary condition.

**5. Many Gyres**
> *Open the Pacific. Watch the ocean fill with eddies.*

- **Setup:** Whole world unmasked. Particles unlocked.
- **Goal:** Identify and tag five subtropical gyres in the
  streamfunction view (a click‑to‑label puzzle layered onto the
  canvas).
- **Concept:** zonal symmetry of large‑scale circulation.

**6. The Deep Beneath**
> *Below the wind‑lit ocean, another ocean turns.*

- **Setup:** Deep‑Flow view unlocked.
- **Goal:** Run the world for 100 model years and observe the deep
  layer spinning up. Take a photo (forced action: must press Photo
  Mode to complete).
- **Concept:** two‑layer model; the deep ocean is *slow*.
- **Act I clear:** unlocks *Thermodynamics* dials.

### Act II — The Modern Conveyor *(building today's ocean)*

Tools unlocked: **Solar Constant, Radiative Forcing, Year Speed.**
Views: **Temperature, Deep Temperature.**

**7. Tropics & Poles**
> *The sun does not heat the ocean evenly. From this, everything follows.*

- **Setup:** Modern continents, modern winds.
- **Goal:** Achieve a stable equator‑to‑pole SST gradient of ≥ 25°C
  within 200 model years.
- **Concept:** radiative balance, thermal advection.

**8. The Frozen North**
> *Sometime in the model‑year, the Arctic should freeze. If it doesn't,
> something is off.*

- **Setup:** Modern Earth. Year Speed unlocked.
- **Goal:** Produce visible seasonal Arctic sea‑ice expansion (ice
  area pulsing with Northern winter).
- **Concept:** ice‑albedo feedback.

**9. The Conveyor**
> *Listen to the AMOC.*

- **Setup:** Modern Earth, no perturbations.
- **Goal:** Bring AMOC to "STRONG" status and hold it for 50 model
  years.
- **Bonus:** Identify which two parameters most affect AMOC strength
  (mini quiz at end).
- **Concept:** thermohaline circulation.

**10. Drake Opens** *(scenario)*
> *34 million years ago, Antarctica isolates. The world cools.*

- **Setup:** Closed Drake Passage starting state. Scenario card:
  *Open Drake Passage.*
- **Goal:** After opening, sustain a circumpolar zonal current at
  ≥ X m/s for 50 years.
- **Concept:** the Antarctic Circumpolar Current and Cenozoic cooling.

**11. Panama Closes** *(scenario, reversed)*
> *Three million years ago, the Atlantic and Pacific stopped talking.*

- **Setup:** "Open Panama" preset (counterfactual ancient state).
  Scenario card: *Close Panama.*
- **Goal:** Show that AMOC strengthens after closure (a before/after
  comparison automatically captured).
- **Concept:** geometry → conveyor.

**12. The Shape of the Modern Ocean**
> *You've built it. Now compare it.*

- **Setup:** Whole world, modern. The game overlays a transparent
  "reference" image (NOAA OISST) on the temperature view.
- **Goal:** RMSE of player's simulated SST vs reference < 4°C.
- **Bonus:** RMSE < 2°C.
- **Concept:** model validation; this is what real climate scientists
  do every day.
- **Act II clear:** unlocks *Forcings* (Freshwater, Greenland melt).

### Act III — Forcing the System *(the climate-change arc)*

Tools unlocked: **Freshwater Forcing, Radiative Forcing range
expanded.** Views: **Salinity, Density.**

**13. Greenhouse**
> *Push the dial. The world warms. What else?*

- **Setup:** Modern Earth. Radiative forcing slider.
- **Goal:** Add +4°C of equilibrium forcing and run to a new
  equilibrium. Observe and tag *three* downstream effects (ice
  retreat, gyre poleward shift, AMOC weakening).
- **Concept:** climate sensitivity.

**14. The Faint Sun**
> *Earth's first ocean lived under a 30%‑dimmer sun. How did it not
> freeze?*

- **Setup:** Solar constant set to 70%.
- **Goal:** Find the snowball threshold — the solar constant at which
  the ice line runs away to the equator.
- **Concept:** runaway ice‑albedo feedback (Budyko's bifurcation).

**15. Stommel's Cliff**
> *Push the freshwater. Slowly. Until something breaks.*

- **Setup:** Modern Earth, AMOC strong.
- **Goal:** Find the critical freshwater forcing F* at which AMOC
  collapses. Hold the planet there for one model year, then release —
  observe hysteresis (it doesn't return at F < F*).
- **Concept:** Stommel 1961, the classic two‑box bifurcation.
- **Logbook entry:** *"This is the most studied tipping point in
  climate science."*

**16. Younger Dryas**
> *12,800 years ago, a meltwater pulse turned off the Atlantic
> conveyor. The North froze for a thousand years.*

- **Setup:** Late‑Pleistocene baseline. A pre‑scripted freshwater
  pulse fires at year 5.
- **Goal:** *Restart* AMOC after the pulse using only the available
  tools (no rewinds).
- **Concept:** abrupt climate change; the historical analogue for
  collapse.

**17. The Anthropocene Drill**
> *2100. The melt is real. Make a call.*

- **Setup:** Modern Earth + scripted ramp of freshwater forcing
  modeling Greenland melt over 80 model years.
- **Goal:** Keep AMOC ≥ "WEAK" through model‑year 80, using only the
  tools available to a 21st‑century policymaker (radiative forcing
  reduction, no other knobs).
- **Concept:** the IPCC sketch of the world as a constrained sim.

**18. The 50/50 Planet**
> *Half land, half ocean, evenly split — what would it look like?*

- **Setup:** Sandbox‑painted preset showing one big ocean, one big
  continent. Free reign.
- **Goal:** Build a habitable planet — global SST in [10, 20]°C with
  AMOC ≥ "WEAK" — sustained for 50 model years.
- **Concept:** the planet design problem.
- **Act III clear:** unlocks *Free paint* and *Custom mission editor*.

### Act IV — Counterfactuals *(the toy box)*

Tools unlocked: **Everything. Free paint everywhere. Mission editor.**

**19. Snowball**
> *The Cryogenian, 700 Mya. Ice to the equator. How do you get out?*

- **Setup:** Frozen planet preset.
- **Goal:** Escape snowball within 200 model years using only solar
  forcing (analog to volcanic CO₂ buildup).
- **Concept:** snowball Earth deglaciation.

**20. Hothouse**
> *The Eocene optimum, 50 Mya. No polar ice anywhere.*

- **Setup:** Eocene continents preset (sketched).
- **Goal:** Eliminate all sea ice and run for 100 stable model years.
- **Concept:** the equable‑climate problem.

**21. Aqua Mundi**
> *No continents. Just water.*

- **Setup:** Painted‑clean planet, no land. Real winds.
- **Goal:** Observe and tag the three latitudinal flow bands that
  emerge.
- **Concept:** zonal symmetry, what continents *break*.

**22. Tidally Locked**
> *Permanent dayside, permanent night. An exoplanet's ocean.*

- **Setup:** Custom solar field — only one hemisphere lit.
- **Goal:** Identify the heat‑transport pattern that emerges from
  permanent insolation asymmetry.
- **Concept:** exoplanet oceanography.

**23. Reverse Earth**
> *What if the continents were exactly mirrored?*

- **Setup:** Mask flipped longitudinally.
- **Goal:** Compare AMOC strength between mirrored and normal Earth.
  Hypothesis test.
- **Concept:** path dependence of climate.

**24. The Captain's Choice**
> *You have the helm. Build the planet you want to live on.*

- **Setup:** Blank slate.
- **Goal:** Pass the *Habitable Planet Audit*: SST 10–20°C globally,
  AMOC ≥ "WEAK", seasonal ice, equator‑pole gradient ≥ 20°C, no
  runaway feedback over 200 years.
- **Concept:** every concept in the game, used together.
- **Reward:** the planet you built becomes shareable as a permanent
  URL. Other players can load and play it as a custom mission.

---

## 6. Meta‑progression: the Logbook

The Logbook is a leather‑bound journal in the bridge UI. Every voyage
adds at least one entry. There are three kinds of entry:

1. **Field notes** (1 per voyage) — the player's run summarized: what
   they did, what the planet did, the final readings. Auto‑generated
   from the run trace.
2. **Concept entries** (1 per voyage on first clear) — short
   illustrated essays on the physics: *Western Intensification*,
   *Ice‑Albedo Feedback*, *The Conveyor*, *Stommel's Bifurcation*…
3. **Deep entries** (3‑star unlock only) — long‑form pieces, ~600
   words, with citations. *"Why is the Gulf Stream there and not
   somewhere else? — A historical reading of Stommel 1948."*

The Logbook is also the **navigation hub**: voyages 7–24 are launched
from inside it, by selecting an unlocked entry and tapping "Replay."

---

## 7. Onboarding — the first 5 minutes

Critical. Almost everything that fails in pedagogical games fails here.
Walk through, second by second:

**0:00 — Splash.** The masthead and onboarding overlay we already
designed: *"Take the helm of a planet."* Single button: *Begin
Observation.*

**0:05 — Bridge with one dial lit.** All controls are dimmed except
the *Wind Strength* dial (which softly pulses). HUD note in italic
serif at the bottom of the viewport: *"Try the Wind dial."*

**0:15 — Player moves the wind dial.** A pause hits at the moment
they let go. The viewport shows particles in the empty ocean,
beginning to trace lines. Note: *"Now press play to let the planet
respond."*

**0:25 — Sim runs.** Particles trace gyres. As soon as KE crosses
1e4 (~30 seconds of real time at default speed), a gentle bell, the
viewport flashes once with a brass border, and the bridge auto‑pauses.

**~1:00 — Mission complete.** Card slides up from below: *"Voyage I
complete: First Light. ★. The ocean wanted to spin. You found it."*
A *Continue* button. The continue button leads into Voyage 2: Western
Wall — and now the *Friction* dial is lit.

By minute 3 the player has done two voyages, used two dials, and seen
two patterns emerge. By minute 5 they're starting Voyage 3 with a
sense that they understand the simulator. They have not been told a
single fact. They've inferred several.

**The onboarding is the design philosophy in miniature: every dial
revealed by the situation that needs it.**

---

## 8. The Daily Anomaly

Every day at 00:00 UTC, the server (or static seeded RNG) chooses one
parameter from a fixed set and shifts it by a known amount. The player
loads in:

- A planet at equilibrium with the perturbed parameter.
- A target streamfunction / temperature view shown side‑by‑side with
  the player's planet at year 0.
- 3 minutes of real time.
- A small panel with the seven possible dials grayed out — the player
  selects the one they think changed and a magnitude.

Score:
- **Correct dial + magnitude within 10%** → 100 points.
- **Correct dial, wrong magnitude** → 50 points.
- **Wrong dial** → 0, but a free Logbook entry of which dial it
  *actually* was.

Daily streaks. A tiny badge on the masthead when you've played today.
A leaderboard that uses the existing `tournament.mjs` infrastructure
under the hood.

---

## 9. UI surfaces to design

We already have the **Bridge** (`console.html`). The game layer needs
five additional surfaces, all in the same captain's‑bridge aesthetic:

### 9.1 Voyage Briefing card

A page from the captain's logbook. Shown between voyages. Layout:

```
┌──────────────────────────────────────────────────┐
│  VOYAGE IX · MISSION BRIEFING                    │
│                                                  │
│  THE CONVEYOR                                    │
│                                                  │
│  Listen to the AMOC.                             │
│                                                  │
│  ─── stakes ─────────────────────────────────    │
│  Modern Earth. No perturbations. Hold the         │
│  Atlantic Meridional Overturning at "STRONG"     │
│  for 50 model years.                             │
│                                                  │
│  ─── tools available ────────────────────────    │
│  Wind · Friction · Viscosity · Year Speed        │
│  Solar · Radiative Forcing                       │
│                                                  │
│  ─── reading list ────────────────────────────   │
│  · Stommel 1961 · Broecker 1991                  │
│                                                  │
│              [ TAKE THE HELM ]                   │
└──────────────────────────────────────────────────┘
```

Type: Cormorant for body, Cormorant SC for section heads, brass corner
brackets, a single decorative woodcut illustration where appropriate
(SVG, not photographic — this is a logbook, not a National Geographic).

### 9.2 Goal HUD on the bridge

While a voyage is active, a thin strip lives at the top of the
viewport:

```
  GOAL · AMOC HOLD ≥ STRONG · 50 yr ──────────  [ ████░░░░░ ] 23 / 50 yr
```

Brass on dark, small caps, JetBrains Mono numeric. The progress bar
fills only when the goal predicate is currently true; if AMOC drops
below STRONG the bar stops advancing (and shows a pulsing amber dot at
its right edge to signal "off‑target").

### 9.3 End‑of‑voyage Report

A two‑page logbook spread, post‑voyage. Left page: the run summary,
including a small zonal‑mean SST plot and a sparkline of the goal
diagnostic. Right page: the awarded stars, what they meant, and the
unlocked Logbook entry. Big *Continue* / *Replay* buttons at the
bottom.

### 9.4 The Logbook itself

Full‑screen view of the journal. Tabs: *Voyages*, *Concepts*, *Deep*.
Voyages is the campaign list with star ratings. Concepts is the
unlocked physics essays. Deep is the long‑form pieces. Visual: like
flipping through a real notebook — the next voyage shows as a sealed
envelope until unlocked.

### 9.5 Daily Anomaly screen

Compact. Three panes:
- The reference planet at top.
- The mystery planet beneath it.
- A diagonal "diff" view in the middle, color‑coded.
Below: dial selector + magnitude slider + submit. Timer in the
masthead clock, replacing the model‑year display.

---

## 10. Diegetic framing — the world you're in

Lean lightly into a world. We don't need a deep fictional universe; we
just need the *feeling* that the bridge is real.

- **The Vessel:** *RV Conveyor.* It does not need a captain's name.
  *You* are the captain.
- **The Logbook:** in the player's hand. Annotated by a previous
  captain in faint marginalia ("Stommel saw this before us. — A.B.,
  1961"). The marginalia are an opportunity for sparse, lovely
  citations.
- **The Voice:** the briefings are written in second person, present
  tense, short sentences. Never sci‑fi. Never goofy. Think *The
  Sea Around Us* (Rachel Carson) or *Annals of the Former World*
  (John McPhee). Lyric, but plain.
- **No characters with dialogue.** The bridge is empty except for
  the player. The planet speaks through instruments. The previous
  captains speak through the marginalia. That's it.

---

## 11. Things we are NOT doing

Worth declaring, because it focuses the rest:

- **No conflict mechanic.** No enemies. No competing factions. The
  ocean is not your adversary; the equations are not optional rules.
- **No currency, no shop, no unlocks behind grind.** Unlocks are
  through *learning*, not repetition.
- **No real‑time multiplayer.** The Daily Anomaly is async; that's the
  social surface.
- **No fictional physics.** Every dial reflects something the model
  actually does. We never pretend a knob does more than it does.
- **No "win the game."** Voyage 24 is a capstone, not a credits roll.
  The sandbox is always there.

---

## 12. The first thing to build

If we want to go from this design to a playable v0 with the smallest
possible scope:

1. **One voyage.** Voyage 1: First Light.
2. **The Goal HUD strip** at the top of the viewport.
3. **The end‑of‑voyage Report card** (just left page — the right page
   can wait).
4. **One Logbook entry** ("Sverdrup, 1947").
5. **A "Voyages" button** in the masthead that, for now, does nothing
   except restart Voyage 1.

That is shippable in a day. It will tell us:
- Whether the soft pause‑on‑success feels good.
- Whether players read the briefing or skip it.
- How long Voyage 1 actually takes (target: 90 seconds).
- Whether the diagnostic predicate (`KE ≥ 1e4`) is the right shape
  for a goal.

If the answers are good, Voyage 2 is half a day; the entire Act I is
maybe a week. The campaign is a 1–2 month effort if pursued seriously.

---

## 13. Risks & open questions

**Risk: the player reaches Voyage 9 and the briefing language starts
sounding repetitive.** Mitigation: every voyage is written by a real
human, not procedurally. They are short.

**Risk: the simulation is slow on weak hardware, the goals time out.**
Mitigation: goals are stated in *model years*, not real seconds.
Progress bars use simulated time. A weak laptop runs the same voyage
in more wall‑time but the same number of model years.

**Risk: the "build a habitable planet" capstone is too unconstrained
and players give up.** Mitigation: the *Habitable Planet Audit* has
sub‑goals lit in sequence. Hit equator‑pole gradient first. Then ice.
Then AMOC. Each lights a green pip in the audit panel.

**Open question: should the campaign require Act II before Act III, or
let players skip ahead?** Current call: Act I gates Act II, Act II
gates Act III, Act III gates Act IV. But every individual voyage
within an act is optional; you only need to clear *any* 3 of the 6 to
unlock the next act.

**Open question: do we want a narrator's voice somewhere — written
captain's‑log entries between voyages?** Tempting. Tentative no — the
marginalia in the Logbook does this job without imposing a voice on
the player's experience.

**Open question: educator mode?** A teacher loads a class roster, each
student plays a curated path, the teacher sees a dashboard. Worth
prototyping after the campaign exists.

---

## 14. Why this design will work

Three reasons:

1. **The simulator is already astonishing.** Most games have to
   *manufacture* a moment of "whoa, look at that." This one has it
   built in: the first time a gyre forms from nothing, the first time
   AMOC collapses and the North Atlantic freezes — those are
   irreducibly real. Our job is to *get out of the way of those
   moments* and frame them.

2. **The science is structured for play.** The barotropic vorticity
   equation is exactly the right size to teach in a campaign: it has
   ~7 parameters, each of which corresponds to something visible. It
   is not too simple (one gyre, one knob) and not too complex (a full
   GCM). The educational arc has a natural shape.

3. **The captain's‑bridge metaphor isn't decoration.** It is a
   *discipline*: every UI decision is constrained by "would this be
   on a research vessel?" That constraint forces tasteful answers.
   The fictional frame protects the design from drifting into
   generic god‑game flailing.

---

*Last revised: 2026‑04‑25.*
*Companion to `v4-physics/console.html`. The game is the campaign
running on top of that shell.*
