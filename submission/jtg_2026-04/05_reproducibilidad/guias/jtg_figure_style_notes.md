# Journal of Transport Geography figure style notes

Reference paper reviewed:
`1-s2.0-S0966692326001201-main.pdf`

## Transferable visual rules

1. Prefer horizontal compositions.
Maps and temporal plots work better as wide panels or small multiples laid out left-to-right, not stacked vertically.

2. Rotate elongated geography to use the canvas efficiently.
Bogota should be shown horizontally so the reader can compare spatial patterns without wasting vertical space.

3. Use restrained editorial styling.
White or near-white panels, light dashed y-gridlines, a visible top border, limited axis clutter, and legends/colorbars placed below the panels.

4. Use a warm sequential map palette and a soft diverging delta palette.
Low values should remain dark and quiet; high values should move toward warm coral/sand tones.

5. Separate context from intervention.
One figure should establish priority and vulnerability geography; another should show scenario allocation; another should show change over time or policy stages.

6. For schedules, use time-window shading.
Residential, workplace, and public windows should appear as soft vertical bands behind the curves.

## Implemented in this repo

Code path:
[`src/social_eq/followup.py`](/Users/jorgeiv500/Documents/social-eq/src/social_eq/followup.py)

New or restyled outputs:

- `fig_priority_map.png`: wide, rotated Bogota priority map
- `fig_social_context_panels.png`: priority + vulnerability + protected geography
- `fig_attention_profile.png`: horizontal two-panel schedule figure with shaded time windows
- `fig_allocation_composite.png`: S0-S3 allocation small multiples
- `fig_growth_progression.png`: stage-to-stage infrastructure growth/reallocation maps
- `fig_official_ev_growth.png`: official citywide EV adoption checkpoints with annual additions
- `fig_lorenz_curves.png`: editorial restyle of equity curves
- `fig_policy_tradeoff.png`: fidelity-versus-coverage frontier across scenarios
- `fig_robustness.png`: two-panel robustness dot/lollipop layout

## Optional annual growth input

If an annual UTAM dataset is provided, the exporter will use it instead of policy-stage deltas for the growth map.

Accepted optional filenames under `data/inputs/`:

- `annual_growth_by_utam.csv`
- `utam_annual_growth.csv`
- `growth_by_year_utam.csv`

Expected schema:

- long format: `UTAM`, `year`, one numeric value column
- or wide format: `UTAM`, `year_2023`, `year_2024`, ...

If no annual file is present, the code falls back to policy-stage growth (`S1-S0`, `S2-S1`, `S3-S2`, `S3-S0`).

## Official citywide EV growth input

The repo now ships with `data/inputs/bogota_official_ev_growth.csv`, a citywide series built from official Bogotá / SDM / OMB checkpoints.

This file is not a spatial UTAM series.
It supports a separate figure for observed EV adoption over time, while the spatial growth panels still rely on scenario deltas unless a UTAM-by-year file is provided.
