# Paper Submission Recommendations

This note curates the final figure and table set for submission.

## Main-text figures

Recommended order:

1. `fig_official_ev_growth.png`
   Two-panel empirical context figure. Use early in the paper to establish the real EV adoption trajectory in Bogotá with official RDA checkpoints.

2. `fig_social_context_panels.png`
   Three-panel composite showing learned priority, social vulnerability, and protected geography. This should introduce the spatial logic before any allocation result is shown.

3. `fig_attention_profile.png`
   Two-panel temporal figure connecting the learned schedule pattern with vulnerability-sensitive attention. This supports the neural-learning contribution.

4. `fig_allocation_composite.png`
   Four-panel scenario comparison (`S0`-`S3`). This is the main spatial policy result and should appear before any inequality metric.

5. `fig_net_change_S2_minus_S1.png`
   Single-map reallocation figure. This is the clearest spatial interpretation of what the socially weighted policy changes relative to the plain equity policy.

6. `fig_lorenz_curves.png`
   Two-panel inequality figure. Keep in the main text because it summarizes distributional improvement in a standard transport-equity format.

7. `fig_policy_tradeoff.png`
   Single trade-off frontier between fidelity loss and vulnerable coverage gains. This is the best compact synthesis of the optimization argument.

## Supplementary figures

- `fig_growth_progression.png`
  Keep as supplementary because it expands the stage-to-stage spatial story but is redundant for the main narrative.

- `fig_gnn_learning_history.png`
  Keep as supplementary model-training evidence.

- `fig_robustness.png`
  Keep as supplementary sensitivity evidence unless the editor specifically asks robustness to be in the main text.

## Composite-figure count

Recommended composite figures in the main text: 5

- `fig_official_ev_growth.png` (2 panels)
- `fig_social_context_panels.png` (3 panels)
- `fig_attention_profile.png` (2 panels)
- `fig_allocation_composite.png` (4 panels)
- `fig_lorenz_curves.png` (2 panels)

Recommended composite figures in the supplement: 2

- `fig_growth_progression.png`
- `fig_robustness.png`

## Main-text tables

1. `table_scenario_design.csv`
   Use as the compact scenario-definition table. It explains each policy and makes clear whether the `S3` floor binds.

2. `table_equity_metrics_final.csv`
   Use as the principal numerical results table.

3. `table_official_ev_growth.csv`
   Use only if the journal prefers a small empirical context table next to the official growth figure; otherwise cite it in supplementary material.

4. `table_policy_tradeoff.csv`
   Use as a support table for `fig_policy_tradeoff.png` or move to supplementary material if the figure alone is sufficient.

## Supplementary tables

- `table_priority_summary.csv`
- `table_equity_metrics.csv`
- `table_robustness.csv`
- `table_allocation_changes_S2_vs_S1.csv`
- `table_gnn_performance_available.csv`
- `scenario_S0.csv`
- `scenario_S1.csv`
- `scenario_S2.csv`
- `scenario_S3.csv`

## Important editorial note

`S3` produces the same allocation as `S2` in the current reproducible run.
That means `fig_net_change_S3_minus_S2.png` is structurally empty and should not be used.
The correct interpretation is not that the figure failed, but that the soft-floor policy is non-binding under the reproduced solution.
