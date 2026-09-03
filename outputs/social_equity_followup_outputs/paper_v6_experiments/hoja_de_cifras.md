# Hoja de cifras — tablas: /Users/jorgeiv500/Documents/social-eq/outputs/social_equity_followup_outputs/paper_v6_experiments/tables

| Escenario | pérdida fidelidad % | Gini | Gini pond. | cobertura vulnerable | share cuartil | media V^H | ρ prior-puertos | zonas con puertos |
|---|---|---|---|---|---|---|---|---|
| S0 | 0.0 | 0.170368 | 0.165051 | 1.0000 (36/36) | 0.280019 | 460.833263 | 0.925629 | 141 |
| S1 | 3.0 | 0.170141 | 0.164848 | 1.0000 (36/36) | 0.279933 | 460.297306 | 0.925807 | 141 |
| S2 | 8.0 | 0.169942 | 0.164653 | 1.0000 (36/36) | 0.279822 | 459.832070 | 0.925967 | 141 |
| S3 | 8.0 | 0.169941 | 0.164654 | 1.0000 (36/36) | 0.279813 | 459.776792 | 0.925777 | 141 |

u* = 108.833700
- S1_vs_S0: 21 zonas cambian, 11 puertos movidos, máx |Δ| 2; ganan ['UTAM101', 'UTAM15', 'UTAM16', 'UTAM109', 'UTAM64', 'UTAM65', 'UTAM20', 'UTAM97', 'UTAM98', 'UTAM590', 'UTAM620']; pierden ['UTAM45', 'UTAM49', 'UTAM23', 'UTAM60', 'UTAM61', 'UTAM25', 'UTAM28', 'UTAM580', 'UTAM660', 'UTAM710']
- S2_vs_S1: 16 zonas cambian, 8 puertos movidos, máx |Δ| 1; ganan ['UTAM27', 'UTAM64', 'UTAM13', 'UTAM25', 'UTAM97', 'UTAM40', 'UPR2', 'UTAM571']; pierden ['UTAM114', 'UTAM104', 'UTAM10', 'UTAM48', 'UTAM57', 'UTAM84', 'UTAM87', 'UTAM640']
- S3_vs_S2: 6 zonas cambian, 3 puertos movidos, máx |Δ| 1; ganan ['UTAM104', 'UTAM84', 'UTAM590']; pierden ['UTAM55', 'UTAM56', 'UTAM62']
- S2_vs_S0: 33 zonas cambian, 18 puertos movidos, máx |Δ| 2; ganan ['UTAM101', 'UTAM27', 'UTAM15', 'UTAM16', 'UTAM109', 'UTAM64', 'UTAM65', 'UTAM20', 'UTAM13', 'UTAM97', 'UTAM98', 'UTAM40']; pierden ['UTAM114', 'UTAM45', 'UTAM49', 'UTAM23', 'UTAM104', 'UTAM60', 'UTAM61', 'UTAM10', 'UTAM48', 'UTAM57', 'UTAM28', 'UTAM84']
- robustez: rank stability 1.0000–1.0000
  - Low decay (all lambda x 0.75): ΔGini_w +0.025155, Δcob +0.0000, rank 1.0000
  - High decay (all lambda x 1.25): ΔGini_w -0.017081, Δcob +0.0000, rank 1.0000
  - PCA weights for vulnerability index: ΔGini_w -0.000012, Δcob +0.0000, rank 1.0000
  - Alternative epsilon tolerance (0.056): ΔGini_w +0.000082, Δcob +0.0000, rank 1.0000
  - Alternative epsilon tolerance (0.080): ΔGini_w +0.000000, Δcob +0.0000, rank 1.0000
  - Alternative epsilon tolerance (0.100): ΔGini_w -0.000069, Δcob +0.0000, rank 1.0000
  - Alternative vulnerability emphasis kappa (2.00): ΔGini_w -0.000012, Δcob +0.0000, rank 1.0000
  - Alternative vulnerability emphasis kappa (4.00): ΔGini_w +0.000000, Δcob +0.0000, rank 1.0000
  - Alternative vulnerability emphasis kappa (6.00): ΔGini_w +0.000007, Δcob +0.0000, rank 1.0000
  - Soft floor gamma (0.65): ΔGini_w +0.000000, Δcob +0.0000, rank 1.0000
  - Soft floor gamma (0.80): ΔGini_w +0.000000, Δcob +0.0000, rank 1.0000
  - Soft floor gamma (0.95): ΔGini_w +0.000001, Δcob +0.0000, rank 1.0000
  - Alternative travel-time matrix: ΔGini_w +nan, Δcob +nan, rank nan