# Envío a *Journal of Transport Geography* — JTRG-D-26-00692

**Manuscrito:** *Public electric-vehicle charging and accessibility inequality in Bogotá: A socio-spatial allocation framework*

| Elemento | Valor |
|---|---|
| Número de manuscrito | JTRG-D-26-00692 |
| Tipo | Full Length Article |
| PDF compilado | 2026-04-17 18:40 UTC (pdfTeX 1.40.27, TeX Live 2025) |
| Enviado a la revista | 2026-04-17 |
| Figuras y tablas | 2026-04-17 |
| Corrida de escenarios S0–S3 | 2026-04-15, Pyomo + NEOS/CPLEX |
| Zonas | 142 UTAM |
| Fuente LaTeX | `00_latex_fuente/`, exportado de Overleaf |

Esta carpeta es el **registro único** de lo enviado. Todo lo demás bajo
`outputs/social_equity_followup_outputs/paper_v6_experiments/` es material intermedio o superado.

---

## 0. Fuente LaTeX (`00_latex_fuente/`)

Proyecto de Overleaf (`final_submission_package`): `main.tex`, `supplementary_appendix.tex`,
`elsarticle.cls`, el `.bst`, las nueve figuras, la portada, el `.bib` y los *highlights*.
`main.tex` corresponde al PDF enviado; su `README.md` detalla cómo compilar y tres
advertencias sobre archivos del paquete que no coinciden con el envío.

## 1. Manuscrito (`01_manuscrito/`)

| Archivo | Corresponde a |
|---|---|
| `00_editorial_manager_JTRG-D-26-00692.pdf` | PDF armado por la revista, 19 pp.: es el registro de lo que efectivamente recibió JTG |
| `01_manuscript.pdf` | Cuerpo del artículo, 13 pp., anonimizado |
| `02_supplementary_appendix.pdf` | Apéndices A (notación GNN), B (notación optimización), C (figuras adicionales) |
| `03_title_page_author_details.pdf` | Portada con autores, filiaciones, CRediT, conflictos, financiación |
| `04_cover_letter_to_editor.pdf` | Carta al editor con el enlace al repositorio |

Composición del PDF de Editorial Manager: p. 1 ficha de la revista, p. 2 carta,
pp. 3–4 portada y CRediT, pp. 5–17 manuscrito, p. 18 enlace al suplemento,
p. 19 *Research Data* apuntando a `https://github.com/jorgeiv500/social-eq`.

## 2. Figuras del texto principal (`02_figuras_texto_principal/`)

| Nº en el paper | Archivo | Contenido |
|---|---|---|
| Figura 1 | `fig_official_ev_growth.png` | Puntos de control oficiales de VE en Bogotá y crecimiento interanual |
| Figura 2 | *(no hay PNG)* | Diagrama TikZ de etapas A–D, generado dentro del LaTeX |
| Figura 3 | `fig_social_context_panels.png` | Prioridad aprendida, vulnerabilidad social y geografía protegida |
| Figura 4 | `fig_attention_profile.png` | Estructura de atención temporal del ST-GNN |
| Figura 5 | `fig_allocation_composite.png` | Mapas de asignación S0–S3 — **versión del 2026-04-15** (5134 × 869), no la que regenera el pipeline |
| Figura 6 | `fig_net_change_S2_minus_S1.png` | Reasignación neta S2 menos S1, con recuadro de foco urbano |
| Figura 7 | `fig_lorenz_curves.png` | Curvas de Lorenz sin ponderar y ponderadas por vulnerabilidad |
| Figura 8 | `fig_policy_tradeoff.png` | Síntesis del compromiso de política, cuatro paneles a–d |

## 3. Figuras del suplemento (`03_figuras_suplemento/`)

| Nº en el suplemento | Archivo | Contenido |
|---|---|---|
| Figura 1 | `fig_growth_progression.png` | Mapas de progresión entre etapas de asignación |
| Figura 2 | `fig_robustness.png` | Chequeos de robustez alrededor de S2 |

Generadas por el pipeline pero **no incluidas** en el envío:
`fig_gnn_learning_history`, `fig_priority_map`, `fig_S3_protected_set`,
`fig_allocation_S0`–`fig_allocation_S3`. Quedan en `paper_v6_experiments/figures/`.

Ocho de las nueve figuras enviadas son del 2026-04-17. La excepción es
`fig_allocation_composite.png`: el proyecto LaTeX conservó la del 2026-04-15 y esa es la
que quedó impresa en la Figura 5. Se verificó extrayendo la imagen incrustada en la
página 9 del PDF enviado (5134 × 869 px, frente a 3274 × 1980 de la versión del pipeline).

## 4. Tablas (`04_tablas/`)

| Tabla del paper | Archivo | Nota |
|---|---|---|
| Tabla 1 (fuentes de datos) | — | redactada directamente en LaTeX, sin CSV de respaldo |
| Tabla 2 (diseño de escenarios) | `principales/table_scenario_design.csv` | el rol de S3 se reescribió en el manuscrito como *diagnostic soft-floor robustness check* |
| Tabla 3 (métricas de equidad) | `principales/table_equity_metrics_final.csv` | incluye además los diagnósticos del piso blando de S3 |
| Tabla 4 (compromiso de política) | `principales/table_policy_tradeoff.csv` | versión 2026-04-16, con columna añadida `vulnerable_coverage_gain_pp_vs_S0` |
| Figura 1 (datos de respaldo) | `principales/table_official_ev_growth.csv` | serie de vehículos registrados |

En `suplementarias/`: asignación por zona y escenario (`scenario_S0..S3.csv`),
cambios S2 frente a S1, métricas de equidad, desempeño disponible del GNN,
resumen de prioridades y la batería de robustez.

## 5. Reproducibilidad (`05_reproducibilidad/`)

| Carpeta | Contenido |
|---|---|
| `codigo/` | `followup.py` (modelo, figuras y tablas), scripts de preparación y ejecución, `pyproject.toml` |
| `manifiestos/` | manifiesto de insumos, manifiesto de la corrida y comparación con la referencia |
| `guias/` | guía de figuras en español y notas de estilo visual para JTG |

> **`followup.py` es la copia del árbol de trabajo, no la del último commit.**
> El commit `4742fa5` todavía contiene la versión anterior, que produce la figura 8 en un solo panel.
> El repositorio se declaró como *Research Data* en el envío, así que los revisores pueden abrirlo.

---

## Cifras clave (para verificar contra el manuscrito)

| Indicador | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| Desviación absoluta del objetivo | 62.828438 | 64.713291 | 67.854713 | 67.854713 |
| Pérdida de fidelidad frente a S0 (%) | 0.0 | 3.0 | 8.0 | 8.0 |
| Cobertura vulnerable a 15 min (%) | 77.78 | 86.11 | 88.89 | 88.89 |
| Gini ponderado por vulnerabilidad | 0.197249 | 0.197041 | 0.196896 | 0.196896 |
| Gini sin ponderar | 0.201468 | 0.201248 | 0.201111 | 0.201111 |
| Spearman prioridad–puertos | 0.922119 | 0.916699 | 0.910472 | 0.910472 |
| Accesibilidad media en el conjunto protegido | 410.68 | 410.32 | 410.26 | 410.26 |

Diagnóstico del piso blando en S3: piso calibrado en 287.140880 unidades, 95.45 % de las 22 zonas
protegidas ya lo cumplen, déficit total de 5.199 unidades, y la asignación coincide con S2.

## Discrepancias detectadas

1. **Orden de autores.** El orden registrado en la ficha de Editorial Manager está invertido
   respecto al de la portada enviada: quien figura primero en la ficha aparece tercero en la
   portada, y viceversa. Los detalles están en los archivos que no se versionan
   (ver `ARCHIVOS_SOLO_LOCALES.md`), porque identifican a los autores.
2. **Título.** La ficha de la revista lo registra sin el subtítulo *A socio-spatial allocation framework*.
3. **Etiqueta de S3.** `table_scenario_design.csv` dice *Soft-floor protection for vulnerable geography*;
   la Tabla 2 del manuscrito dice *Diagnostic soft-floor robustness check*. Solo cambia la etiqueta.
4. **Repositorio desactualizado.** Ver la nota de la sección 5.
5. **Figura 5 desfasada del pipeline.** El proyecto LaTeX usa la versión del 2026-04-15 de
   `fig_allocation_composite.png`. Si se recompila tomando las figuras del pipeline, esa
   figura cambiará respecto a lo que evaluó la revista.
6. **Nombres heredados.** `references_parta.bib` y `highlights_parta.txt` conservan el sufijo
   del antecesor dirigido a *Transportation Research Part A*; su contenido sí está al día.

---

## Sumas de verificación

Se omiten los cuatro archivos que identifican a los autores (ver `ARCHIVOS_SOLO_LOCALES.md`).

```
726ca1457466939dbc334a5132a9a41f  00_latex_fuente/MANIFEST.txt
e2700771060aacdb7529c5f38c7915d8  00_latex_fuente/_build_anterior/main_build_2026-04-17T15-07.pdf
a2939023e6bfe325fd978679db1e6f44  00_latex_fuente/data_additional/highlights_parta.txt
991419c222cdb3964eb42d5a96583a30  00_latex_fuente/data_additional/jtg_pre_submission_checklist.txt
1b587e3ee6805cd237cd0b1cb50d95ab  00_latex_fuente/data_additional/references_parta.bib
c400dcf243df7f3c34630204d92b8116  00_latex_fuente/elsarticle-harv.bst
f0bd46e64395049eea9ddc74e2a232b0  00_latex_fuente/elsarticle.cls
d8cb4f3385bd8a08b9d338db9b94844a  00_latex_fuente/figures/fig_allocation_composite.png
987f018ceec25575aacc523b090c3819  00_latex_fuente/figures/fig_attention_profile.png
6c4e419e4e79f6b3b4dbf839cee129e3  00_latex_fuente/figures/fig_growth_progression.png
49945c29fa2591a3871db0b751fc0581  00_latex_fuente/figures/fig_lorenz_curves.png
66a0a597eaa9ce912d03e151c694c69b  00_latex_fuente/figures/fig_net_change_S2_minus_S1.png
c12abeed525f27f7a3f04ad7a132cfc9  00_latex_fuente/figures/fig_official_ev_growth.png
907394bedddfd0fb63f47497917fcfe1  00_latex_fuente/figures/fig_policy_tradeoff.png
efd57bb6002c6c9167f3d543ca011cd0  00_latex_fuente/figures/fig_robustness.png
b2560fb3767092c40b1e48a147e5de03  00_latex_fuente/figures/fig_social_context_panels.png
7ddf1c2c25b52d69a9d3b197b28c9353  00_latex_fuente/main.tex
1e6e4719bf176ca3b24f17a268f0b889  00_latex_fuente/supplementary_appendix.tex
baa687e4d677ff45236d4fbb7a3a2758  01_manuscrito/01_manuscript.pdf
d7c33ef502610323864ce7723f73db4e  01_manuscrito/02_supplementary_appendix.pdf
d8cb4f3385bd8a08b9d338db9b94844a  02_figuras_texto_principal/fig_allocation_composite.png
987f018ceec25575aacc523b090c3819  02_figuras_texto_principal/fig_attention_profile.png
49945c29fa2591a3871db0b751fc0581  02_figuras_texto_principal/fig_lorenz_curves.png
66a0a597eaa9ce912d03e151c694c69b  02_figuras_texto_principal/fig_net_change_S2_minus_S1.png
c12abeed525f27f7a3f04ad7a132cfc9  02_figuras_texto_principal/fig_official_ev_growth.png
907394bedddfd0fb63f47497917fcfe1  02_figuras_texto_principal/fig_policy_tradeoff.png
b2560fb3767092c40b1e48a147e5de03  02_figuras_texto_principal/fig_social_context_panels.png
6c4e419e4e79f6b3b4dbf839cee129e3  03_figuras_suplemento/fig_growth_progression.png
efd57bb6002c6c9167f3d543ca011cd0  03_figuras_suplemento/fig_robustness.png
5cc36fb6258c08f5ad32ff36cca4fb4b  04_tablas/principales/table_equity_metrics_final.csv
2618d4f6261dc1869bbacdb4b988a851  04_tablas/principales/table_official_ev_growth.csv
f1d88a3dc68ffd898c3123be029b5d6c  04_tablas/principales/table_policy_tradeoff.csv
0d92d93507149961e7bcb6e2772b07da  04_tablas/principales/table_scenario_design.csv
0d66a77e0b02263942f9e8166491d6b8  04_tablas/suplementarias/scenario_S0.csv
c659c39bb309262a532fdb70d9414908  04_tablas/suplementarias/scenario_S1.csv
99a5782a334119cb11ab90de8cfa3652  04_tablas/suplementarias/scenario_S2.csv
99a5782a334119cb11ab90de8cfa3652  04_tablas/suplementarias/scenario_S3.csv
8da18a194813cc2b1fa0c06652ae973a  04_tablas/suplementarias/table_allocation_changes_S2_vs_S1.csv
3ccc4e95e79593571e62d018dfe2510c  04_tablas/suplementarias/table_equity_metrics.csv
f1e18b11029d622095da65eb079ad6f2  04_tablas/suplementarias/table_gnn_performance_available.csv
1ad11ac2de632bfbb4680b1d0b86685f  04_tablas/suplementarias/table_priority_summary.csv
0379939aff72180eb4fa9005e58ee074  04_tablas/suplementarias/table_robustness.csv
8de51eb61c989e78f7473acb3e7bc98e  05_reproducibilidad/codigo/followup.py
0ff55bb9b05e3c872c7d5ba803bd8c66  05_reproducibilidad/codigo/pyproject.toml
4f082e59b233863d87964b97662cf0a3  05_reproducibilidad/codigo/reproduce_followup.py
2ac918ddb1c13886aa1ab20b93e6c95c  05_reproducibilidad/codigo/stage_inputs.py
3aecbc40b0a1f4103fb679653b635fdf  05_reproducibilidad/guias/GUIA_FIGURAS_PAPER_ES.md
42e5d8e220567ec3acf72115eceeccc3  05_reproducibilidad/guias/jtg_figure_style_notes.md
9371ec40ca86b0ba1dc4d7778d584ab0  05_reproducibilidad/guias/paper_submission_recommendations.md
8bb2031e23e107e9704ae991d76e547e  05_reproducibilidad/manifiestos/input_manifest.json
367c77dbd54aa47cf466c7498b1ead76  05_reproducibilidad/manifiestos/reference_comparison.json
4f5c87d8f0db5b85a3fc44302967abd5  05_reproducibilidad/manifiestos/run_manifest.json
25816cdea6b56f03a255107f92265ed0  ARCHIVOS_SOLO_LOCALES.md
```
