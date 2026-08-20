# Fuente LaTeX del envío

Exportado de Overleaf como `final_submission_package`. Compila con `elsarticle`
en clase `[final,5p,times,twocolumn,authoryear]` y `\journal{Journal of Transport Geography}`.

## Cómo compilar

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplementary_appendix
```

El suplemento es un documento aparte (`\documentclass[11pt]{article}`) y no lleva bibliografía.

## Correspondencia con el PDF enviado

`main.tex` **sí** corresponde al manuscrito enviado (build del 2026-04-17 18:40 UTC).
Lo verifiqué comparando palabra por palabra: la única diferencia entre el build anterior
y el enviado es que se eliminó la sección *Declaration of generative AI and AI-assisted
technologies*, y `main.tex` ya no la contiene.

## Advertencias

**`_build_anterior/main_build_2026-04-17T15-07.pdf` no es lo que se envió.**
Es un build 3.5 horas anterior que todavía incluye la declaración de IA generativa.
Venía dentro del paquete de Overleaf como `main.pdf`. El PDF válido es
`../01_manuscrito/01_manuscript.pdf` (build de las 18:40 UTC).

**`figures/fig_allocation_composite.png` es la versión del 2026-04-15** (5134 × 869 px),
no la que regenera el pipeline el 2026-04-17 (3274 × 1980 px). Lo confirmé extrayendo la
imagen incrustada en la página 9 del PDF enviado: mide 5134 × 869. Las otras ocho figuras
sí son las del 2026-04-17. Si se recompila el proyecto tomando las figuras del pipeline,
**la Figura 5 cambiará de aspecto respecto a lo que evaluó la revista**.

**El nombre `references_parta.bib` y `highlights_parta.txt` viene del antecesor** dirigido a
*Transportation Research Part A*. El contenido sí está actualizado a JTG; solo el nombre quedó.
No conviene renombrarlos sin ajustar `\bibliography{data_additional/references_parta}` en la
línea 960 de `main.tex`.

**`MANIFEST.txt` lista `supplementary_appendix.pdf`, que no venía en el zip.**
El PDF compilado del suplemento está en `../01_manuscrito/02_supplementary_appendix.pdf`.

## Figuras declaradas en el fuente

| Archivo | Llamada desde | Figura |
|---|---|---|
| `fig_official_ev_growth.png` | `main.tex:207` | 1 |
| — | `main.tex` (dos `tikzpicture`) | 2 |
| `fig_social_context_panels.png` | `main.tex:699` | 3 |
| `fig_attention_profile.png` | `main.tex:706` | 4 |
| `fig_allocation_composite.png` | `main.tex:734` | 5 |
| `fig_net_change_S2_minus_S1.png` | `main.tex:741` | 6 |
| `fig_lorenz_curves.png` | `main.tex:814` | 7 |
| `fig_policy_tradeoff.png` | `main.tex:845` | 8 |
| `fig_growth_progression.png` | `supplementary_appendix.tex:121` | S1 |
| `fig_robustness.png` | `supplementary_appendix.tex:127` | S2 |
