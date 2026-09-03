# E7 — Figuras al tamaño de impresión (R2.m2)

Experimento de la revisión JTRG-D-26-00692. No resuelve nada nuevo: **regenera** las
figuras del manuscrito a partir de los datos archivados de la corrida oficial.

## Por qué existe

El Revisor 2 escribió: *"Maps such as Figures 3 and 5 are too small and difficult to
read"*. La pasada del Frente E cambió las proporciones, pero la verificación a tamaño de
impresión del 2026-09-02 mostró que el problema de fondo seguía en pie: cada figura se
dibujaba con un `figsize` de 8.6–15.3 pulgadas de ancho y se colocaba en el `.tex` a
5.1–7.1 pulgadas. Como el `fontsize` de matplotlib está en puntos y el lienzo en pulgadas,
todo el texto se reproducía al **46–79 %** del tamaño con que se había rotulado, y la
fuente más pequeña de cada figura caía entre **3.2 y 5.2 pt**.

`\textwidth` de `elsarticle 5p twocolumn` = 522 pt = 7.22 in; `\textheight` = 682 pt = 9.44 in.

## Qué hace

`regenerar_todas.py` construye los argumentos de `export_figures`
(`src/social_eq/followup.py`) desde las tablas archivadas y llama a la misma función que usa
el pipeline, escribiendo en `salidas/`. Nunca toca `outputs/` ni `latex/figures/`.

Insumos, todos archivados:

- `outputs/social_equity_followup_outputs/utam_shp/UTAM2023/`
- `outputs/.../paper_v6_experiments/tables/scenario_S{0,1,2,3}.csv` — de ahí salen
  `priority_norm` y `vulnerability_index` (el script comprueba que coincidan en los cuatro
  escenarios antes de usarlos) y el conjunto protegido de S3 (cuantil 0.85 → 22 unidades,
  `ModelConfig.s3_protected_share = 0.15`)
- `outputs/.../tables/table_robustness.csv`
- `data/inputs/temporal_attention_by_utam.csv` y `priority_utam_scores.csv`

El mapa de contexto (Figura 2) no pasa por `export_figures`: se regenera con
`../E3_mapas/regenerar_figuras.py`.

## Control de que la regeneración es fiel

Antes de tocar ningún tamaño se ejecutó el script tal cual: **siete de las ocho figuras
salieron byte a byte idénticas** a las del manuscrito (`md5` coincidente). La octava es la
del contexto, que viene de E3 y también se reprodujo. Solo después se cambiaron los
`figsize`.

## Resultado

| Figura | fuente mínima antes → después | ppp efectivos | alto impreso |
|---|---|---|---|
| `fig_official_ev_growth` | 4.3 → 8.2 pt | 304 | 4.40 in |
| `fig_lorenz_curves` | 5.0 → 8.0 pt | 339 | 2.84 in |
| `fig_attention_profile` | 4.6 → 7.8 pt | 340 | 3.02 in |
| `fig_social_context_panels` | 4.6 → 7.7 pt | 311 | 8.02 in |
| `fig_allocation_composite` | 4.7 → 7.5 pt | 318 | 8.02 in |
| `fig_study_area_context` | 4.4 → 7.4 pt | 322 | 3.33 in |
| `fig_net_change_S2_minus_S1` | 5.2 → 6.6 pt | 302 | 4.27 in (franja inferior añadida el 2026-09-02, ver abajo) |
| `fig_policy_tradeoff` | 3.2 → 6.4 pt | 329 | 5.89 in |

Suplemento (añadido el 2026-09-02; `\textwidth` del suplemento = 8.5 − 2 × 0.75 = 7.0 in):

| Figura | ancho de diseño → impreso | fuente mínima antes → después | ppp efectivos | alto impreso |
|---|---|---|---|---|
| `fig_growth_progression` (C1) | 13.46 in → 7.0 in (escala 0.52) | 8 pt rotulados → 4.2 pt impresos → **8 pt** | 305 | 8.17 in |
| `fig_robustness` (C2) | 11.91 in → 7.0 in (escala 0.59) | 8.4 pt rotulados → 4.9 pt impresos → **8.4 pt** | 305 | 5.01 in |

Las fuentes mínimas del suplemento salen del código (`labelsize=8` de la barra de color en
`add_horizontal_colorbar`; `fontsize=8.4` de las etiquetas del eje y en la figura de robustez)
multiplicadas por la escala de impresión.

Tres cambios de diseño acompañaron al redimensionado, porque al lienzo de impresión no le
cabía el contenido con el que las figuras estaban rotuladas:

1. **`fig_policy_tradeoff`** — la frase explicativa del panel (b) fijaba por sí sola un ancho
   de ~4.8 in: se movió al caption, igual que el "95.5 % of protected UTAMs meet the floor"
   del panel (d). Se separaron las anotaciones `P10` y `Soft floor` y se acortó la etiqueta
   del eje y.
2. **`fig_official_ev_growth`** — de 1×2 a dos paneles apilados: en 1×2 el panel derecho
   quedaba en 2.4 in y solapaba título, rótulos de barra y marcas de eje. Se quitó la nota al
   pie, ya presente literalmente en el caption.
3. **Rótulos de localidad** — `add_locality_labels` los coloca por área descendente con prueba
   de colisión sobre una caja proporcional al largo del texto, y reserva las cajas de la
   flecha de norte, la barra de escala y el locator. Omite el rótulo que caería encima de
   otro en vez de superponerlo.

## Uso

```bash
python experiments/revision_2026/E7_figuras_impresion/regenerar_todas.py
python experiments/revision_2026/E3_mapas/regenerar_figuras.py
```

Después se copian a `../../latex/figures/` las ocho que usa el manuscrito y las dos que usa el
suplemento (`fig_growth_progression.png`, `fig_robustness.png`).

## Segunda pasada (2026-09-02, tarde)

La revisión visual de `fig_net_change_S2_minus_S1` ya regenerada mostró tres defectos que la
tabla de tamaños no detecta, más uno mal diagnosticado:

1. **El localizador tapaba el polígono de UTAM572**, una de las diez unidades reasignadas. El
   recuadro de situación se dibuja como eje hijo con `zorder` 5, por encima de los polígonos
   (`zorder` < 5) y por debajo de líneas guía, puntos y rótulos (6–9): la unidad quedaba
   oculta y solo se veía su punto de anclaje flotando sobre el recuadro. Corrección: el recorte
   lleva ahora una **franja inferior** (`pad_bottom = 0.45` del alto del rectángulo de unidades
   reasignadas), de modo que el rectángulo arranca en la fracción 0.28 del recorte y el
   mobiliario (localizador hasta 0.25, barra de escala en 0.055) queda debajo de cualquier
   unidad. La figura pasa de 7.00 × 4.70 a 7.00 × 5.20 in de lienzo (4.27 in de alto impreso).
2. **Los dos últimos rótulos de la escalera derecha** (UTAM65, UTAM572) se imprimían encima del
   localizador. Ahora ambas escaleras arrancan por encima de la franja (`ladder_floor = 0.30`).
3. **"Cota" caía sobre el punto de anclaje de UTAM610**, que es la propia unidad de ese
   municipio, así que el nombre y el punto compartían coordenada. Los rótulos UTAM se dibujan
   ahora antes que los nombres y reservan sus cajas (`label_box`) y sus puntos de anclaje; el
   nombre que colisiona se omite, como ya hacía `add_locality_labels` con las localidades.
4. **El "halo de un rótulo" tras la barra de escala no existía.** Renderizada la figura sin la
   barra, lo que asomaba era el contorno del polígono del anillo metropolitano sobre el que
   caía la barra. `add_scale_bar` acepta ahora `x_frac` (por defecto 0.5, sin cambio en las
   demás figuras) y en esta figura se coloca en 0.67, sobre fondo limpio a la izquierda del
   localizador.

Control de regresión: tras cada cambio se regeneraron las quince figuras del script y se
compararon por `md5` con la tanda anterior; solo cambió la figura tocada (y, en la tanda del
suplemento, solo `fig_growth_progression` y `fig_robustness`).

**Suplemento.** `fig_growth_progression` se dibujaba con tres paneles en fila
(`figsize = (13.35, 5.5)`) y `fig_robustness` a 11.8 in de ancho; ambas se colocan a
`\textwidth` = 7.0 in en el suplemento, así que su texto se imprimía al 52 % y 59 %. La
primera pasa a tres paneles apilados con la barra de color debajo; la segunda mantiene los dos
paneles en fila pero comparte el eje y (las 13 etiquetas se leen una sola vez, a 8.4 pt) y
acorta los títulos porque el eje x ya dice contra qué se compara. El suplemento pasa de 6 a 7
páginas y compila sin avisos.


## Pasada de forma (2026-09-02, noche)

El autor señaló que, tras la re-corrida, varias figuras dejaban demasiado blanco. Diagnóstico
con la fracción de filas en blanco por página del PDF: las páginas 8, 10, 12 y 13 eran páginas
de flotantes con entre el 44 % y el 55 % de la altura vacía. Dos causas:

1. **Marco alargado.** Las Figuras 4, 6 y 7 usaban el marco completo de la región (153 × 57 km,
   relación 2.7, dominado por la UPR de Sumapaz), lo que obligaba a apilar paneles al 70 % del
   ancho. Ahora las tres usan el **mismo recorte al núcleo urbano** (`urban_core_extent`: caja
   de las 111 UTAM urbanas con 3 km de margen, 48 × 28 km, relación 1.7; contiene 126 de 142
   unidades y el 92 % de los puertos) en cuadrícula 2 × 2 a todo el ancho; el panel 4d es el
   localizador de ese marco sobre la región completa. Los mapas quedan **2.6 veces más grandes**
   y las figuras pasan de 8.0 in a 5.2 in (Fig. 4) y de 8.0 in a 5.65 in (Fig. 6) de alto.
2. **Fracciones de flotantes.** Por la mañana se había bajado `\dblfloatpagefraction` a 0.6
   para que la Figura 9 no fuera al final; con las figuras compactas eso hacía que LaTeX armara
   páginas de flotantes medio vacías. Se fija en 0.9, con `\textfraction` 0.12 para que el texto
   rellene bajo los flotantes altos.

Resultado: 21 páginas (antes 22); ninguna página de figuras con más del 38 % de filas en
blanco (antes hasta 55 %); la Figura 9 sigue en la página 15, junto a su texto.

| Figura | antes (alto impreso) | ahora | marco |
|---|---|---|---|
| Fig. 4 | 8.02 in, 3 paneles apilados al 70 % | 5.16 in, 2 × 2 al 98 % | núcleo urbano + localizador |
| Fig. 6 | 8.02 in, 4 apilados al 70 % | 5.65 in, 2 × 2 al 98 % | núcleo urbano |
| Fig. 7 | 4.27 in, caja de las zonas que cambian (con datos corregidos, alargada por la UPR2) | 4.75 in | núcleo urbano; UTAM640 fuera, anotada |
| Fig. 3 | 3.26 in | 2.90 in | — |

## Figura 9 rediseñada (2026-09-03)

El autor observó que la Figura 9 ocupaba mucho y "no tenía variaciones". Estaba bien trazada,
pero con los insumos corregidos la cobertura del cuartil vulnerable es 100 % en todos los
escenarios: el panel principal (frontera cobertura-fidelidad) era una línea plana y los paneles
de robustez y de ganancia repetían "+0.0 pp". Se rediseñó para mostrar lo que sí varía:

- **(a) El precio de la equidad**: Gini ponderado frente a pérdida de fidelidad en escala
  simétrica-logarítmica, con S0–S2, el barrido de tolerancia (A1) y el benchmark de equidad
  directa (A2); recuadro de zoom sobre la familia lexicográfica, cuya caída (6 × 10⁻⁴) es
  invisible a la escala del benchmark.
- **(b) Intensidad de provisión**: puertos por 10.000 habitantes del cuartil vulnerable frente
  al resto (10.4 vs 15.6 en S0–S2) y del benchmark, que **deja al cuartil vulnerable con 1.7**:
  equalizar accesibilidad y proveer capacidad donde vive la población vulnerable son objetivos
  distintos. Este hallazgo se llevó también a §5.3 y §5.5.
- **(c) Robustez del Gini ponderado** en las doce especificaciones (la cobertura no se dibuja
  porque no cambia en ninguna).
- **(d) Salvaguarda de S3**, sin cambios.

El barrido y el benchmark no los conoce el pipeline oficial: `regenerar_todas.py` los lee de
`A1_barrido_epsilon/` y `A2_benchmark_equidad/` y llama a `fig_policy_tradeoff` con esas
superposiciones; sin ellas la función dibuja solo S0–S3. Los puertos per cápita salen de
`ports_per_capita_by_quartile` (nueva en `followup.py`) con la población DANE de
`utam_social_vulnerability.csv`, sobre las 120 zonas con población.
