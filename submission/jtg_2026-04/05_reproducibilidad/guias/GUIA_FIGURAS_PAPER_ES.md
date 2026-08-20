# Guía para integrar las figuras en el paper

Este paquete reúne las figuras exportadas y una guía práctica para que se ajusten el manuscrito, los captions y el texto de resultados/discusión.

## Actualización 2026-04-17

- Reemplazar en el manuscrito las versiones previas de `fig_social_context_panels.png`, `fig_net_change_S2_minus_S1.png`, `fig_growth_progression.png`, `fig_policy_tradeoff.png` y `fig_robustness.png`.
- `fig_social_context_panels.png` quedó realineada: el panel derecho ya no desciende respecto a los otros dos.
- `fig_net_change_S2_minus_S1.png` y `fig_growth_progression.png` ahora tienen mejor contraste en celdas con cambio cero, para que la reasignación neta no se vea “lavada”.
- `fig_policy_tradeoff.png` pasó a una síntesis ancha de cuatro paneles; en `b` la leyenda y el texto de apoyo quedaron fuera del área de datos, y en `c` las etiquetas auxiliares se movieron a una columna lateral para no tapar puntos.
- El paquete incluye también tablas de apoyo en `tables/` para facilitar captions, apéndice o respuesta a revisores.

## Recomendación editorial general

- Mantener en el texto principal las figuras que sostienen la cadena argumental del paper: contexto empírico, contexto socioespacial, aprendizaje temporal, comparación de escenarios, cambio espacial S2 vs S1, Lorenz y trade-off.
- Dejar en suplemento las figuras de respaldo metodológico o de inspección detallada por escenario.
- En `elsarticle` `5p` a dos columnas:
  - usar `figure*` para figuras anchas con varios paneles o mapas;
  - usar `figure*` también para `fig_policy_tradeoff.png`, porque la versión actual es ancha y pierde legibilidad si se fuerza a una columna;
  - no volver a envolver `fig_policy_tradeoff.png` en `tikzpicture` con cajas extra.

## Orden sugerido en texto principal

1. `fig_official_ev_growth.png`
2. `fig_social_context_panels.png`
3. `fig_attention_profile.png`
4. `fig_allocation_composite.png`
5. `fig_net_change_S2_minus_S1.png`
6. `fig_lorenz_curves.png`
7. `fig_policy_tradeoff.png`

## Reglas rápidas de redacción

- Hablar de cambios modestos pero relevantes bajo cuotas fijas e integrales.
- No presentar `S3` como una redistribución adicional si la asignación coincide con `S2`.
- En mapas, describir patrones espaciales y localización relativa; no afirmar causalidad espacial fuerte.
- En figuras del modelo, hablar de “priority extraction”, “temporal attention” y “decision translation”, no de superioridad algorítmica universal.

## Cambios LaTeX recomendados

- Para `fig_policy_tradeoff.png`, usar:

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{figures/fig_policy_tradeoff.png}
\caption{Policy trade-off synthesis across allocation scenarios. Panel (a) shows the coverage-fidelity frontier, where the move from S0 to S1 delivers the largest marginal gain in vulnerable-zone coverage and the move from S1 to S2/S3 yields a smaller additional improvement at higher fidelity cost. Panel (b) summarizes robustness around S2 across alternative decay, vulnerability-index, epsilon, kappa and soft-floor specifications. Panel (c) reports weighted-Gini improvements relative to S0 alongside gains in vulnerable coverage and mean accessibility for high-vulnerability areas. Panel (d) reports the lower-tail safeguard for the protected set under S3. S2 and S3 overlap on the frontier because the soft-floor safeguard is non-binding in the reproduced allocation.}
\label{fig:policy_tradeoff}
\end{figure*}
```

- Para las figuras anchas del cuerpo principal, mantener `figure*` y `\includegraphics[width=0.98\textwidth]{...}` o `\textwidth`.
- Para suplemento, usar `figure*` cuando haya mapas múltiples o muchas etiquetas.

## Instrucciones por figura

### 1. `fig_official_ev_growth.png`

**Uso recomendado**
Texto principal, final de la subsección de datos / motivación empírica.

**Qué sostiene**
Demuestra que el problema de asignación no es hipotético ni marginal: el parque EV de Bogotá creció con rapidez y justifica la necesidad de planificar capacidad pública.

**Cómo hablar de ella**
Enfatizar el crecimiento observado, el salto entre checkpoints y el hecho de que las fechas no siempre coinciden con cierres de diciembre.

**Frase sugerida**
“Official registration checkpoints show a steep expansion of Bogotá’s EV fleet, from 2,039 vehicles in November 2020 to 8,899 in November 2024, which motivates a more explicit allocation framework for public charging under fixed quotas.”

**No sobreinterpretar**
No vender la serie como una serie mensual regular ni como evidencia espacial por UTAM; es contexto citywide.

**Caption recomendado**
Mantener una formulación tipo: “Official registration checkpoints for electric vehicles in Bogotá and year-to-year additions. Checkpoint dates do not always coincide with December year-end.”

### 2. `fig_social_context_panels.png`

**Uso recomendado**
Texto principal, al inicio de Resultados, antes de mostrar asignaciones.

**Qué sostiene**
Presenta el contexto socioespacial de la decisión: prioridad aprendida, vulnerabilidad social y geografía protegida de `S3`.

**Cómo hablar de ella**
Decir que prioridad y vulnerabilidad se superponen solo parcialmente; la geografía protegida es compacta y localizada; esto abre espacio para reasignaciones focalizadas sin romper la lógica de demanda.

**Frase sugerida**
“Figure~\\ref{fig:social_context} shows that learned priority and social vulnerability only partially overlap, while the protected geography remains spatially compact, making localized vulnerability-sensitive reallocation feasible under fixed quotas.”

**No sobreinterpretar**
No decir que la vulnerabilidad “explica” la prioridad; la figura muestra coexistencia espacial, no causalidad.

### 3. `fig_attention_profile.png`

**Uso recomendado**
Texto principal, en la subsección de resultados sobre aprendizaje socioespacial y atención temporal.

**Qué sostiene**
Da evidencia de que el modelo extrae estructura temporal interpretable y que esa estructura cambia por vulnerabilidad social.

**Cómo hablar de ella**
Subrayar que la atención es más alta en las primeras horas y luego se aplana; las zonas de alta vulnerabilidad reciben más peso en horas tempranas.

**Frase sugerida**
“The temporal attention profile peaks in the earliest hours of the day and then flattens through the work and public-service windows; when split by social vulnerability, higher-vulnerability zones receive relatively more early-hour attention.”

**No sobreinterpretar**
No afirmar que la atención “prueba” comportamiento causal de carga; es una señal aprendida e interpretable, no observación directa de transacciones.

### 4. `fig_allocation_composite.png`

**Uso recomendado**
Texto principal, como principal figura espacial de resultados.

**Qué sostiene**
Resume la comparación entre `S0`, `S1`, `S2` y `S3`. Muestra que la estructura metropolitana general se preserva y que las diferencias son locales.

**Cómo hablar de ella**
Enfatizar que la reasignación es incremental bajo cuotas fijas; `S3` es visualmente igual a `S2`; la figura ya quedó en formato `2×2`, mejor para journal.

**Frase sugerida**
“The scenario maps preserve the broad metropolitan structure of the learned allocation, while equity-oriented scenarios reallocate ports within the consolidated urban core rather than producing a citywide reshuffling.”

**No sobreinterpretar**
No describir las diferencias como grandes redistribuciones sistémicas; son cambios discretos bajo restricciones enteras.

### 5. `fig_net_change_S2_minus_S1.png`

**Uso recomendado**
Texto principal, inmediatamente después de la figura compuesta o en la misma subsección.

**Qué sostiene**
Hace visible la lógica espacial exacta del cambio de `S1` a `S2`: pocas zonas cambian y los movimientos son locales.

**Cómo hablar de ella**
Decir que el cambio implica diez zonas, con ganancias y pérdidas compensadas, en un clúster urbano compacto.

**Frase sugerida**
“The S2 reallocation relative to S1 is sparse and localized: five one-port gains offset five one-port losses, concentrated in a compact set of adjacent UTAMs rather than in a center-periphery inversion.”

**No sobreinterpretar**
No decir que la figura muestra expansión de cobertura por sí sola; muestra cambio neto de puertos, no cambio directo en accesibilidad.

### 6. `fig_lorenz_curves.png`

**Uso recomendado**
Texto principal, junto a la tabla de métricas de equidad.

**Qué sostiene**
Entrega una lectura estándar de desigualdad de accesibilidad en formato de transport equity.

**Cómo hablar de ella**
Decir que las curvas se desplazan levemente hacia mayor igualdad; la mejora mayor ocurre entre `S0` y `S1`, con una mejora adicional más pequeña de `S1` a `S2/S3`.

**Frase sugerida**
“The Lorenz diagnostics show a modest but consistent shift toward equality under the equity-oriented scenarios, with the largest visual gain from S0 to S1 and a smaller additional gain under socially weighted allocation.”

**No sobreinterpretar**
No afirmar grandes reducciones distributivas; la magnitud es pequeña y coherente con cambios discretos.

### 7. `fig_policy_tradeoff.png`

**Uso recomendado**
Texto principal, como síntesis compacta del argumento de optimización.

**Qué sostiene**
Condensa el intercambio entre fidelidad a prioridad aprendida, mejora distributiva, robustez local alrededor de `S2` y resguardo del lower tail en `S3`.

**Cómo hablar de ella**
Decir que `S0→S1` entrega la mayor ganancia marginal en cobertura por un costo relativamente bajo; `S1→S2` añade una mejora menor a mayor costo; `S2 = S3`; y la batería de robustez no altera el orden cualitativo de las políticas.

**Frase sugerida**
“The trade-off synthesis indicates that the move from S0 to S1 delivers the largest marginal increase in vulnerable-zone coverage, whereas the move from S1 to S2 yields a smaller additional improvement at higher fidelity cost; robustness checks preserve the qualitative ordering, and S3 overlaps with S2 because the soft floor is non-binding.”

**No sobreinterpretar**
No presentar `S3` como un nuevo punto distinto ni como una mejora extra observada en la corrida reproducida.

**Nota de maquetación**
Usarla en ancho de página (`figure*`). No añadir overlays ni cajas TikZ encima del PNG, y no forzarla a una sola columna porque los paneles `b` y `c` pierden legibilidad.

### 8. `fig_priority_map.png`

**Uso recomendado**
Suplemento o material de apoyo si el editor pide una vista standalone de prioridad.

**Qué sostiene**
Permite inspeccionar la superficie de prioridad sin combinarla con vulnerabilidad ni con escenarios.

**Cómo hablar de ella**
Usarla para decir dónde se concentran las prioridades aprendidas antes de cualquier traducción a reglas de asignación.

**Frase sugerida**
“The standalone priority map isolates the learned spatial signal before any equity-oriented translation into allocation targets.”

**No sobreinterpretar**
No usarla sola como evidencia de equidad o inequidad.

### 9. `fig_S3_protected_set.png`

**Uso recomendado**
Suplemento.

**Qué sostiene**
Aclara visualmente cuáles UTAM pertenecen al conjunto protegido de `S3`.

**Cómo hablar de ella**
Presentarla como una figura de definición espacial del policy safeguard.

**Frase sugerida**
“The protected-set map clarifies the spatial footprint of the top-vulnerability zones subject to the S3 soft-floor safeguard.”

**No sobreinterpretar**
No tratarla como resultado; es una figura de definición del conjunto protegido.

### 10. `fig_growth_progression.png`

**Uso recomendado**
Suplemento.

**Qué sostiene**
Muestra la progresión espacial de cambios entre escenarios o etapas.

**Cómo hablar de ella**
Usarla si se quiere reforzar la idea de transición de política o si un revisor pide más detalle espacial entre etapas.

**Frase sugerida**
“The growth-progression maps expand the stage-to-stage spatial story behind the final allocation comparison.”

**No sobreinterpretar**
No es necesaria si ya están `allocation_composite` y `net_change`; puede sentirse redundante en el texto principal.

### 11. `fig_robustness.png`

**Uso recomendado**
Suplemento, salvo que un revisor exija robustez en el cuerpo principal.

**Qué sostiene**
Documenta que el orden de escenarios es estable y que la mayor sensibilidad viene de los parámetros de decaimiento.

**Cómo hablar de ella**
Enfatizar estabilidad alta de ranking y sensibilidad distributiva moderada a la calibración de decaimiento.

**Frase sugerida**
“Robustness checks preserve the main scenario ordering and high rank stability, while the strongest sensitivity appears in the accessibility-decay specification.”

**No sobreinterpretar**
No vender la figura como prueba de completa invariancia; sí hay sensibilidad paramétrica, especialmente en `lambda`.

### 12. `fig_gnn_learning_history.png`

**Uso recomendado**
Suplemento metodológico.

**Qué sostiene**
Da transparencia al proceso de entrenamiento del modelo.

**Cómo hablar de ella**
Usarla solo si se quiere respaldar estabilidad del entrenamiento o si el editor pide más detalle del componente GNN.

**Frase sugerida**
“The training-history figure provides transparency on the evolution of the learning objective and error metrics across iterations.”

**No sobreinterpretar**
No convertirla en el centro del argumento; el paper no se vende como benchmark puramente predictivo.

### 13. `fig_allocation_S0.png`

**Uso recomendado**
Suplemento.

**Qué sostiene**
Vista aislada del escenario base de fidelidad a prioridad.

**Cómo hablar de ella**
Útil para describir el baseline sin la densidad de la figura compuesta.

**Frase sugerida**
“The standalone S0 map isolates the pure priority-matching allocation before equity relaxation.”

**No sobreinterpretar**
No repetir en el texto principal si `fig_allocation_composite` ya está.

### 14. `fig_allocation_S1.png`

**Uso recomendado**
Suplemento.

**Qué sostiene**
Vista aislada del escenario de equalización territorial.

**Cómo hablar de ella**
Sirve para mostrar la primera relajación de equidad sin pesos de vulnerabilidad.

**Frase sugerida**
“The standalone S1 map shows territorial equalization before the socially weighted correction introduced in S2.”

**No sobreinterpretar**
No decir que `S1` ya corrige explícitamente vulnerabilidad social; esa lógica entra en `S2`.

### 15. `fig_allocation_S2.png`

**Uso recomendado**
Suplemento.

**Qué sostiene**
Vista aislada del escenario socialmente ponderado.

**Cómo hablar de ella**
Úsala si se necesita una lectura detallada de la geografía final de `S2`.

**Frase sugerida**
“The standalone S2 map highlights the final socially weighted allocation pattern without the visual compression of the multi-panel comparison.”

**No sobreinterpretar**
No presentar `S2` como ruptura completa con la superficie de prioridad; sigue siendo una relajación acotada.

### 16. `fig_allocation_S3.png`

**Uso recomendado**
Suplemento.

**Qué sostiene**
Vista aislada del escenario con soft floor.

**Cómo hablar de ella**
Decir explícitamente que reproduce la misma asignación que `S2` en la corrida reproducida.

**Frase sugerida**
“The standalone S3 map is visually identical to S2 because the calibrated soft-floor safeguard is non-binding in the reproduced run.”

**No sobreinterpretar**
No describir diferencias espaciales que no existen.

## Figuras que no deben abrir una nueva afirmación en el paper

- `fig_allocation_S0.png`
- `fig_allocation_S1.png`
- `fig_allocation_S2.png`
- `fig_allocation_S3.png`
- `fig_priority_map.png`
- `fig_S3_protected_set.png`
- `fig_gnn_learning_history.png`

Estas deben funcionar como respaldo, detalle o suplemento, no como el corazón narrativo del artículo.

## Qué afirmaciones conviene reforzar en el manuscrito

- El paper une aprendizaje espacial y traducción auditable a reglas de asignación.
- Los cambios distributivos son modestos en agregado, pero relevantes bajo cuotas fijas e integrales.
- `S2` mejora cobertura vulnerable y Gini ponderado sin romper el vínculo con prioridad aprendida.
- `S3` opera como chequeo de robustez de lower tail, no como una redistribución adicional observada.

## Qué afirmaciones conviene suavizar

- Evitar lenguaje de “optimal social justice outcome”.
- Evitar decir que el GNN recupera demanda real de carga observada.
- Evitar presentar cambios pequeños como transformaciones urbanas amplias.
- Evitar leer la geografía protegida como frontera normativa definitiva; es una calibración de política.
