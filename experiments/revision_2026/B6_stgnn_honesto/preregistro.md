# B6 — Preregistro: ¿supera un ST-GNN sin fuga a los competidores clásicos?

**Fijado el 2026-10-11 a las 10:54 (-05:00), antes de escribir el código de modelado y antes de
ver cualquier resultado.** Hasta esta hora solo se inspeccionaron nombres de columnas y conteos de
modos de la EM-2023 (sin construir perfiles ni métricas). Este archivo no se edita después; si algo
se recorta por tiempo, se declara en el README como desviación, sin tocar el criterio.

Una adenda (régimen secundario fuera de tiempo, pedido por el coordinador) se añade al final con su
propia hora y **no modifica** la métrica primaria ni el criterio GO/NO-GO.

## 1. Tarea

Para cada zona *i*, el perfil horario de **salidas** (zona de origen, hora de `hora_ini`) y
**llegadas** (zona de destino, hora de `hora_fin`) de viajes con `modo_principal_agrupado == "AUTO"`
(principal) y de todos los modos (secundaria), ponderados por `fexp_vj`. Perfil = 48 valores
(24 h × {salida, llegada}).

De ese perfil salen dos cantidades:

- (a) **Participación por ventana de sede** ω_iv, con las ventanas del código
  (`followup.build_venue_weights`): residencial 22–5 h, trabajo 8–17 h, público 18–21 h, sobre la
  suma salidas + llegadas, normalizada para que las tres sumen 1 (como `build_inputs`).
- (b) **Nivel diario**: viajes expandidos (salidas + llegadas) de la zona, reescalados a población
  (factor Σ fexp_hg total / Σ fexp_hg de la submuestra).

Escalas: **UTAM** (142 polígonos de `UTAM2023`) y **ZAT** (polígonos de `ZAT2023.shp` cuyo centroide
cae dentro de alguna de las 142 UTAM; la UTAM y la localidad de cada ZAT salen de ese cruce).
Viajes con origen o destino fuera de esas zonas no cuentan para el perfil de esa zona.

## 2. Validación sin fuga

- **Partición de hogares** (`key_hg`): K = 10 particiones aleatorias (semillas 0–9) en mitades A y B.
  Todo modelo se ajusta y se calibra **solo con A**. La verdad es el perfil construido con B.
- Dentro de A, una segunda partición de hogares A1/A2 (misma semilla + 100) sirve para el ajuste:
  los modelos aprendidos se entrenan con entrada A1 → objetivo A2 y se validan con entrada A2 →
  objetivo A1 (selección de hiperparámetros y parada temprana); el modelo final se reajusta con
  ambas direcciones y predice con entrada A completa. Los métodos sin entrenamiento (EB, IDW) eligen
  su hiperparámetro con la misma validación A2 → A1.
- **Régimen principal ("relleno")**: todas las zonas, cada una con su propio dato de A disponible.
- **Régimen de *kriging* por bloques** (adicional): las localidades (`LOCNombre`) se reparten al azar
  (semilla de la partición) en 5 grupos; en cada pliegue, las zonas de un grupo quedan **sin dato
  propio de A** (perfil y nivel ocultos en la entrada, excluidas del entrenamiento) y se predicen;
  verdad = B. Las covariables de hogares de esas zonas (tipo censo) sí se usan. Los viajes de A que
  llegan a zonas no enmascaradas desde zonas enmascaradas siguen contando para la zona no
  enmascarada.
- **Fiabilidad entre mitades**: correlación ponderada de ω entre A y B y error A vs B (es el
  competidor ii). Cota aproximada del error alcanzable = wMAE(A,B)/√2.

## 3. Competidores (misma información: hogares y viajes de A; nada de B; sin `trips_total_w`)

Vecindario común: los 8 vecinos más cercanos por centroide **entre las zonas con dato disponible**,
pesos inversos a la distancia (IDW). Covariables estáticas por zona (solo hogares de A, módulos
hogares y vehículos): log de hogares expandidos, estrato medio, % estrato 1–2, % estrato 5–6,
% apartamento, personas por hogar, automóviles por hogar, lon, lat. Ningún agregado de viajes de la
zona objetivo entra como covariable estática.

1. **Media ciudad**: perfil medio ponderado de la ciudad (A); nivel = mediana de los niveles de A
   (en *kriging*, de las zonas disponibles).
2. **Crudo A**: perfil y nivel de A de la propia zona (en *kriging* no existe; se reporta como no
   aplicable y se sustituye por la media ciudad).
3. **EB / James-Stein**: perfil de A contraído hacia la media de su localidad en A:
   p̂ = (n·p_A + k·p_loc)/(n + k), con *n* = viajes no ponderados de la zona en A; el log-nivel se
   contrae igual con su propio k_L. Rejilla de 4 valores de (k, k_L). Si la localidad no tiene dato
   (régimen *kriging*), el prior es el rezago espacial IDW.
4. **RF con rezago espacial**: bosque aleatorio (300 árboles) con covariables estáticas + perfil
   propio de A (48 participaciones, log-nivel, log n) + rezago espacial IDW del perfil y del
   log-nivel de los vecinos; dos bosques (participaciones multisalida y log-nivel), peso de muestra =
   viajes del objetivo. 4 configuraciones (profundidad máx. {None, 12} × hoja mínima {3, 10}).
   En *kriging* se quitan las variables propias. Ajuste anidado según §2.
5. **Suavizado por grafo / IDW**: p̂ = (n_i p_i + λ Σ_j w_ij n_j p_j)/(n_i + λ Σ_j w_ij n_j);
   igual en log-nivel. 4 valores de (λ, potencia de distancia).
6. **GCN sin módulo temporal**: 2 capas de agregación ponderada sobre el grafo común (con
   autoenlace), entrada = covariables + perfil propio aplanado + log-nivel + log n + bandera de
   máscara; salida = 48 logits (softmax) + log-nivel.
7. **ST-GNN inductivo**: por hora, entrada (salidas_h, llegadas_h) + bandera de máscara → 2 capas de
   agregación en el grafo (pesos compartidos entre horas) → GRU bidireccional pequeña sobre las 24 h
   → 2 logits por hora (softmax conjunto sobre 48) ; rama estática MLP con covariables fusionada a
   cada hora; cabeza de log-nivel. Entrenamiento con **enmascaramiento de nodos estilo IGNNK**
   (25 % de nodos ocultos en la entrada por época), salida multihora, pérdida = entropía cruzada del
   perfil + 0,5·MSE del log-nivel, ponderadas por los viajes del objetivo.

Presupuesto común: 4 configuraciones por método con hiperparámetros (EB, RF, IDW, GCN, ST-GNN: GCN y
ST-GNN con oculto {16, 32} × decaimiento {1e-4, 1e-3}, Adam 1e-2 → hasta 400 épocas, paciencia 40).

## 4. Métricas

- **Primaria (una por escala)**: wMAE de ω contra B =
  Σ_i w_i · (1/3)Σ_v |ω̂_iv − ω^B_iv| / Σ_i w_i, con w_i = viajes expandidos de B en la zona *i*
  (AUTO), sobre zonas con viajes de B en las ventanas. **Escala de decisión: ZAT, régimen principal,
  modo AUTO.**
- Secundarias: wMAE del log-nivel; Spearman del nivel; mismas métricas en *kriging* y en todos los
  modos.
- **Secundaria de decisión**: puertos mal ubicados en S0 exacto (redondeo por sede, como B5) con las
  cuotas 2024 del artículo (7.352 / 2.739 / 779): meta t_iv = Q_v · prio_i ω_iv / Σ, prio = nivel
  normalizado min-máx (como `priority_norm`), a escala UTAM (las predicciones ZAT se agregan a UTAM
  sumando conteos predichos). Puertos mal ubicados = ½ Σ_iv |n_iv(competidor) − n_iv(B)|.

## 5. Suelo de ruido e inferencia

- Variación entre particiones (DE entre las K).
- Diferencia ST-GNN − mejor de (3, 4) con **bootstrap por bloques de localidad** (2.000 réplicas,
  remuestreo de localidades con reemplazo; en cada réplica, wMAE agregado sobre las K particiones),
  IC 95 % percentil. "Mejor de (3, 4)" = el de menor wMAE medio sobre las K particiones (ZAT,
  principal, AUTO), fijado antes del bootstrap.

## 6. Criterio de decisión (fijo)

**GO** solo si, en escala ZAT, régimen principal, modo AUTO, el ST-GNN mejora la wMAE de ω frente al
mejor de EB (3) y RF con rezago (4) con IC 95 % de la diferencia que excluye 0 **y** mejora relativa
≥ 10 %. En cualquier otro caso **NO-GO** (empate o pérdida), y se reporta igual.

Recortes permitidos por tiempo (se declaran, no cambian el criterio): K = 5, solo AUTO, *kriging*
con menos particiones.

---

## Adenda 1 — régimen secundario fuera de tiempo (2026-10-11 10:56 -05:00, antes de ver resultados)

Pedido del coordinador. **No altera** la métrica primaria ni el criterio GO/NO-GO.

- Partición temporal de hogares por la columna `fecha` del módulo hogares: A = hogares con fecha en
  abril y mayo de 2023 (y los de marzo, si los hay, por ser anteriores); B = junio; B' = octubre
  (solo si tiene muestra; se reporta con la advertencia de tamaño). Una sola partición (no K).
- Mismos competidores, mismo ajuste anidado dentro de A (A1/A2 aleatorio por hogares), ambas
  escalas, modo AUTO (todos los modos si da tiempo). El nivel se reescala por Σ fexp_hg de cada
  submuestra.
- Inferencia: bootstrap por bloques de localidad de la diferencia ST-GNN − mejor de (3, 4).
- Advertencia que se verifica y declara en el README: `fecha` puede ser la fecha de la entrevista,
  no la del día de viaje.
- Opcional (si da tiempo): perfiles separados día hábil / fin de semana. No se promete.
