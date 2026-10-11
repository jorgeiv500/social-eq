# C3 — Preregistro: de la zona al sitio (parqueaderos)

**Fijado el 2026-10-11 a las 11:56 (-0500), antes de construir la asignación por sitio y antes de ver
cualquier resultado de C2 sobre disponibilidad de parqueaderos por cuartil.** Decisiones tomadas por el
autor sobre opciones propuestas por el coordinador. Este archivo no se edita después; las desviaciones
se declaran en el README.

## Diseño de dos niveles
- Nivel 1 (zona): los puertos por UTAM y sede salen de la asignación S2 publicada (sin cambios).
- Nivel 2 (sitio): dentro de cada UTAM, los puertos se reparten entre parqueaderos candidatos en proporción
  a la demanda de llegadas de conductores de auto (nivel, no perfil horario) de la ZAT que contiene cada
  sitio, respetando la capacidad por sitio.

## Decisiones fijadas
1. **Candidatos (principal):** parqueaderos fuera de vía (OpenStreetMap amenity=parking y manzanas de
   Catastro con uso de parqueadero). **Sensibilidad:** se añaden las Zonas de Parqueo Pago en vía.
2. **Capacidad por sitio:** hasta el 10 % de los cupos del parqueadero se electrifica (redondeo hacia abajo,
   mínimo 1 si hay cupos). Sin dato de cupos: mediana de cupos de su tipo. **Sensibilidad:** 5 % y 20 %.
3. **Zona sin sitios suficientes:** **principal:** se reporta la brecha (puertos de la cuota sin sitio) por
   cuartil de vulnerabilidad, sin mover puertos. **Sensibilidad:** los puertos sin sitio se reubican en la
   UTAM vecina (contigüidad) con capacidad libre más cercana, y se recalcula el Gini ponderado y la
   provisión del cuartil vulnerable.
4. **Estimador de demanda (principal):** RF con rezago espacial (el mejor de B6). **Comparación:** ST-GNN,
   GCN y contracción bayesiana empírica de B6; métrica: puertos que cambian de sitio (½ Σ|Δ|) entre
   estimadores, como % del sistema.

## Métricas primarias (fijadas ahora)
- Fracción de la cuota zonal instalable, por cuartil de vulnerabilidad (principal de la brecha de sitios).
- Distancia (o tiempo) de cada ZAT, ponderada por población, al sitio con cargador más cercano, por
  cuartil; brecha entre el cuartil vulnerable y el resto.
- En la sensibilidad con reubicación: Δ Gini ponderado y Δ provisión del cuartil vulnerable
  (puertos/10.000 hab.).

## Control de sesgo
Todas las métricas se reportan con candidatos OSM, con candidatos Catastro y con ambos, para separar
"no hay parqueaderos" de "no están mapeados".

## Adenda 1 — 2026-10-11 12:03 (-0500): desviaciones declaradas antes de construir la asignación
Motivo: C2 reveló huecos que el preregistro no cubría (sin ver todavía ningún resultado de C3).
- A1. Catastro no tiene cupos y OSM los tiene solo en 2,5 % de candidatos; la "mediana de su tipo" no existe
  para Catastro. Regla: a OSM y Catastro sin cupos se les imputa la mediana de cupos de los parqueaderos
  OSM fuera de vía con cupos declarados; las ZPP usan sus cupos de auto. Sensibilidad: cuartiles 25 y 75.
- A2. Se excluyen como sitio de carga pública los candidatos OSM con acceso privado explícito
  (access=private/no). Los de acceso "customers" se mantienen.
- A3. Deduplicación: un punto OSM dentro de una manzana Catastro de parqueaderos cuenta una sola vez (OSM).
- A4. Alcance por sede: el nivel de sitio aplica a los puertos de **trabajo y públicos** (carga fuera del
  hogar). Los puertos residenciales representan carga en el hogar y no se ubican en parqueaderos.
  **Sensibilidad:** en las UTAM del cuartil vulnerable, los puertos residenciales también deben ubicarse en
  parqueaderos (hogares sin garaje), y se mide la brecha resultante.
- A5. La demanda por ZAT usa el nivel de llegadas de conductores de auto (C2), suavizado con el estimador
  de B6 a la escala ZAT; en ZAT con menos de 30 llegadas manda el valor suavizado.
