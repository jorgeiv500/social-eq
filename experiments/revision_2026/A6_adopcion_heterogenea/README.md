# A6 — Escenarios 2030 con adopción espacialmente diferenciada

**Responde:** la objeción previsible a §5.6 (y R2.2). En A3 las cuotas 2030 se escalan
proporcionalmente y la adopción se supone espacialmente uniforme; la señal de demanda zonal
no cambia y la invarianza del Gini (0.164653 → 0.164840) sale en parte por construcción.
A6 conserva las cuotas por sede de A3-2030 y cambia la **distribución espacial** de la
demanda con un proxy zonal de adopción con fuente citable.

**Estado: COMPLETADO (2026-10-10)** — 6 corridas NEOS (S0/S1/S2 × H1/H2), las seis
`optimal`, timelimit 3600 s, mipgap 0.01, ε y κ del paper (0.03; 0.08, κ = 4).

## Método

Objetivo por zona *i* y sede *v* en 2030:

    T_iv(2030) = Q_v(2030) · [ s · π_iv + (1 − s) · g_iv^H ]

- `Q_v(2030)` = 15,853 / 5,906 / 1,680 (total 23,439): cuotas de A3, recalculadas en tiempo
  de ejecución con `run_a3.derivar_cuotas` (no hay cifras codificadas).
- `π_iv`: participación del paper (`make_targets`: prioridad ST-GNN × peso de sede).
- `s = VE(2024)/VE(2030) = 8,899 / 19,189 = 0.463755`: la flota de 2024 se supone ya
  distribuida como la señal observada; solo el **incremento** 2024→2030 (53.6 % de la flota
  de 2030) se reparte de forma diferenciada.
- `g_iv^H`, distribución espacial del incremento (cada columna suma 1, así que el total por
  sede se conserva exactamente):
  - **U (control):** `g = π` → reproduce A3-2030 al bit (diferencia máxima 5.7e-14).
  - **H1, adopción concentrada:** `g_iv ∝ π_iv · c_iv`. Para la sede residencial, `c` es la
    proporción de hogares de la zona con automóvil particular propio; para workplace y
    public, la proporción de los viajes que **llegan** a la zona hechos por miembros de
    hogares con automóvil (la demanda en destino no es residencial: así se evita asignar
    a un aeropuerto la tenencia de sus dos hogares encuestados).
  - **H2, convergencia/difusión:** `g_iv ∝ V_iv · w_iv`, con `V` = hogares expandidos
    (residencial) o viajes expandidos que llegan a la zona (workplace/public) y `w_iv` el peso
    de sede del paper. El incremento sigue el volumen y no la señal aprendida; equivale a un
    crecimiento inversamente proporcional a la intensidad actual de la señal por hogar
    (o por viaje), es decir, la adopción por hogar converge entre zonas.

Multiplicador implícito por zona `a_i = Σ_v T_iv / Σ_v T_iv^A3` (p5 / mediana / p95):
H1 0.597 / 0.966 / 1.737; H2 0.636 / 0.986 / 1.827. Masa de objetivo reubicada frente a
A3: H1 14.9 %, H2 9.3 %. Spearman de `a_i` con ν_i: H1 −0.328, H2 +0.292. Participación del
cuartil vulnerable en el objetivo total: A3 35.07 %, H1 29.75 %, H2 35.42 %
(`salidas/tablas/comprobaciones_objetivos.csv`).

## Fuente del proxy

Encuesta de Movilidad de Bogotá 2023 (Secretaría Distrital de Movilidad), la misma que
alimenta el aprendizaje del paper (Tabla de datos, fila *Household travel survey*):
módulos de hogares (22,755 hogares, `fexp_hg`), vehículos (17,392 registros) y viajes
(`fexp_vj`). MD5 y número de filas en `salidas/tablas/fuente_proxy_em2023.json`.

- Automóvil: `tipo_vehículo` ∈ {Automóvil, Campero/Camioneta}, placa privada, propiedad del
  propio hogar.
- Control: 24.97 % de los hogares expandidos del área de estudio tienen automóvil.
- Zonas con muestra < 30 (hogares en residencia o viajes en destino) toman el valor de su
  localidad (Bogotá) o municipio: residencia UPR4, UTAM117; destino UPR5, UTAM89.

**Validación interna del proxy (Spearman, 142 UTAM):** tenencia de automóvil vs. VE e
híbridos por hogar declarados en la misma encuesta 0.633; vs. estrato medio 0.876; vs. ν_i
−0.293; tenencia en destino vs. ν_i −0.398. Los VE + híbridos declarados son demasiado
escasos para usarse directamente (aparecen en 41 de 142 UTAM; 12,477 vehículos expandidos
incluidos los híbridos), pero se concentran donde hay más automóviles, que es lo que H1
supone. La literatura de difusión de VE describe adopción temprana en hogares de ingreso
alto con vehículo propio y estacionamiento privado; **el autor debe añadir la cita que
prefiera, verificada** (no hay una entrada de difusión/ingreso en el `.bib` actual).

## Supuestos declarados

1. Cuotas por sede de A3-2030 (con todos sus supuestos S1–S4).
2. El stock de 2024 se distribuye como la señal del paper; solo el incremento cambia.
3. La tenencia de automóvil de 2023 es un proxy de propensión a adoptar, no una proyección.
4. H1 y H2 son cotas de escenario, no pronósticos; ν_i, matriz de tiempos, pesos de sede y
   parámetros del modelo no cambian.
5. Umbral de muestra mínima 30, con respaldo de localidad/municipio.

## Resultados (2030, cuotas 23,439; `salidas/tablas/resumen_adopcion_heterogenea.csv`)

| Escenario | Term. | Gini pond. | Gini no pond. | Cobertura vuln. 15 min | Puertos/10k cuartil vuln. | Puertos/10k resto | Gini per cápita | ρ vs A3 mismo esc. | ρ vs S2 paper |
|---|---|---|---|---|---|---|---|---|---|
| S2 paper (2024) | archivada | 0.164653 | 0.169942 | 1.0 | 10.417 | 15.623 | 0.2958 | — | 1 |
| A3-2030 U, S2 | optimal | 0.164840 | 0.170141 | 1.0 | 22.495 | 33.638 | 0.2955 | 1 | 0.9998 |
| H1 S0 | optimal | 0.160878 | 0.165641 | 1.0 | 19.098 | 37.846 | 0.3676 | 0.8680 | 0.8691 |
| H1 S1 | optimal | 0.160758 | 0.165522 | 1.0 | 19.087 | 37.851 | 0.3678 | 0.8687 | 0.8692 |
| H1 S2 | optimal | 0.160675 | 0.165439 | 1.0 | 19.073 | 37.862 | 0.3680 | 0.8688 | 0.8694 |
| H2 S0 | optimal | 0.165261 | 0.170386 | 1.0 | 22.733 | 32.634 | 0.2263 | 0.9135 | 0.9127 |
| H2 S1 | optimal | 0.165133 | 0.170253 | 1.0 | 22.720 | 32.645 | 0.2265 | 0.9133 | 0.9130 |
| H2 S2 | optimal | 0.165069 | 0.170189 | 1.0 | 22.720 | 32.642 | 0.2266 | 0.9133 | 0.9134 |

Pérdida de fidelidad: 3.0 % (S1) y 8.0 % (S2) frente al u\* de cada hipótesis
(u\* = 111.543 en H1, 109.760 en H2, 106.679 en A3).

**Reasignación por la ponderación de vulnerabilidad (S2 vs S1, mismas cuotas):**
paper 8 puertos / 16 zonas; A3-2030 8 / 15; **H1 8 / 13; H2 5 / 8**.

**Cambio de geografía frente a A3-2030 (mismas cuotas, mismo escenario):** H1 mueve
3,471–3,479 puertos (≈ 14.8 % de 23,439) y cambia 140–141 zonas; H2 mueve 1,934–1,937
(≈ 8.3 %) y cambia 140–142 zonas.

Razón de provisión cuartil vulnerable / resto (S2): paper 0.667; A3-2030 0.669;
H1 0.504; H2 0.696.

### Lectura

1. **La invarianza de A3 era en efecto por construcción.** Con adopción diferenciada la
   geografía cambia de verdad (ρ ≈ 0.87–0.91 frente a > 0.9997 en A3) y los indicadores se
   mueven.
2. **H1 (concentrada) amplía la brecha de provisión:** el cuartil vulnerable baja de 22.50 a
   19.07 puertos/10k mientras el resto sube de 33.64 a 37.86; el Gini per cápita sube de
   0.2955 a 0.3680. **Pero el Gini de accesibilidad baja** (0.164840 → 0.160675). No es
   contradicción: A_i es una medida gravitatoria con decaimiento en distancia, de modo que
   los puertos de una zona aportan accesibilidad a sus vecinas; una explicación plausible
   (no verificada aquí zona a zona) es que las zonas de alta tenencia que ganan puertos
   irradian accesibilidad a zonas contiguas. Implicación: el Gini de accesibilidad solo no basta para juzgar el
   crecimiento; hay que leerlo junto con la intensidad per cápita.
3. **H2 (convergencia) cierra parcialmente la brecha:** cuartil vulnerable 22.72 vs. resto
   32.64; Gini per cápita 0.2266; Gini de accesibilidad prácticamente igual a A3 (0.165069).
4. **La reasignación inducida por ν sigue siendo pequeña** en los tres patrones de adopción
   (5–8 puertos): el resultado de §5.6 —crecer el programa no cambia lo que la regla
   redistribuye; lo que fija el gradiente es la regla y el patrón de demanda— se sostiene
   y se refuerza: el patrón de adopción mueve la provisión del cuartil vulnerable mucho más
   (−3.4 a +0.2 puertos/10k) que la ponderación por vulnerabilidad.
5. Cobertura a 15 min del cuartil vulnerable: completa en todas las corridas.

## Comando exacto

```bash
cd /Users/jorgeiv500/Documents/social-eq
# Corrida en seco: construye el proxy, objetivos y comprobaciones; sin NEOS
.venv/bin/python experiments/revision_2026/A6_adopcion_heterogenea/run_a6.py
# 6 corridas NEOS
.venv/bin/python experiments/revision_2026/A6_adopcion_heterogenea/run_a6.py --lanzar
```

Los microdatos EM-2023 se leen de `EM2023_DIR` (variable de entorno; por defecto la carpeta
de Drive del autor, la misma que usa `scripts/retrain_stgnn.py`). Si no están disponibles se
usa la tabla derivada archivada `salidas/tablas/proxy_adopcion_utam.csv`.

## Salidas (`salidas/tablas/`)

- `proxy_adopcion_utam.csv` — proxy por UTAM (bruto y con respaldo, muestras, estrato, VE).
- `fuente_proxy_em2023.json` — fuente, MD5, definiciones, zonas con respaldo, controles.
- `validacion_proxy.json` — correlaciones de validación.
- `comprobaciones_objetivos.csv`, `objetivos_2030.npz` — objetivos U/H1/H2 y chequeos.
- `referencias_sin_neos.csv` — métricas A3-2030 y S2 del paper (sin NEOS).
- `resumen_adopcion_heterogenea.csv` — resumen completo (precisión completa).
- `asignacion_<esc>_<H>.csv`, `manifiesto_<esc>_<H>.json` — por corrida.
- `../log_neos.txt` — registro de la corrida NEOS.

No se modificó `src/` ni `latex/`.
