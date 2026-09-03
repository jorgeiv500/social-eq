# E6 — Anclaje empírico de los parámetros de decaimiento λ (responde R1.7)

**Punto del revisor (R1.7):** los valores λ_res = 15, λ_work = 25, λ_pub = 10 minutos carecen
de justificación: ¿por qué el trabajo decae más lento y los servicios públicos más rápido?
¿Salen de literatura, calibración de encuesta o supuesto subjetivo?

**Qué hace este experimento:** calcula la distribución ponderada de duración de viaje por
propósito con el microdato de la Encuesta de Movilidad (módulo de viajes, Drive sincronizado)
y la compara contra el orden y la magnitud de los λ_v del manuscrito. No es una estimación
formal de una función de deterrencia: es un **anclaje empírico** — se verifica si el orden
λ_work > λ_res > λ_pub y las magnitudes (25/15/10 min) son consistentes con la tolerancia de
viaje observada por propósito. La respuesta al revisor se apoya además en la sensibilidad
±25 % ya publicada (`submission/jtg_2026-04/04_tablas/suplementarias/table_robustness.csv`,
filas *Low/High decay*).

## Insumo

- `/Users/jorgeiv500/Library/CloudStorage/GoogleDrive-jorgeiv500@gmail.com/Mi unidad/0. archivos para colab_sdewes/d. Modulo viajes.csv`
  (separador `;`, decimal `,`, latin-1; columnas usadas: `duracion_min`, `motivo_viaje`,
  `modo_principal_agrupado`, `nom_mun_des`, `fexp_vj`). Es el mismo módulo de viajes con el
  que se construyó el tensor del ST-GNN (ver `propuestas/r1/verificacion_insumos_B.md`, §1.2–1.3).

## Diseño registrado ex ante (2026-09-01, antes de calcular duraciones)

**Mapeo propósito → tipo de sede** (los propósitos no listados se excluyen y quedan en el
manifiesto):

| Tipo de sede del modelo | Propósitos EM incluidos |
|---|---|
| workplace | "A trabajar" |
| residential | "A regresar al hogar" (el destino es la vivienda) |
| public (núcleo) | "A realizar algún trámite personal", "A realizar compras", "A asuntos médicos personales" |
| public (extendido, robustez) | núcleo + "A realizar actividades recreativas y culturales", "A realizar actividades físicas y/o deportivas", "A asistir actividad de tipo religioso y/o culto" |

Excluidos por ambigüedad respecto de los tres tipos de sede del modelo: "A Estudiar",
"A visitar a alguien", "A acompañar a alguien (con/sin remuneración)", "A llevar y/o dejar
algo", "A buscar trabajo", "Conduzco vehículo como forma de trabajo", "NS/NR".

**Filtros:** destino en Bogotá (`nom_mun_des == "Bogotá"`; el modelo cubre solo las 142 UTAM
de Bogotá), `duracion_min > 0`, `fexp_vj > 0`.

**Modos:** especificación principal restringida a `AUTO` (el kernel de accesibilidad usa
tiempos de red vial y la población de interés son conductores de VE); variantes reportadas:
motorizado privado (AUTO + MOTO + TAXI OCUPADO) y todos los modos.

**Estadísticos:** media, mediana, p25 y p75 ponderados por `fexp_vj`; n muestral sin ponderar
y suma de pesos. Sin recortes de colas (la mediana/p25/p75 ya son robustos).

**Lectura acordada de antemano:**
- Si el orden empírico de duraciones es trabajo > regreso al hogar > servicios, respalda el
  orden λ_work > λ_res > λ_pub y la respuesta se redacta como calibración de orden con
  anclaje de magnitud.
- Si el orden no se cumple, se reporta tal cual y λ queda declarado como supuesto acotado por
  la sensibilidad ±25 % ya corrida. No se maquilla.

**Advertencia conceptual (va también en el texto):** `duracion_min` es duración puerta a
puerta autorreportada; λ_v es la escala de un kernel exponencial sobre tiempos de red
zona-a-zona. La comparación es de orden y magnitud, no una identidad estadística.

## Ejecución

```bash
cd /Users/jorgeiv500/Documents/social-eq
.venv/bin/python experiments/revision_2026/E6_lambdas/run_e6.py
```

Sin NEOS, sin aleatoriedad: cálculo determinista local. Salidas en `salidas/tablas/`:
`duracion_por_proposito.csv` (estadísticos por grupo × filtro de modo) y
`manifiesto_e6.json` (conteos, exclusiones, filtros, md5 del insumo).
