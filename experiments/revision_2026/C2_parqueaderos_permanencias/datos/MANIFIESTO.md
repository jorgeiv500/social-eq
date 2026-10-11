# Manifiesto de datos — C2 (parqueaderos candidatos y permanencias)

Fecha de consulta y descarga de todo lo listado: **2026-10-11** (hora local de Bogotá, 11:25–11:58).
Las sumas SHA-256 de los crudos guardados en esta carpeta están en `SHA256SUMS`.

## 1. OpenStreetMap — `amenity=parking`

### 1a. Overpass API (FALLÓ; documentado)

Consulta intentada (Overpass QL), primero por área administrativa y luego por caja:

```
[out:json][timeout:180];area["wikidata"="Q2841"]->.a;
(node["amenity"="parking"](area.a);way["amenity"="parking"](area.a);relation["amenity"="parking"](area.a););
out center tags;

[out:json][timeout:180][bbox:4.45,-74.25,4.84,-73.98];(nwr["amenity"="parking"];);out center tags;
```

| Hora (UTC) | Servidor | Resultado |
|---|---|---|
| 16:25:06 | https://overpass-api.de/api/interpreter | HTTP 406 (rechazo del agente de usuario por defecto de curl) |
| 16:25:37 | https://lz4.overpass-api.de/api/interpreter | HTTP 406 (ídem) |
| 16:26:13 | https://overpass-api.de/api/interpreter (con agente identificado) | HTTP 504, tiempo de espera del servidor |
| 16:28:20 | https://overpass.kumi.systems/api/interpreter | HTTP 500 *Internal Server Error* |
| 16:29:06 | https://overpass.openstreetmap.ru/api/interpreter | sin respuesta (código 000) |
| 16:36:48 | https://overpass-api.de/api/interpreter (consulta por caja) | HTTP 504 |

Entre intentos se esperaron 30–45 s. No se insistió más.

### 1b. Extracto Geofabrik (LO QUE FUNCIONÓ)

- **URL:** https://download.geofabrik.de/south-america/colombia-latest.osm.pbf → redirige a
  https://download.geofabrik.de/south-america/colombia-261009.osm.pbf
- **Fecha de los datos:** extracto del 2026-10-09 (`Last-Modified: Fri, 09 Oct 2026 22:53:48 GMT`).
- **Tamaño / MD5:** 331 462 401 bytes; MD5 `3ab3809b82d65af0d401d61f508a3fc9` (coincide con
  el `.md5` publicado por Geofabrik).
- **No se versiona** (331 MB). Se guardó fuera del repositorio; para reproducir, descargar la URL
  anterior y pasar la ruta con `--pbf`.
- **Lectura:** controlador OSM de GDAL 3.10.3 vía `pyogrio` (no hay osmium ni pyrosm en el `.venv`).
  Capas `points` (nodos), `multipolygons` (vías cerradas y relaciones multipolígono) y `lines`
  (vías abiertas), filtradas por `amenity=parking` y por la caja lon −74,25…−73,98, lat 4,45…4,84.
  El filtro definitivo es espacial: solo quedan los que caen dentro de una ZAT de Bogotá
  (MUNCod 11001). Se descartaron 323 elementos de la caja fuera de Bogotá.
- **Licencia:** Open Database License (ODbL) 1.0, «© Colaboradores de OpenStreetMap».
- **CRS de origen:** WGS84 (EPSG:4326).

## 2. Zonas de Parqueo Pago. Bogotá D. C. (Terminal de Transporte S.A.)

- **Ficha CKAN:** https://datosabiertos.bogota.gov.co/dataset/zona-de-parqueo-pago
  (id `7e4e45a6-33ab-4227-92ec-544df31a7cf4`; metadatos completos de la API en
  `metadatos_ckan_zona_parqueo_pago.json`, obtenidos con
  `https://datosabiertos.bogota.gov.co/api/3/action/package_search?q=%22parqueo%20pago%22`).
  Fecha de cita de la ficha: 2026-04-30; creada en el portal el 2026-07-02.
- **Licencia:** Creative Commons Atribución 4.0 (CC-BY-4.0), según la ficha.
- **Descarga:** la descarga directa del GeoJSON del portal
  (`.../resource/88fd7256-39f9-4bf9-9d78-14c356657302/download/zonas_de_parqueo_pago.geojson`)
  expiró sin respuesta. Se usó el **servicio REST que la misma ficha enlaza** como recurso oficial:
  https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/movilidad/zonasparqueopago/MapServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson
- **Archivo:** `zonas_parqueo_pago_rest.geojson` — 862 tramos (conteo del servicio = 862; sin
  `exceededTransferLimit`). Metadatos de la capa: `metadatos_zonasparqueopago_capa0.json`.
- **Geometría:** líneas (tramos viales). Campos de cupos: `CUP_AUT_` (automóviles, usado como
  capacidad), `CUP_ELEC_` (eléctricos), `CUP_TOTAL_`, entre otros.
- **CRS:** el servicio publica en EPSG:3857; se pidió en EPSG:4326.
- **Autoría declarada en la capa:** Terminal de Transportes de Bogotá – TTSA; IDECA.

## 3. Catastro (UAECD) — «Grupo de uso económico predominante por manzana»

- **Servicio:** https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/catastro/usopredominante/MapServer/0
  (metadatos en `metadatos_catastro_usopredominante_capa0.json`). El mismo conjunto está en el
  portal como https://datosabiertos.bogota.gov.co/dataset/grupo_uso-por-manzana (SHP por
  vigencias; licencia CC BY 4.0 según el portal).
- **¿Hay categoría de parqueaderos?** Sí, pero **combinada**: el dominio de `GRUPOUSOECON` es
  RESIDENCIAL, BODEGAS, COMERCIO Y OFICINAS, **DEPÓSITOS Y PARQUEADEROS**, DOTACIONAL, INDUSTRIAL,
  OTROS. La categoría mezcla depósitos y parqueaderos y designa el uso PREDOMINANTE por área
  construida de la manzana. No se puede aislar «solo parqueaderos» con esta capa.
- **Vigencia usada:** 2026 (`ANO` = 2026-01-01). En 2026 la etiqueta viene sin tilde
  («DEPOSITOS Y PARQUEADEROS», 336 manzanas); en 2017–2025 con tilde (288–330 manzanas por año).
- **URL exacta:** en `catastro_url.txt`
  (`where=GRUPOUSOECON='DEPOSITOS Y PARQUEADEROS' AND ANO=date '2026-01-01'`, `outSR=4326`).
- **Archivo:** `catastro_manzanas_depositos_parqueaderos_2026.geojson` — 336 manzanas, sin
  `exceededTransferLimit`.
- **Licencia:** CC BY 4.0 (portal de Datos Abiertos Bogotá); autoría «Unidad Administrativa
  Especial de Catastro Distrital – UAECD».
- **CRS de origen:** el servicio publica en WGS84 (EPSG:4326, wkid 4326) y se pidió en EPSG:4326.

## 4. Insumos locales ya presentes en el proyecto (no descargados hoy)

| Insumo | Ruta | CRS |
|---|---|---|
| ZAT 2023 (1 215 polígonos, 940 de Bogotá) | `$EM2023_DIR/ZAT2023.shp` | EPSG:3116 (MAGNA-SIRGAS / Bogotá) |
| UTAM 2023 (142) | `outputs/social_equity_followup_outputs/utam_shp/UTAM2023/UTAM2023.shp` | EPSG:4686 |
| EM-2023, módulos hogares, personas, viajes, etapas (CSV `;`, latin-1) | `$EM2023_DIR/` | — |
| 42 puntos de carga OSM | `datos/osm_estaciones_bogota.csv` (consulta Overpass del 2026-09-01, ODbL) | EPSG:4326 |
| Cuartil de ν y población DANE por UTAM | `experiments/revision_2026/B5_nucleo_robusto/salidas/b5_zonas_S2.csv` | — |

`EM2023_DIR` por defecto: `~/Library/CloudStorage/GoogleDrive-.../Mi unidad/0. archivos para colab_sdewes`
(la misma carpeta que usa `scripts/retrain_stgnn.py`). Todos los cruces espaciales se hacen en
**EPSG:9377** (MAGNA-SIRGAS / Origen-Nacional).

### Diccionario de la EM-2023

No hay diccionario de datos en la carpeta de Drive ni en el repositorio, y no se encontró en
línea el 2026-10-11 (búsquedas en bogota.gov.co y el Observatorio de Movilidad). El significado
de `fecha` (hogares) y el formato de `hora_ini`/`hora_fin` se verificaron con los propios datos
(ver README, §2.1).
