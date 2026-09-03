# Manifiesto de descarga — datos A4 (proxy de capacidad eléctrica por zona)

Fecha de descarga de todo lo listado: **2026-09-01**.

## 1. `capacidad_electrica_pot555.gpkg` (crudo)

- **Fuente:** Datos Abiertos Bogotá — conjunto "Capacidad eléctrica. POT Bogotá D.C.",
  publicado por la Secretaría Distrital de Planeación (SDP).
- **Página del conjunto:** https://datosabiertos.bogota.gov.co/dataset/capacidad-electrica-bogota-d-c
- **URL exacta del recurso descargado (GPKG):**
  https://datosabiertos.bogota.gov.co/dataset/1f2d0d54-c08c-4042-b9dc-e6571fea5dd3/resource/3d0804d8-ba0b-4881-9a40-49708c31d5e1/download/capacidad_electrica.gpkg
- **Contenido verificado:** 154 multipolígonos (463 polígonos simples en la versión GeoJSON),
  áreas de servicio de **31 subestaciones** con atributos `ESTACION`, `CAPACIDAD_` (MVA,
  red de 11.4 kV), `CAPACIDAD1` (MVA, red de 34.5 kV), `CARGA_11`, `CARGA_34` (factores de
  utilización), `AREA_KM2`. Fecha de captura declarada en los atributos: 29/12/2021
  (Decreto 555 de 2021, POT). Última actualización del metadato: 2022-11-16.
- **Licencia:** Creative Commons Attribution 4.0 (CC BY 4.0), declarada en el portal.
- **CRS:** el GPKG trae el plano cartesiano local de Bogotá («PCS_CarMAGBOG»,
  equivalente a la malla urbana MAGNA-SIRGAS de la ciudad). **Advertencia:** el recurso
  GeoJSON del mismo conjunto declara EPSG:4326 pero contiene estas coordenadas planas
  (x∈[83.8k, 107.2k], y∈[85.9k, 125.7k]); usar el GPKG, no el GeoJSON, o reasignar el CRS.

## 2. `cap_i_demo_utam.csv` (derivado, demostración)

`cap_i` de demostración por UTAM: holgura de cada subestación
(`CAPACIDAD_ × (1 − CARGA_11)`, en MVA) repartida entre las UTAM proporcionalmente al
área intersectada de su zona de servicio. Cruce espacial (EPSG:9377) entre este GPKG y el
shapefile `UTAM2023` ya presente en el repositorio
(`.tmp_build/utam_shp/UTAM2023/UTAM2023.shp`, 142 unidades). Generado con
geopandas el 2026-09-01. Es una ilustración de viabilidad, no un insumo final.
