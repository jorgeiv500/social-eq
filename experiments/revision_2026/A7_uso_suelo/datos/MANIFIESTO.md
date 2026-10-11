# Manifiesto de descarga — datos A7 (uso del suelo, POT Decreto 555 de 2021)

Fecha de descarga y consulta de todo lo listado: **2026-10-10** (hora local de Bogotá ~20:10).

## Origen común

- **Servicio:** "Plan de Ordenamiento Territorial Bogotá. Decreto 555 de 2021. (Base de datos)",
  servicio de entidades de ArcGIS Online publicado por la cuenta `ideradmin` (cartografía oficial
  del POT: "Cartografía oficial del Plan de Ordenamiento Territorial de Bogotá, Decreto 555 de 2021
  'Bogotá reverdece 2021-2035'").
- **Ficha del ítem:** https://www.arcgis.com/home/item.html?id=ee7fed3b93dd4d52b8d00bcc81ce0ea0
  (metadatos JSON guardados en `metadatos_item_ee7fed3b.json`; última modificación del ítem
  2026-08-19).
- **Servicio REST:** https://services7.arcgis.com/lsxbLWF2l19Rmhqj/ArcGIS/rest/services/POT_Bogota_Decreto_555_2021/FeatureServer
- **Licencia:** Creative Commons Atribución 4.0 (CC BY 4.0), declarada en el campo
  `accessInformation`/`licenseInfo` del ítem (reconocer autoría; se permiten obras derivadas).
- **Atributos de procedencia de cada polígono:** `ACTO_ADMINISTRATIVO` = "Decreto 555 de 29 de
  diciembre de 2021"; `ESCALA_CAPTURA` = 1:2000; `FECHA_CAPTURA` = 2021-12-29.
- **CRS:** el servicio publica en Web Mercator (EPSG:3857, wkid 102100); los crudos se pidieron
  con `outSR=4326` (GeoJSON en WGS84, EPSG:4326). El cruce se hace en EPSG:9377 (igual que A4).
- **Salvedad de procedencia:** se intentó localizar el mismo conjunto en el catálogo CKAN de
  Datos Abiertos Bogotá (https://datosabiertos.bogota.gov.co, API `package_search`); las
  consultas expiraron (>120 s) el 2026-10-10 y no se pudo confirmar allí la ficha. La organización
  de ArcGIS que aloja el servicio (`lsxbLWF2l19Rmhqj`) es la misma que publica la capa
  "Capacidad eléctrica. Bogotá. POT Decreto 555 de 2021" (capa 136 del mismo servicio), cuyo
  equivalente se descargó del portal de Datos Abiertos Bogotá para el experimento A4.
- **Norma de referencia (definiciones):** Decreto 555 de 2021, arts. 240 (Áreas de Actividad del
  suelo urbano y de expansión urbana) y 241 (sectores de protección de características singulares),
  Régimen Legal de Bogotá: https://www.alcaldiabogota.gov.co/sisjur/normas/Norma1.jsp?i=119582
  (consultado 2026-10-10).

## 1. `area_actividad_pot555.geojson` (crudo)

- Capa 14: "Área de actividad. Bogotá. POT Decreto 555. 2021. 1:2000. (Capa)".
- URL exacta de la descarga:
  https://services7.arcgis.com/lsxbLWF2l19Rmhqj/ArcGIS/rest/services/POT_Bogota_Decreto_555_2021/FeatureServer/14/query?where=1%3D1&outFields=*&outSR=4326&f=geojson&resultRecordCount=2000
- 1 143 polígonos (conteo del servicio = 1 143; la respuesta no excedió el límite de transferencia).
- Códigos (`CODIGO_AREA_ACTIVIDAD`, conteo de polígonos): AAPRSU 288 (Proximidad, receptora de
  soportes urbanos), AAPGSU 168 (Proximidad, generadora de soportes urbanos), AAERAE 289
  (Estructurante, receptora de actividades económicas), AAERVIS 129 (Estructurante, receptora de
  vivienda de interés social), AAGSM 258 (Grandes Servicios Metropolitanos), PEMP 11 (Plan
  Especial de Manejo y Protección, BIC nacional).
- Metadatos de la capa: `metadatos_capa14.json`.
- SHA-256: `f5704b45b88a1998cb97dd7ae7fee7e405e3f2e55044f37ca4579b249a1f6cd4`.

## 2. `sectores_incompatibles_residencial_pot555.geojson` (crudo)

- Capa 16: "Sectores incompatibles al uso residencial. Bogotá. POT Decreto 555 de 2021. 1:2000.
  (Capa)" (art. 241 num. 3: "Zonas en las que no se permite la localización del uso residencial").
- URL exacta: https://services7.arcgis.com/lsxbLWF2l19Rmhqj/ArcGIS/rest/services/POT_Bogota_Decreto_555_2021/FeatureServer/16/query?where=1%3D1&outFields=*&outSR=4326&f=geojson&resultRecordCount=2000
- 15 polígonos. Metadatos: `metadatos_capa16.json`.

Las sumas SHA-256 de todos los archivos de esta carpeta están en `SHA256SUMS`.

## Cobertura de la capa

La capa zonifica solo **suelo urbano y de expansión** (art. 240) y no cubre espacio público
(vías, parques, estructura ecológica). Por eso la fracción de compatibilidad se calcula sobre el
**suelo zonificado** de cada UTAM, no sobre su área total. 108 de las 142 UTAM tienen cobertura
≥ 5 %; las 34 restantes (UPR rurales y borde) quedan sin tope (ver README).
