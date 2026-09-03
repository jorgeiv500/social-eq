# Manifiesto de descarga — datos E2 (puntos de carga existentes en Bogotá)

Fecha de descarga de todo lo listado: **2026-09-01**.

## 1. `osm_overpass_bogota_charging.json` (crudo)

- **Fuente:** OpenStreetMap, vía Overpass API (espejo `lz4.overpass-api.de`).
- **URL del servicio:** https://lz4.overpass-api.de/api/interpreter
- **Consulta ejecutada (Overpass QL):**

  ```
  [out:json][timeout:120];
  area["wikidata"="Q2841"]->.a;
  (
    node["amenity"="charging_station"](area.a);
    way["amenity"="charging_station"](area.a);
    relation["amenity"="charging_station"](area.a);
  );
  out center tags;
  ```

  (`Q2841` = Bogotá, Distrito Capital; el área administrativa completa, no un rectángulo,
  para excluir Soacha, Mosquera, Chía, etc.)
- **Marca temporal de la base OSM:** `2026-09-01T16:55:25Z` (campo `osm3s.timestamp_osm_base` del propio JSON).
- **Conteo verificado:** **42 elementos** con `amenity=charging_station` (40 nodos, 2 vías).
- **Licencia:** Open Database License (ODbL) 1.0. Atribución: "© Colaboradores de OpenStreetMap".
- **Nota:** el primer intento contra `overpass-api.de` devolvió un error transitorio del
  despachador; el espejo `kumi.systems` devolvió *Internal Server Error*; la consulta
  exitosa fue contra `lz4.overpass-api.de`.

## 2. `osm_estaciones_bogota.csv` (derivado)

Tabla plana derivada del JSON anterior (mismo conteo: 42 filas): tipo e id OSM,
coordenadas (centroide para vías), nombre, operador, marca, capacidad (nº de plazas),
acceso, tarifa, conectores (`socket:*`), `motorcar`, horario. Generado con Python el
2026-09-01, sin filtrado ni imputación.

## 3. `ocm_bogota_intento_fallido.txt` (intento fallido, documentado)

- **Fuente intentada:** OpenChargeMap API v3.
- **URL intentada:** `https://api.openchargemap.io/v3/poi/?output=json&countrycode=CO&boundingbox=(4.45,-74.25),(4.85,-73.98)&maxresults=2000&compact=true&verbose=false`
- **Resultado (3 intentos, 2026-09-01):** la API rechaza toda consulta sin clave —
  `"You must specify an API key using the key query parameter or x-api-key header."`
  Se probó sin clave, con `key=test` y con cabecera `X-API-Key: test`; también el
  endpoint `/v3/referencedata/`. Todos rechazados.
- **Conclusión:** el conteo de OpenChargeMap para Bogotá **no quedó verificado**. La clave
  es gratuita registrándose en https://openchargemap.org (perfil → *my apps*); documentación
  en https://openchargemap.org/site/develop/api.
