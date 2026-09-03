# Insumos de aprendizaje sustituidos (versión enviada en abril de 2026)

Estos tres archivos son los que alimentaron la versión del artículo enviada a la revista en
abril de 2026. Se conservan para que la diferencia con los insumos corregidos pueda
reproducirse, no para usarse.

El notebook que los generó emparejaba los códigos de zona como cadenas literales: los
módulos de la encuesta traen los códigos con ceros a la izquierda (`UTAM043`, `UPR0002`) y el
shapefile no (`UTAM43`, `UPR2`), así que solo casaron las 44 zonas de código de tres dígitos y
la etiqueta de entrenamiento quedó en cero en 98 de las 142 zonas. Además, los factores de
expansión con coma decimal se leyeron como cero y los viajes no se ponderaron por su factor.

Los insumos vigentes están en `data/inputs/` y se regeneran con `scripts/retrain_stgnn.py`.
El detalle de la corrección está en `data/inputs/input_manifest.json` (clave `superseded`).
