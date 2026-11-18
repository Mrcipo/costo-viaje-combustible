# Calidad de datos: precios_historicos

Filas analizadas: 200,000

| columna | tipo | % vacíos | valores distintos | aviso |
|---|---|---|---|---|
| id_estacion | object | 0.0 | 380 | >255 categorías (no usar como categórica directa) |
| empresa | object | 0.0 | 285 | >255 categorías (no usar como categórica directa) |
| direccion | object | 0.0 | 380 | >255 categorías (no usar como categórica directa) |
| localidad | object | 0.0 | 240 |  |
| provincia | object | 0.0 | 22 |  |
| producto | object | 0.0 | 5 |  |
| horario | object | 0.0 | 2 |  |
| precio | float64 | 0.0 | 11,863 |  |
| fecha | datetime64[ns] | 0.0 | 32,352 |  |
| bandera | object | 0.0 | 13 |  |
| lat | float64 | 0.0 | 371 |  |
| lon | float64 | 0.0 | 371 |  |
| provincia_n | object | 0.0 | 22 |  |
| producto_n | object | 0.0 | 5 |  |
| fecha_original | object | 0.0 | 32,353 | >255 categorías (no usar como categórica directa) |

## Productos encontrados (nombre original -> clave)

- `Nafta (súper) entre 92 y 95 Ron` -> `nafta_comun` (811,817 filas)
- `Gas Oil Grado 2` -> `diesel_comun` (807,764 filas)
- `Gas Oil Grado 3` -> `diesel_premium` (801,416 filas)
- `Nafta (premium) de más de 95 Ron` -> `nafta_premium` (787,774 filas)
- `GNC` -> `gnc` (174,334 filas)

## Precios inválidos (<=0, vacíos o > 50.000): 235 de 3,383,105

## Fechas inválidas (vacías, antes de 2016 o futuras): 325. Ejemplos: 20/07/2007 10:30, 20/07/2007 10:30, 01/01/1572 00:00, 01/01/1572 00:00, 01/01/0008 08:00, 01/01/0008 08:00, 02/01/0002 19:45, 02/01/0002 19:45


Rango de meses: 2016-01 a 2026-11
Columnas originales: idempresa, empresa, direccion, localidad, provincia, idproducto, producto, idtipohorario, tipohorario, precio, fecha_vigencia, idempresabandera, empresabandera, latitud, longitud, geojson, anio, mes

## Cobertura: estaciones que informan nafta súper por mes (últimos 24 meses)

mes
2024-12-01    3637
2025-01-01    3643
2025-02-01    3594
2025-03-01    3576
2025-04-01    3557
2025-05-01    3513
2025-06-01    3407
2025-07-01    1625
2025-08-01    1480
2025-09-01    1238
2025-10-01    1103
2025-11-01    1098
2025-12-01     966
2026-01-01     690
2026-02-01     784
2026-03-01     892
2026-04-01     418
2026-05-01     731
2026-06-01     222
2026-07-01     334
2026-08-01     310
2026-09-01     697
2026-10-01     495
2026-11-01      16

OJO: la Res. 717/2025 (BO 2/6/2025) derogó la obligación de informar (Res. 314/2016); desde jul-2025 informan muchas menos estaciones.
Provincia-mes con menos de 3 estaciones (marcados no confiables): 551 de 13388
