# Calidad de datos: precios_vigentes

Filas analizadas: 36,613

| columna | tipo | % vacíos | valores distintos | aviso |
|---|---|---|---|---|
| indice_tiempo | object | 0.0 | 117 |  |
| id_estacion | object | 0.0 | 4,605 | >255 categorías (no usar como categórica directa) |
| empresa | object | 0.0 | 3,477 | >255 categorías (no usar como categórica directa) |
| direccion | object | 0.0 | 4,588 | >255 categorías (no usar como categórica directa) |
| localidad | object | 0.0 | 1,064 | >255 categorías (no usar como categórica directa) |
| provincia | object | 0.0 | 24 |  |
| producto | object | 0.0 | 5 |  |
| horario | object | 0.0 | 2 |  |
| precio | float64 | 0.0 | 2,923 |  |
| fecha | datetime64[ns] | 0.0 | 6,928 |  |
| bandera | object | 0.0 | 11 |  |
| lat | float64 | 0.0 | 4,468 |  |
| lon | float64 | 0.0 | 4,468 |  |
| provincia_n | object | 0.0 | 24 |  |
| producto_n | object | 0.0 | 5 |  |
| fecha_original | object | 0.0 | 6,928 | >255 categorías (no usar como categórica directa) |

Precios inválidos: 0 de 36,613
Fechas de vigencia: 2016-06-01 11:58:00 a 2026-11-14 06:00:00
Columnas originales: indice_tiempo, idempresa, cuit, empresa, direccion, localidad, provincia, region, idproducto, producto, idtipohorario, tipohorario, precio, fecha_vigencia, idempresabandera, empresabandera, latitud, longitud, geojson

## Productos

- `Gas Oil Grado 2` -> `diesel_comun` (8,337)
- `Nafta (súper) entre 92 y 95 Ron` -> `nafta_comun` (8,315)
- `Gas Oil Grado 3` -> `diesel_premium` (8,163)
- `Nafta (premium) de más de 95 Ron` -> `nafta_premium` (8,090)
- `GNC` -> `gnc` (3,708)

Estaciones sin coordenadas válidas: 18

## Estado de los precios 'vigentes'

estado          desactualizado (> 120 días)  fuera de rango    ok
producto_n                                                       
diesel_comun                           6772              10  1555
diesel_premium                         6586              21  1556
gnc                                    2911               2   795
nafta_comun                            6718              22  1575
nafta_premium                          6551              14  1525
