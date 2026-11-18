# Calculadora de costo de combustible de un viaje (TP Ciencia de Datos)

El usuario elige vehículo, origen, destino y fecha; la app estima cuánto gasta en combustible
a ~100 km/h y a ~130 km/h, con el precio estimado por provincia y el clima de ese día.

## Ver la app online
App publicada en Streamlit Community Cloud: https://costo-viaje-combustible.streamlit.app . Usa los datos procesados incluidos en el
repositorio; la ruta y el clima se consultan en vivo.

## Requisitos
- Docker Desktop abierto (no hace falta Python instalado).
- Abrir **PowerShell** en la carpeta del proyecto (donde está este archivo):
  ```powershell
  cd <carpeta donde lo descargaste>\costo_viaje_combustible
  ```

## Comandos (PowerShell)

| Qué | Comando | Qué mirar |
|---|---|---|
| 1. Construir la imagen (la primera vez ~20 min; después ~1 min) | `docker compose build` | `Image costo-viaje:1.0 Built`, sin `ERROR`. |
| 2. Descargar TODOS los datos | `docker compose run --rm datos python -m src.descargar todo` | Bloque `RESUMEN DESCARGA`: todo en `OK`. Si el histórico dice `INCOMPLETA`, volvé a correr el mismo comando. |
| 3. Procesar + analizar + ejecutar el notebook (sin volver a bajar) | `docker compose run --rm datos python -m src.pipeline --sin-descarga` | Al final: `notebook ejecutado sin errores` y `copia HTML`. Los resultados quedan en `reportes\` y en el notebook. |
| (alternativa) Bajar y procesar todo junto | `docker compose run --rm datos` | Igual que 2 + 3. |
| Solo volver a ejecutar el notebook | `docker compose run --rm datos python -m src.pipeline --solo-notebook` | `notebook ejecutado sin errores`. |
| 4. Levantar la app y Jupyter | `docker compose up app` | App: http://localhost:8501 · Jupyter: http://localhost:8888 . Para cortar: `Ctrl+C`. |
| Tests con datos sintéticos (sin internet, en carpeta temporal) | `docker compose run --rm datos python -m pytest -q tests` | `10 passed`. |

Si cambiás `requirements.txt` o el `Dockerfile`, volvé a correr `docker compose build`. Si cambiás código,
**no** hace falta: la carpeta del proyecto está montada dentro del contenedor.

Para ver los resultados sin instalar nada: abrí `reportes\TP_costo_viaje_combustible.html` en el navegador.

## Estructura
```
src/descargar.py   paso 1: API/descarga -> CSV (precios, inflación, dólar, provincias, localidades, consumos oficiales)
src/etiqueta.py    base OFICIAL de consumos (Ministerio de Ambiente, LCA): limpieza y validación
src/procesar.py    calidad de datos + normalización + serie mensual por provincia y producto
src/analisis.py    paso 2: anomalías (z-score móvil, STL, Isolation Forest) / paso 3: predicción 1-3 meses
src/consumos.py    base de consumos + modelo físico 100 vs 130 km/h (calibrado con pruebas reales)
src/viaje.py       ruta (OSRM), provincias (Georef), clima (Open-Meteo), precio y costo
app/app.py         app Streamlit (embudo de selección + resultados)
notebooks/         notebook principal del TP
data/ref/          base de consumos armada a mano, con la fuente de cada fila
data/processed, geo  datos ya procesados (se incluyen para que la app funcione sin descargar nada)
data/raw, cache      se generan al correr (no se suben: el histórico pesa ~770 MB)
reportes/          reportes de calidad y resultados
presentacion/      resumen para la presentación
```

## Fuentes y licencias
- Precios en surtidor: Secretaría de Energía, Res. 314/2016 — datos.energia.gob.ar — CC-BY 4.0.
- Inflación y dólar: ArgentinaDatos (api.argentinadatos.com).
- Provincias/localidades y ubicación de puntos: API Georef (apis.datos.gob.ar/georef).
- Rutas: OSRM (servidor público de demostración, datos © OpenStreetMap, ODbL). Uso liviano: 1 pedido/seg.
- Clima: Open-Meteo (pronóstico y archivo histórico), CC-BY 4.0.
- Consumos oficiales: Ministerio de Ambiente, Licencias de Configuración Ambiental (CIAM, dataset 566,
  https://ciam.ambiente.gob.ar/repositorio.php?tid=10&stid=68&dsid=566). No informa el año del modelo.
- Base propia (27 versiones, con año): etiquetas IRAM/AITA 10274 y fichas oficiales, citadas fila por fila en
  `data/ref/consumos_vehiculos.csv`; pruebas a velocidad constante en `data/ref/consumos_velocidad_constante.csv`.
- Nota: la Res. 717/2025 (BO 2/6/2025) derogó la obligación de informar precios (Res. 314/2016); desde
  jul-2025 hay muchas menos estaciones en los datos de precios.
