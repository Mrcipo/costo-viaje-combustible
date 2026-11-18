# ¿Cuánto me sale el viaje en combustible?

Trabajo práctico de **Infraestructura para la ciencia de datos**. Calculadora del costo de combustible de un
viaje por Argentina: compara manejar a ~100 km/h y a ~130 km/h, con el precio del litro estimado por
provincia para la fecha del viaje, el clima de ese día y las estaciones más baratas cerca de la ruta.

**App online:** _(link pendiente)_

## Qué hay adentro
- **Datos públicos descargados por código**: precios en surtidor (Secretaría de Energía, 3,3 millones de
  registros 2016–2026), inflación y dólar, provincias y localidades (Georef), rutas (OSRM / OpenStreetMap),
  clima (Open-Meteo) y consumos oficiales de 1.316 versiones de autos (Ministerio de Ambiente).
- **Calidad de datos**: fechas imposibles, precios inválidos, outliers por estación y el quiebre de la
  Res. 717/2025 (desde jul-2025 informan muchas menos estaciones).
- **Detección de anomalías**: z-score móvil, STL e Isolation Forest sobre la serie mensual por provincia.
- **Predicción**: precio del litro a 1, 2 y 3 meses con Gradient Boosting, validado en el tiempo
  (walk-forward) contra modelos ingenuos.
- **Modelo físico 100 vs 130 km/h**, calibrado con pruebas reales: a 130 km/h se gasta ~32% más.

El notebook ejecutado está en [`reportes/TP_costo_viaje_combustible.html`](reportes/TP_costo_viaje_combustible.html)
y el detalle de cómo correr todo con Docker, en [`LEEME.md`](LEEME.md).

## Correrlo en tu computadora (Docker)
```powershell
docker compose build
docker compose up app        # app: http://localhost:8501 · Jupyter: http://localhost:8888
```
Los datos procesados ya vienen en el repositorio. Para descargar y procesar todo de nuevo:
`docker compose run --rm datos`.

## Fuentes y licencias
Precios: Secretaría de Energía (CC-BY 4.0). Rutas: © colaboradores de OpenStreetMap (ODbL), vía OSRM.
Clima: Open-Meteo (CC-BY 4.0). Provincias y localidades: API Georef. Inflación y dólar: ArgentinaDatos.
Consumos: Ministerio de Ambiente, Licencias de Configuración Ambiental. Detalle en [`LEEME.md`](LEEME.md).
