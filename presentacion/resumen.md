# ¿Cuánto cuesta un viaje en combustible? — Resumen para la presentación

Todos los números salen de la corrida del 06/11/2026 con datos reales (`reportes/` y el notebook ejecutado).

## 1. Problema
Antes de salir a la ruta: ¿cuánto voy a gastar en combustible, y cuánto más me cuesta ir a ~130 km/h
en vez de ~100 km/h? Con el precio del litro **de la fecha del viaje** en cada provincia que cruzo y el
clima esperado ese día.

## 2. Datos (todos públicos, descargados por código: `src/descargar.py`)
- **Precios en surtidor por estación** (Secretaría de Energía, Res. 314/2016, CC-BY): histórico de
  **3.383.105 registros** (2016-2026) + precios vigentes (36.613 filas estación-producto).
- **Inflación y dólar** (ArgentinaDatos) para pasar a pesos constantes y US$.
- **Rutas** (OSRM / OpenStreetMap), **provincias** (API Georef: 24 provincias, 4.037 localidades),
  **clima** (Open-Meteo: pronóstico ≤ 15 días o promedio de 10 años).
- **Consumos oficiales** (Ministerio de Ambiente, Licencias de Configuración Ambiental): 1.654 configuraciones
  de 94 marcas. Tras separar homologaciones compartidas ("PEUGEOT / CITROËN") y unificar marcas ("212" → BAIC,
  "MERCEDES-AMG" → MERCEDES-BENZ): **1.316 versiones de 65 marcas** en la app (nafta/diésel, sin problemas, sin
  duplicados). **No informa el año**: la app elige marca → modelo → versión.
- **Base propia**: 27 versiones con año y fuente por fila + 16 pruebas reales a velocidad constante.
  Las filas tomadas de etiquetas oficiales coinciden con la base del Ministerio (0-1% de diferencia); las de
  catálogos y del ciclo brasileño difieren 12-26% (estaban marcadas como aproximadas).

## 3. Calidad de datos: lo que una demo no muestra
- 325 fechas imposibles (años 1572, 0008...) y 235 precios inválidos (≤ 0 o absurdos): descartados.
- 40.324 estación-mes con precio fuera de rango contra su provincia (z robusto): excluidos de la mediana.
- El archivo de "precios vigentes" trae ~80% de precios con más de 120 días: sólo se usan los recientes.
- **Quiebre**: la Res. 717/2025 (2/6/2025) derogó la obligación de informar precios. Estaciones que informan
  nafta súper por mes: ~3.576 (ene-may 2025) → ~817 (desde jul-2025). Tratamiento: arrastre máximo de 3 meses
  y provincia-mes con menos de 3 estaciones excluidos (551 de 13.388).
- Base oficial de consumos: 83 filas sin ningún consumo, valores fuera de rango, modelos mezclados en una misma
  celda: se marcan y se excluyen con el motivo.

## 4. Método (pipeline de los 3 ejercicios de referencia)
1. **API / descarga → CSV**: descarga reanudable, valida que el archivo esté completo, registra URL y fecha.
2. **Anomalías**: z-score móvil, STL e Isolation Forest sobre el precio real mensual (provincia × combustible);
   z robusto (mediana/MAD) para estaciones. Errores de datos se sacan; saltos reales se mantienen.
3. **Predicción** a 1-3 meses: Ridge y Gradient Boosting contra **ingenuo (último precio)** e
   **ingenuo + inflación**, con validación temporal walk-forward (sin mezclar al azar).
4. **Física 100 vs 130 km/h**: consumo = c100 · [(1−f) + f·(v/100)²], k y f calibrados con pruebas reales.

## 5. Resultados
- **Anomalías**: 465 provincia-mes detectados por 2+ métodos; los grandes coinciden con eventos reales en las
  24 provincias a la vez: sep-2018 (+18,5% nafta súper, dólar +26%), congelamiento de sep-2023 (0% nominal,
  −12% real) y dic-2023 (+52% nominal, dólar +57%).
- **Predicción** (error porcentual medio del precio del litro):

  | Meses | Gradient Boosting | Ingenuo + inflación | Último precio |
  |---|---|---|---|
  | 1 | 3,2% | 3,5% | 4,3% |
  | 2 | 4,2% | 5,1% | 5,8% |
  | 3 | 6,3% | 6,8% | 8,8% |

  Lectura honesta: el modelo tiene **menos error típico** (MAPE y MAE), pero en **RMSE** (errores grandes)
  queda igual o peor que "ingenuo + inflación" a 1 y 3 meses ($101 vs $100; $163 vs $150), y el **R² del
  cambio de precio ≈ 0**: ningún modelo anticipa saltos bruscos (devaluaciones). Por eso la app muestra un rango.
  El R² sobre el precio en pesos (~0,95) da alto hasta para el ingenuo: es una métrica engañosa acá.
- **Velocidad**: a 130 km/h se gasta **~32% más** por km que a 100 km/h (rango 27–38%), con ~23% menos tiempo
  de manejo. Validación dejando uno afuera: error mediano **5%** a 130 km/h.
- **Ejemplo real** (Córdoba → Mendoza, 689 km por Córdoba, San Luis y Mendoza): $85.839 a 100 km/h vs
  $113.443 a 130 km/h (+32%, +$27.604), 6,9 h vs 5,3 h de manejo, con estaciones baratas sobre la ruta.

## 6. Limitaciones
- La base oficial de consumos no informa el año del modelo; las pruebas a velocidad constante son 16 y de
  pocas fuentes.
- Ajustes por clima con supuestos simples (aire acondicionado, lluvia); la ruta no considera pendientes ni tránsito.
- Desde jul-2025 hay pocas estaciones informando: en provincias chicas el precio sale de 1-2 estaciones (la app lo avisa).
- Más allá de 3 meses el precio se extrapola con un supuesto (2% mensual): poco confiable.

## 7. Cómo convertirlo en un servicio
- **Fletes y flotas**: presupuesto de combustible por viaje y política de velocidad (≈32% más de combustible a
  130 km/h es una decisión de costos), dónde cargar en la ruta.
- **Apps de viaje / navegación**: "costo del viaje" junto al tiempo estimado, con la estación más barata sobre la ruta.
- **Datos**: API con precio esperado por provincia y fecha, actualizada con los precios vigentes; reentrenamiento
  mensual. Riesgo: depende de que las estaciones sigan informando precios (ver quiebre 2025).
