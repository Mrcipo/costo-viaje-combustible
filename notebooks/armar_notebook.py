import json, sys

celdas = []
def md(t): celdas.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})
def code(t): celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")})

md("""
# TP: ¿Cuánto cuesta un viaje en combustible? 100 vs 130 km/h

**Problema.** Antes de un viaje en auto queremos saber cuánto vamos a gastar en combustible, y cuánto
más cuesta ir rápido (~130 km/h) en vez de tranquilo (~100 km/h), con el precio del litro que habrá
**en la fecha del viaje** en cada provincia que atravesamos y el clima esperado ese día.

El trabajo sigue el pipeline de los tres ejercicios de referencia:

| Paso | Ejercicio de referencia | Acá |
|---|---|---|
| 1 | Web scraping / API → CSV | Descarga de precios por estación (Secretaría de Energía), inflación y dólar (ArgentinaDatos), provincias (Georef) |
| 2 | Detección de anomalías en series de tiempo | Saltos de precio (z-score móvil, STL, Isolation Forest) y estaciones con precios fuera de rango |
| 3 | Predicción con machine learning | Precio del litro a 1-3 meses por provincia, contra un modelo ingenuo, con validación temporal |

Además: una base propia de consumos de vehículos (con la fuente de cada fila) y un modelo físico simple
para pasar de 100 a 130 km/h, calibrado y validado con pruebas reales.

> Para correr este notebook: `docker compose run --rm datos` (descarga + procesamiento) y después
> `docker compose up app` → Jupyter en http://localhost:8888.
""")

code("""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src import config as C, consumos, analisis
pd.set_option("display.max_columns", 30); pd.set_option("display.width", 160)
plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
print("Datos en:", C.DATA)
""")

md("""
## 1. Datos (paso "API / descarga → CSV")

`src/descargar.py` baja todo con `requests`, en streaming y con reanudación (el histórico es grande),
respetando las fuentes (un pedido a la vez, pausas cortas). Todo queda como CSV en `data/raw`.
""")
code("""
meta = json.loads((C.RAW / "_descargas.json").read_text()) if (C.RAW / "_descargas.json").exists() else {}
pd.DataFrame([{"fuente": k, "descargado": v.get("descargado"), "MB": round(v.get("bytes", 0) / 1e6, 1) or None}
              for k, v in meta.items()])
""")
md("""
### 1.1 Calidad de los datos

Antes de usar nada revisamos cada archivo: columnas vacías, columnas con demasiadas categorías
(más de 255), precios imposibles (≤ 0 o absurdamente altos, típicamente una coma corrida al cargar),
provincias escritas de distintas formas ("CAPITAL FEDERAL" vs "Ciudad Autónoma de Buenos Aires") y
productos. El reporte completo queda en `reportes/calidad_*.md`.
""")
code("""
from IPython.display import Markdown
Markdown((C.REPORTES / "calidad_precios_historicos.md").read_text())
""")
code("""
Markdown((C.REPORTES / "calidad_precios_vigentes.md").read_text())
""")
md("""
### 1.2 Un quiebre en los datos: la derogación de la Res. 314/2016

La Res. 314/2016 obligaba a las estaciones a informar cada cambio de precio. La **Res. 717/2025 del
Ministerio de Economía (Boletín Oficial del 2/6/2025) la derogó**. Desde julio de 2025 informan muchas menos
estaciones. Consecuencias que tomamos en cuenta:
- el último precio de una estación se "arrastra" como mucho 3 meses (si lo arrastráramos más, estaciones que
  dejaron de informar hace un año bajarían la mediana artificialmente, porque los precios suben);
- una provincia-mes con menos de 3 estaciones se considera **no confiable** y no entra al modelo;
- el archivo de "precios vigentes" trae muchos precios viejos: sólo usamos los de los últimos 120 días.
""")
code("""
cob = pd.read_csv(C.REPORTES / "cobertura_mensual.csv", parse_dates=["mes"]).set_index("mes")
cob = cob[cob.index < pd.Timestamp.today().to_period("M").to_timestamp()]
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(cob.index, cob.estaciones_que_informan, color="tab:blue")
ax.axvline(pd.Timestamp("2025-06-02"), color="tab:red", ls="--", label="Res. 717/2025 (2/6/2025)")
ax.set_title("Estaciones que informan precio de nafta súper cada mes"); ax.legend(); plt.show()
print("Promedio mensual ene-may 2025:", int(cob.loc["2025-01":"2025-05"].mean().iloc[0]),
      "| promedio desde jul-2025:", int(cob.loc["2025-07":].mean().iloc[0]))
""")
md("""
**Cómo pasamos de reportes de estaciones a una serie mensual.** Las estaciones informan su precio
cuando cambia. Para cada estación, producto y horario tomamos el último precio informado de cada mes y lo
mantenemos vigente hasta 3 meses si no vuelve a informar (ver 1.2 por qué no más). Después calculamos, por provincia y mes, la
**mediana** de las estaciones (la mediana no se mueve por unas pocas estaciones con errores) habiendo
sacado antes las estaciones fuera de rango (ver 2.2).

Para comparar precios de distintos años los pasamos a **pesos constantes** (con el índice de inflación
mensual, base = último mes disponible) y a **dólares** (oficial, promedio mensual).
""")
code("""
s = pd.read_parquet(C.PROC / "serie_real.parquet")
print(f"{s.provincia.nunique()} provincias, productos: {sorted(s.producto.unique())}, "
      f"meses: {s.mes.min():%Y-%m} a {s.mes.max():%Y-%m}")
provs = ["Ciudad Autónoma de Buenos Aires", "Córdoba", "Mendoza", "Neuquén", "Salta"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for p in provs:
    g = s[(s.provincia == p) & (s.producto == "nafta_comun")]
    ax[0].plot(g.mes, g.precio_mediana, label=p); ax[1].plot(g.mes, g.precio_real); ax[2].plot(g.mes, g.precio_usd)
ax[0].set_title("Nafta súper: $ corrientes / litro"); ax[0].set_yscale("log")
ax[1].set_title(f"$ constantes (base {s.base_ipc.iloc[0]:%m/%Y})"); ax[2].set_title("US$ (oficial) / litro")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()
""")
md("""
En pesos corrientes la curva sube siempre (inflación) y no dice mucho. En pesos constantes y en dólares
se ven los movimientos de verdad: períodos de congelamiento (el precio real cae) y saltos (devaluaciones,
liberación de precios).
""")

md("""
## 2. Anomalías (paso "detección de anomalías en series de tiempo")

### 2.1 Saltos de precio en la serie mensual
Aplicamos tres métodos sobre el precio real de cada provincia y producto:

1. **z-score móvil**: compara la variación del mes contra la media y desvío de los 12 meses anteriores
   (sin mirar el futuro). Anomalía si |z| > 3.
2. **STL** (descomposición en tendencia + estacionalidad + residuo): anomalía si el residuo es muy grande
   (|z robusto| > 3,5).
3. **Isolation Forest**: un modelo de ML que mira varias variables a la vez (variación real y nominal,
   inflación, dólar, diferencia con el resto del país) y aísla los meses "raros" (3% más raro).

Un mes es "anómalo" con más confianza cuando lo marcan **2 o más métodos**.
""")
code("""
a = pd.read_parquet(C.PROC / "anomalias_series.parquet")
print("Meses-provincia marcados por cada método:")
print(a[["anom_z", "anom_stl", "anom_iforest"]].sum().rename({"anom_z": "z-score", "anom_stl": "STL",
      "anom_iforest": "Isolation Forest"}))
print("\\nCoincidencias (cantidad de métodos que marcan el mismo mes):")
print(a.n_metodos.value_counts().sort_index())
""")
code("""
p, prod = "Córdoba", "nafta_comun"
g = a[(a.provincia == p) & (a.producto == prod)]
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(g.mes, g.precio_real, color="0.3", lw=1.5, label="precio real")
for col, m, c in (("anom_z", "o", "tab:blue"), ("anom_stl", "s", "tab:orange"), ("anom_iforest", "^", "tab:red")):
    k = g[g[col]]
    ax.scatter(k.mes, k.precio_real, marker=m, s=70, facecolors="none", edgecolors=c, lw=2, label=col)
ax.set_title(f"{p} – {prod}: anomalías por método"); ax.legend(); plt.show()
""")
md("**Tabla para interpretar** (meses donde ≥ 2 métodos coinciden, resumido a nivel país):")
code("""
t = analisis.tabla_anomalias(a)
t.tail(25)
""")
code("""
cols = ["provincia", "producto", "mes", "var_nominal_%", "var_real_%", "inflacion_mm", "var_usd_%", "n_metodos", "interpretacion"]
a[a.n_metodos >= 2].sort_values("mes", ascending=False)[cols].head(20).round(1)
""")
md("""
**Lectura.** Los saltos que marcan varios métodos y en muchas provincias a la vez son eventos de país
(devaluaciones, aumentos de las petroleras, cambios de impuestos). Los que aparecen en una sola provincia
con una diferencia grande contra el resto del país suelen ser problemas de datos locales (pocas estaciones
informando ese mes) o cambios impositivos provinciales.

### 2.2 Estaciones con precios fuera de rango
Para cada provincia, producto y mes comparamos cada estación contra la mediana provincial con un
**z robusto** (mediana y MAD). Se marca si |z| > 5, o si el precio es menos de la mitad o más del doble de la
mediana. Esto separa errores de carga (por ejemplo un precio 100 veces mayor: falta la coma) de precios
caros/baratos de verdad. Estas estaciones se excluyen del cálculo de la serie.
""")
code("""
e = pd.read_parquet(C.PROC / "anomalias_estaciones.parquet")
print(e.motivo.value_counts())
cols = [c for c in ["fuente", "provincia_n", "producto_n", "mes", "bandera", "localidad", "precio", "ratio_mediana", "z_robusto", "motivo"] if c in e]
e.sort_values("ratio_mediana", ascending=False)[cols].head(15).round(2)
""")

md("""
### 2.3 ¿Qué se hizo con cada anomalía ANTES del modelo?

No todas las anomalías se tratan igual: los **errores de datos** se sacan; los **eventos reales**
(devaluaciones, aumentos) se dejan, porque son justamente lo que el modelo tiene que aprender.
""")
code("""
L = json.loads((C.REPORTES / "resumen_limpieza.json").read_text())
a2 = int((a.n_metodos >= 2).sum())
pd.DataFrame([
    ["Precio <= 0, vacío o > $50.000", "regla fija", L["precios_invalidos_descartados"], "registro descartado"],
    ["Fecha imposible (antes de 2016 o futura)", "regla fija", L["fechas_invalidas_descartadas"], "registro descartado"],
    ["Estación con precio fuera de rango vs. su provincia", "z robusto (mediana/MAD) > 5 y >10% de diferencia, o < 0,5x / > 2x la mediana",
     L["estacion_mes_fuera_de_rango_excluidos"], "excluida de la mediana provincial"],
    [f"Provincia-mes con menos de {L['min_estaciones']} estaciones", "conteo", L["provincia_mes_no_confiables_excluidos"],
     "excluida de la serie del modelo"],
    ["Salto de precio en la serie (2+ métodos)", "z-score móvil, STL, Isolation Forest", a2,
     "SE MANTIENE (evento real); se documenta en la tabla 2.1"],
], columns=["anomalía", "cómo se detecta", "cantidad", "tratamiento"])
""")
md("""
Ejemplos de fechas imposibles encontradas en el archivo oficial:
""")
code("""
print(L["ejemplos_fechas_invalidas"])
""")

md("""
## 3. Predicción del precio del litro a 1, 2 y 3 meses (paso "machine learning")

**Qué predecimos:** el cambio porcentual (en log) del precio nominal del litro en cada provincia y producto
de acá a `h` meses (h = 1, 2, 3).

**Variables (sólo información disponible en el momento de predecir):** variaciones pasadas del precio
(1, 2, 3, 6, 12 meses), variación del dólar, inflación reciente, diferencia contra el promedio del país,
cuán "atrasado" está el precio real contra su promedio de 12 meses, y el mes del año.

**Modelos comparados:**
- **Ingenuo (último precio)**: el precio no cambia. Es la vara mínima que pidió la consigna.
- **Ingenuo + inflación**: el precio sube como la inflación de los últimos 3 meses.
- **Ridge** (regresión lineal regularizada) y **Gradient Boosting** (árboles).

**Validación temporal ("walk-forward")**: para cada mes de prueba entrenamos SOLO con datos anteriores
(cuyo resultado ya se conocía en ese momento) y predecimos hacia adelante. Nunca mezclamos al azar,
porque eso dejaría al modelo "ver el futuro".
""")
code("""
r = pd.read_csv(C.REPORTES / "validacion_modelos.csv")
r
""")
code("""
fig, ax = plt.subplots(figsize=(9, 4))
for m, g in r.groupby("modelo"):
    ax.plot(g.h, g.MAPE_pct, marker="o", label=m)
ax.set_xlabel("meses hacia adelante (h)"); ax.set_ylabel("error porcentual medio (%)"); ax.set_xticks([1, 2, 3])
ax.set_title("Error de predicción del precio del litro (validación temporal)"); ax.legend(); plt.show()
for h in (1, 2, 3):
    rh = r[r.h == h].set_index("modelo"); gana = rh.MAPE_pct.idxmin()
    if gana.startswith("ingenuo (último"):
        print(f"h={h}: NINGÚN modelo le gana al ingenuo -> usamos el último precio")
    else:
        print(f"h={h}: gana '{gana}' con {rh.loc[gana, 'MAPE_pct']:.2f}% de error "
              f"({rh.loc[gana, 'mejora_vs_ingenuo_%']:.0f}% menos que el ingenuo; "
              f"vs 'ingenuo + inflación': {rh.loc['ingenuo + inflación', 'MAPE_pct']:.2f}%)")
""")
md("""
**Cómo leer las métricas**
- **MAPE** (error % medio) y **MAE** (error medio en $ por litro): el error típico.
- **RMSE** (en $): como el MAE pero castiga más los errores grandes (por ejemplo, no anticipar una devaluación).
- **R² del cambio**: qué parte de la variación del precio explica el modelo. 0 = no mejor que predecir
  el cambio promedio; negativo = peor.
- **R² del nivel**: R² sobre el precio en pesos. Da alto para *cualquier* modelo (incluso el ingenuo) porque
  los precios de 2017 y de 2026 son muy distintos: es una métrica **engañosa** acá, la mostramos para explicarlo.
""")
code("""
cols = ["h", "modelo", "MAPE_pct", "MAE_pesos", "RMSE_pesos", "R2_cambio", "R2_nivel", "acierto_direccion_real"]
display(r[cols])
for h in (1, 2, 3):
    rh = r[r.h == h].set_index("modelo"); ml = rh.drop(["ingenuo (último precio)", "ingenuo + inflación"]).MAPE_pct.idxmin()
    ref = rh.loc["ingenuo + inflación"]
    comp = [f"{m}: {'mejor' if rh.loc[ml, m] < ref[m] else 'PEOR'} ({rh.loc[ml, m]:.2f} vs {ref[m]:.2f})"
            for m in ("MAPE_pct", "MAE_pesos", "RMSE_pesos")]
    print(f"h={h}: '{ml}' vs 'ingenuo + inflación' -> " + "; ".join(comp) + f"; R² cambio {rh.loc[ml, 'R2_cambio']:.2f}")
""")
md("""
**Importante para leer bien la tabla:** con inflación alta, el ingenuo "último precio" es fácil de vencer,
porque cualquier modelo que sume la inflación mejora. La comparación honesta es contra
**"ingenuo + inflación"**: si el ML no le gana a ese, el ML no aporta y lo decimos.

`acierto_direccion_real` mide (como el ejercicio de bitcoin, que predice si sube o baja) si el modelo acierta
cuando el precio sube **más o menos que la inflación**.
""")
code("""
pron = pd.read_parquet(C.PROC / "pronosticos.parquet")
t = pron[pron.producto == "nafta_comun"].pivot_table(index="provincia", columns="h", values="precio_pred")
t.columns = [f"{h} mes(es)" for h in t.columns]
t.insert(0, "actual", pron[(pron.producto == "nafta_comun") & (pron.h == 1)].set_index("provincia").precio_actual)
print(f"Pronóstico nafta súper ($/litro), desde {pron.mes_origen.iloc[0]:%m/%Y}, modelo: {pron.modelo.unique()}")
t.round(0)
""")

md("""
## 4. Consumo de los vehículos y el paso de 100 a 130 km/h

### 4.1 Base OFICIAL de consumos (Ministerio de Ambiente)
El dataset de la etiqueta vehicular de datos.gob.ar fue dado de baja (404). Los datos vigentes están en el
Centro de Información Ambiental (CIAM, dataset 566): un Excel de **Licencias de Configuración Ambiental** con el
consumo urbano / extraurbano / mixto certificado de cada configuración homologada. `src/descargar.py` lee la
página del CIAM, encuentra el enlace al Excel más nuevo y lo baja; `src/etiqueta.py` lo limpia y lo valida.

**Limitación de la fuente: no informa el año del modelo.** No lo inventamos: la app elige el vehículo por
marca → modelo → versión (denominación, caja, tracción y potencia del certificado), sin año.

**Limpieza de marcas y modelos** (`src/etiqueta.py`): homologaciones compartidas como "PEUGEOT / CITROËN" se
separan en una fila por marca (PARTNER / BERLINGO); "212" (marca todoterreno del grupo BAIC) pasa a BAIC
"212 T01"; "MERCEDES-AMG" a MERCEDES-BENZ "AMG ..."; "DFAC DONGFENG" a DONGFENG; "CITROEN" a "CITROËN";
"NUEVO DUSTER" a "DUSTER"; "CLASE GLC" a "GLC".
""")
code("""
of = pd.read_csv(C.PROC / "consumos_oficiales.csv")
print(f"Configuraciones: {len(of):,} | marcas: {of.marca.nunique()} | usables: {int(of.usable.sum()):,} | "
      f"híbridos: {int(of.hibrido.sum())}")
print("\\nProblemas encontrados (filas):")
print(of.problemas.fillna("").replace("", "ninguno").value_counts().head(10).to_string())
bo = consumos.cargar_base_oficial()
print(f"\\nEn la app (nafta/diésel, usables, sin duplicados): {len(bo):,} configuraciones de {bo.marca.nunique()} marcas")
""")
md("""
### 4.2 Base propia y comparación con la oficial
Antes de encontrar la base oficial armamos una base con **27 versiones**
de los autos más vendidos con su consumo urbano / extraurbano / mixto y **la fuente de cada fila**
(etiquetas IRAM/AITA 10274 en PDF oficiales, fichas oficiales citadas por terceros, catálogos). Las filas
con dudas están marcadas como `aproximado` con la explicación. No se inventó ningún valor: cuando faltaba
el mixto se calculó con la ponderación del ciclo NEDC y se marcó. La app usa la base oficial; la base propia
queda para **validar** (abajo) y para vincular las pruebas reales a 100 km/h con las versiones oficiales.

Comparamos cada fila de la base propia con la configuración oficial más parecida (misma marca, familia y
combustible; la de consumo mixto más cercano). Ojo: elegir la más cercana hace que la diferencia sea la
**mínima posible**; si aun así es grande, la fila propia es dudosa.
""")
code("""
b = consumos.cargar_base()
filas = []
for _, x in b.iterrows():
    o = bo[(bo.marca == x.marca.upper()) & (bo.modelo == x.modelo.upper()) & (bo.combustible == x.combustible)]
    if o.empty:
        filas.append([x.marca, x.modelo, x.version, x.ciclo, x.consumo_mixto_l100, None, None, "sin equivalente oficial"]); continue
    o = o.loc[(o.consumo_mixto_l100 - x.consumo_mixto_l100).abs().idxmin()]
    filas.append([x.marca, x.modelo, x.version, x.ciclo, x.consumo_mixto_l100, o.consumo_mixto_l100,
                  round((x.consumo_mixto_l100 / o.consumo_mixto_l100 - 1) * 100, 1), o.version[:60]])
comp = pd.DataFrame(filas, columns=["marca", "modelo", "version propia", "ciclo propio", "mixto propio",
                                    "mixto oficial", "dif %", "version oficial más parecida"])
comp
""")
code("""
print(b.tipo_fuente.value_counts(), "\\nAproximados:", int(b.aproximado.sum()), "de", len(b))
b[["marca", "modelo", "version", "anio_desde", "anio_hasta", "combustible", "consumo_urbano_l100",
   "consumo_extraurbano_l100", "consumo_mixto_l100", "ciclo", "aproximado"]]
""")
md("""
### 4.3 Modelo físico

A velocidad constante el motor vence la **rodadura** (casi constante) y el **aire** (crece con el cuadrado de
la velocidad). Por eso:

$$ \\text{consumo}(v) = c_{100} \\cdot \\left[(1-f) + f\\left(\\frac{v + v_{viento}}{100}\\right)^2\\right] $$

- $c_{100}$: consumo a 100 km/h. Si hay una medición real de esa versión la usamos; si no,
  $c_{100} = k \\cdot$ extraurbano de la etiqueta (el ciclo extraurbano tiene velocidad media ~62 km/h y máxima 120,
  por eso hace falta corregirlo).
- $f$: qué parte del consumo a 100 km/h se va en vencer el aire.

Calibramos $k$ y $f$ con **pruebas reales a velocidad constante** (Auto Test, datos de fábrica):
""")
code("""
p = consumos.calibrar()
pr = consumos.cargar_pruebas()
print(f"k = {p.k_med:.2f} (rango intercuartil {p.k_p25:.2f}-{p.k_p75:.2f}, {p.n_k} autos con etiqueta oficial y prueba a 100 km/h)")
print(f"f = {p.f_med:.2f} (rango {p.f_p25:.2f}-{p.f_p75:.2f}, {p.n_f} autos con prueba a 100 y 130 km/h)")
print(f"=> ir a 130 km/h gasta ~{((1-p.f_med)+p.f_med*1.69-1)*100:.0f}% más por km que ir a 100 km/h "
      f"(rango {((1-p.f_p25)+p.f_p25*1.69-1)*100:.0f}%-{((1-p.f_p75)+p.f_p75*1.69-1)*100:.0f}%)")
v = np.linspace(60, 150, 50)
fig, ax = plt.subplots(figsize=(8, 4))
for esc, ls in (("bajo", ":"), ("med", "-"), ("alto", ":")):
    ax.plot(v, [consumos.consumo_velocidad(5.0, x, p, escenario=esc) for x in v], ls, color="tab:blue")
r130 = (pr.consumo_130_l100 / pr.consumo_100_l100).dropna() * consumos.consumo_velocidad(5.0, 100, p)
ax.scatter(np.full(len(r130), 130), r130, color="tab:red", label="pruebas reales (relación 130/100)")
ax.set_xlabel("km/h"); ax.set_ylabel("l/100 km"); ax.set_title("Auto con extraurbano 5,0 l/100 km"); ax.legend(); plt.show()
""")
md("""
### 4.4 Validación "dejando uno afuera"
Para cada auto con prueba real, recalibramos **sin** ese auto y predecimos su consumo:
""")
code("""
val = consumos.validar_modelo()
print(f"Error mediano a 130 km/h: {val['error_130_%'].abs().median():.1f}%  |  "
      f"a 100 km/h (desde la etiqueta): {val['error_100_%'].abs().median():.1f}%")
val
""")
md("""
**Lectura.** Desde la etiqueta oficial la estimación a 100 km/h tiene errores chicos; las filas que vienen de
catálogos comerciales (Onix, Kwid) o del ciclo brasileño (Sandero RS) dan errores grandes: por eso las
marcamos como aproximadas y no las usamos para calibrar. El paso de 100 a 130 km/h (la parte física) se
predice con un error típico de pocos puntos porcentuales.

### 4.5 Ajustes por clima (supuestos)
""")
code("""
from src import viaje
for k, t in viaje.AJUSTES_CLIMA.items():
    print(f"- {t}\\n")
""")

md("""
## 5. Ejemplo de viaje (necesita internet: OSRM, Georef y Open-Meteo)
""")
code("""
from datetime import date, timedelta
loc = viaje.localidades()
def lugar(nombre, prov):
    r = loc[(loc.nombre == nombre) & (loc.provincia == prov)].iloc[0]; return (r.lat, r.lon)
fila, _ = consumos.elegir(b, "Fiat", "Cronos", "Like / Drive 1.3 MT5", 2025, "nafta")
c100, fuente = consumos.c100_medido(fila.id)
try:
    R = viaje.calcular(lugar("Córdoba", "Córdoba"), lugar("Mendoza", "Mendoza"), date.today() + timedelta(days=10),
                       "nafta_comun", fila.consumo_extraurbano_l100, p, c100)
    print(f"Distancia: {R['km']:.0f} km | provincias: {R['km_provincia']}")
    for v_ in (100, 130):
        x = R["velocidades"][v_]
        print(f"{v_} km/h: {x['med']['litros']:.1f} l, $ {x['med']['costo']:,.0f} "
              f"(rango $ {x['bajo']['costo']:,.0f} - $ {x['alto']['costo']:,.0f}), {x['horas']:.1f} h de manejo")
    print(f"Ir a 130 cuesta $ {R['diferencia_pesos']:,.0f} más ({R['diferencia_pct']:.0f}%)")
    display(pd.DataFrame(R["precios"]).T)
    display(R["estaciones"])
except Exception as ex:
    print("No se pudo calcular (¿sin internet?):", ex)
""")

md("""
## 6. Conclusiones

Los números de esta sección se calculan con los resultados de arriba (no están escritos a mano).
""")
code("""
L = json.loads((C.REPORTES / "resumen_limpieza.json").read_text())
r = pd.read_csv(C.REPORTES / "validacion_modelos.csv").set_index(["h", "modelo"])
ml = "gradient boosting"
p = consumos.calibrar()
val = consumos.validar_modelo()
print(f"1. DATOS: {L['filas_leidas']:,} registros de precios por estación; {L['precios_invalidos_descartados']} precios "
      f"inválidos y {L['fechas_invalidas_descartadas']} fechas imposibles descartados; "
      f"{L['estacion_mes_fuera_de_rango_excluidos']:,} estación-mes fuera de rango excluidos; "
      f"serie final: {L['provincia_mes_total'] - L['provincia_mes_no_confiables_excluidos']:,} provincia-producto-mes confiables.")
print(f"2. ANOMALÍAS: {int((a.n_metodos >= 2).sum())} provincia-mes con saltos detectados por 2+ métodos; "
      f"los más fuertes coinciden con devaluaciones y congelamientos (tabla 2.1).")
for h in (1, 2, 3):
    g, i, ii = r.loc[(h, ml)], r.loc[(h, "ingenuo + inflación")], r.loc[(h, "ingenuo (último precio)")]
    print(f"3. PREDICCIÓN h={h}: {ml} MAPE {g.MAPE_pct:.1f}% vs ingenuo+inflación {i.MAPE_pct:.1f}% vs último precio "
          f"{ii.MAPE_pct:.1f}% | RMSE ${g.RMSE_pesos:.0f} vs ${i.RMSE_pesos:.0f} | R² del cambio {g.R2_cambio:.2f}")
print(f"4. VELOCIDAD: a 130 km/h se gasta ~{((1-p.f_med)+p.f_med*1.69-1)*100:.0f}% más por km que a 100 km/h "
      f"(rango {((1-p.f_p25)+p.f_p25*1.69-1)*100:.0f}-{((1-p.f_p75)+p.f_p75*1.69-1)*100:.0f}%); "
      f"error mediano de validación a 130 km/h: {val['error_130_%'].abs().median():.1f}%.")
""")
md("""
**Interpretación**

1. **Los datos reales tenían problemas que una demo no muestra**: fechas imposibles, precios absurdos, un
   archivo de "vigentes" lleno de precios viejos y un quiebre por la derogación de la obligación de informar.
   La mayor parte del trabajo fue detectarlos y decidir qué hacer con cada uno (tabla 2.3).
2. **Anomalías**: los tres métodos coinciden en los eventos grandes; usarlos juntos sirve para separar eventos
   de país (muchas provincias a la vez) de problemas locales de datos (una sola provincia).
3. **Predicción**: hay que leer la tabla de la sección 3 completa. Ganarle al "último precio" es fácil con
   inflación; la vara honesta es "ingenuo + inflación". El R² del cambio cerca de 0 dice que **ningún modelo
   anticipa los saltos bruscos** (devaluaciones, liberación de precios): el modelo sirve para el "día a día"
   del precio, no para predecir shocks. Por eso la app muestra siempre un **rango**.
4. **Velocidad**: ir a 130 km/h en lugar de 100 km/h cuesta alrededor de un tercio más de combustible, a cambio
   de ~23% menos de tiempo de manejo. Es el resultado más robusto del trabajo (física simple + pruebas reales).

**Limitaciones**: la fuente oficial de consumos no informa el año; las pruebas a velocidad constante son
pocas y de pocas fuentes; los ajustes por clima son supuestos simples; la ruta no considera pendientes ni
tránsito; desde jul-2025 hay muchas menos estaciones informando precios.
""")

nb = {"cells": celdas, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"celda-{i:02d}"
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("ok", len(celdas))
