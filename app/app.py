"""Calculadora de costo de combustible de un viaje (Streamlit).

Correr:  streamlit run app/app.py   (o con Docker: docker compose up app)
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import consumos, viaje  # noqa: E402

st.set_page_config(page_title="¿Cuánto me sale el viaje?", page_icon="⛽", layout="wide")
st.title("⛽ ¿Cuánto me sale el viaje en combustible?")
st.caption("Comparación manejando a ~100 km/h vs ~130 km/h, con precio estimado por provincia y clima del día.")

FALTAN = [p for p in (C.PROC / "estaciones_vigentes.parquet", C.GEO / "localidades.csv") if not p.exists()]
if FALTAN:
    st.error("Faltan datos. Corré primero la descarga y el procesamiento: `docker compose run --rm datos`.\n\n"
             + "\n".join(f"- falta `{p.relative_to(C.RAIZ) if C.RAIZ in p.parents else p}`" for p in FALTAN))
    st.stop()


@st.cache_data
def base_propia():
    return consumos.cargar_base()


@st.cache_data
def base_oficial():
    return consumos.cargar_base_oficial()


@st.cache_data
def params():
    return consumos.calibrar()


@st.cache_data
def ratio_urbano():
    """Relación típica consumo urbano / consumo a 100 km/h en la base oficial (para la opción manual)."""
    b = consumos.cargar_base_oficial()
    b = b if b is not None and len(b) else consumos.cargar_base()
    b = b[~b.aproximado]
    return float((b.consumo_urbano_l100 / (P_k() * b.consumo_extraurbano_l100)).median())


def P_k():
    return consumos.calibrar().k_med


def _leer(path):
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


@st.cache_data
def pronosticos():
    d = _leer(C.PROC / "pronosticos.parquet")
    if d is not None:
        d["mes_origen"], d["mes_objetivo"] = pd.to_datetime(d.mes_origen), pd.to_datetime(d.mes_objetivo)
    return d


@st.cache_data
def historico():
    d = _leer(C.PROC / "precios_mensuales.parquet")
    if d is not None:
        d["mes"] = pd.to_datetime(d.mes)
    return d


@st.cache_data
def validacion():
    return _leer(C.REPORTES / "validacion_modelos.csv")


MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def nombre_mes(m):
    m = pd.Timestamp(m)
    return f"{MESES[m.month - 1]} {m.year}"


@st.cache_data
def locs():
    return viaje.localidades()


def indice(opciones, preferidas):
    """Posición de la primera opción preferida que exista (para arrancar con valores lógicos)."""
    norm = [str(o).upper() for o in opciones]
    for p in preferidas:
        if str(p).upper() in norm:
            return norm.index(str(p).upper())
    return 0


# Capitales: para que la localidad inicial sea la ciudad principal y no la primera en orden alfabético
CAPITALES = {"Buenos Aires": "La Plata", "Catamarca": "San Fernando del Valle de Catamarca",
             "Chaco": "Resistencia", "Chubut": "Rawson", "Córdoba": "Córdoba", "Corrientes": "Corrientes",
             "Entre Ríos": "Paraná", "Formosa": "Formosa", "Jujuy": "San Salvador de Jujuy", "La Pampa": "Santa Rosa",
             "La Rioja": "La Rioja", "Mendoza": "Mendoza", "Misiones": "Posadas", "Neuquén": "Neuquén",
             "Río Negro": "Viedma", "Salta": "Salta", "San Juan": "San Juan", "San Luis": "San Luis",
             "Santa Cruz": "Río Gallegos", "Santa Fe": "Santa Fe", "Santiago del Estero": "Santiago del Estero",
             "Tierra del Fuego, Antártida e Islas del Atlántico Sur": "Ushuaia", "Tucumán": "San Miguel de Tucumán",
             "Ciudad Autónoma de Buenos Aires": "Ciudad de Buenos Aires"}


def fmt_num(x):
    return f"{x:,.0f}".replace(",", ".")


def fmt_pesos(x):
    return "—" if x is None or np.isnan(x) else f"$ {x:,.0f}".replace(",", ".")


B_PROPIA, B_OFICIAL, P, L = base_propia(), base_oficial(), params(), locs()
RATIO_URBANO = ratio_urbano()

# ================================================================ 1. vehículo (embudo)
st.header("1. Tu vehículo")
# Base: la OFICIAL del Ministerio de Ambiente (consumos certificados). Si todavía no se procesó, se usa la
# base propia como respaldo (27 versiones con fuente por fila).
B = B_OFICIAL if B_OFICIAL is not None and len(B_OFICIAL) else B_PROPIA
if B is B_PROPIA:
    st.warning("No encontramos la base oficial de consumos procesada: se usa la base propia (27 versiones). "
               "Corré `docker compose run --rm datos python -m src.etiqueta` para tener todas.")
manual = st.toggle("Opción avanzada: prefiero cargar mi consumo real", value=False)
fila, c100_usuario, urbano = None, None, None
if manual:
    c1, c2 = st.columns(2)
    c100_usuario = c1.number_input("Tu consumo en ruta a ~100 km/h (litros cada 100 km)", 3.0, 25.0, 7.0, 0.1)
    comb = c2.selectbox("Combustible", ["nafta", "diesel"])
    extraurbano = None
    urbano = c100_usuario * RATIO_URBANO
    st.info(f"Con tu consumo a 100 km/h, el paso a 130 km/h se estima con el modelo físico calibrado. "
            f"Para los tramos urbanos estimamos {urbano:.1f} l/100 km (tu consumo × {RATIO_URBANO:.2f}, la relación "
            "típica entre consumo urbano y en ruta de la base oficial).")
else:
    filtro_comb = st.radio("Combustible del vehículo", ["todos", "nafta", "diesel"], horizontal=True,
                           format_func=lambda x: {"todos": "Todos", "nafta": "Nafta",
                                                  "diesel": "Diésel (gasoil)"}[x],
                           help="Filtra marcas, modelos y versiones según el combustible que figura en el "
                                "certificado oficial. Los híbridos a nafta están dentro de «Nafta».")
    BF = B if filtro_comb == "todos" else B[B.combustible == filtro_comb]
    c = st.columns(3)
    marcas = consumos.opciones(BF, "marca")
    marca = c[0].selectbox("Marca", marcas, index=indice(marcas, ["FIAT", "Fiat"]))
    modelos = consumos.opciones(BF, "modelo", marca=marca)
    modelo = c[1].selectbox("Modelo", modelos, index=indice(modelos, ["CRONOS", "Cronos"]))
    version = c[2].selectbox("Versión", consumos.opciones(BF, "version", marca=marca, modelo=modelo),
                             help="Tal como figura en el certificado oficial: denominación · caja · tracción · potencia.")
    fila, aviso = consumos.elegir(BF, marca, modelo, version)
    if aviso:
        st.warning(aviso)
    comb = fila.combustible
    extraurbano = float(fila.consumo_extraurbano_l100)
    urbano = float(fila.consumo_urbano_l100)
    c100_med, fuente_med = consumos.c100_medido(fila.id, fila)
    if c100_med is not None:
        c100_usuario = c100_med
    txt = (f"**Combustible:** {'nafta' if comb == 'nafta' else 'diésel (gasoil)'} · **Consumo certificado:** "
           f"urbano {fila.consumo_urbano_l100:.1f} · extraurbano {fila.consumo_extraurbano_l100:.1f} · "
           f"mixto {fila.consumo_mixto_l100:.1f} l/100 km — fuente: [{fila.tipo_fuente}]({fila.fuente_url})")
    if c100_med is not None:
        txt += f"\n\n✅ Para esta versión usamos una **{fuente_med}** ({c100_med} l/100 km)."
    else:
        txt += (f"\n\nConsumo estimado a 100 km/h: **{P.k_med * extraurbano:.1f} l/100 km** "
                f"(extraurbano × {P.k_med:.2f}, ver supuestos).")
    if fila.aproximado and str(fila.nota).strip():
        txt += f"\n\n⚠️ {fila.nota}"
    st.markdown(txt)

# ================================================================ 2. viaje
st.header("2. Tu viaje")
provs = sorted(L.provincia.dropna().unique())
c = st.columns(3)
po = c[0].selectbox("Provincia de origen", provs, index=provs.index("Ciudad Autónoma de Buenos Aires")
                    if "Ciudad Autónoma de Buenos Aires" in provs else 0)
locs_o = sorted(L[L.provincia == po].nombre.unique())
lo = c[0].selectbox("Localidad de origen", locs_o, index=indice(locs_o, [CAPITALES.get(po, ""), po]))
pd_ = c[1].selectbox("Provincia de destino", provs, index=provs.index("Córdoba") if "Córdoba" in provs else 0)
locs_d = sorted(L[L.provincia == pd_].nombre.unique())
ld = c[1].selectbox("Localidad de destino", locs_d, index=indice(locs_d, [CAPITALES.get(pd_, ""), pd_]))
fecha = c[2].date_input("Fecha del viaje", date.today() + timedelta(days=7), min_value=date.today(),
                        max_value=date.today() + timedelta(days=365))
ETIQ_CALIDAD = {("nafta", "comun"): "Súper (92–95 RON)", ("nafta", "premium"): "Premium (+95 RON)",
                ("diesel", "comun"): "Gasoil grado 2 (común)", ("diesel", "premium"): "Gasoil grado 3 (premium)"}
calidad = c[2].radio("Nafta que cargás" if comb == "nafta" else "Gasoil que cargás", ["comun", "premium"],
                     horizontal=True, format_func=lambda x: ETIQ_CALIDAD[(comb, x)],
                     help="El tipo de combustible (nafta o diésel) lo define el vehículo elegido arriba.")
producto = f"{comb}_{calidad}"

FP = viaje.fecha_precios()
txt_fp = (f"Precios de surtidor: el dato más nuevo es del **{FP['ultima']:%d/%m/%Y}** "
          f"({fmt_num(FP['estaciones_recientes'])} estaciones con precio de los últimos 120 días).")
if FP["dias"] > 10:
    st.warning(txt_fp + f" Tienen {FP['dias']} días: para actualizarlos corré "
               "`docker compose run --rm datos python -m src.descargar precios` y después "
               "`docker compose run --rm datos python -m src.procesar`.")
else:
    st.caption(txt_fp)
if (fecha - date.today()).days > 92:
    st.warning("Para fechas a más de 3 meses el precio se extrapola con un supuesto (2% mensual): tomalo como "
               "una referencia gruesa.")

if not st.button("Calcular", type="primary"):
    st.stop()
if po == pd_ and lo == ld:
    st.error("El origen y el destino son el mismo lugar: elegí otro destino.")
    st.stop()

o = L[(L.provincia == po) & (L.nombre == lo)].iloc[0]
d = L[(L.provincia == pd_) & (L.nombre == ld)].iloc[0]
with st.spinner("Calculando ruta, provincias, clima y precios..."):
    try:
        R = viaje.calcular((o.lat, o.lon), (d.lat, d.lon), fecha, producto, extraurbano, P, c100_usuario,
                           urbano=urbano)
    except Exception as e:
        st.error(f"No se pudo calcular el viaje: {e}")
        st.stop()

# ================================================================ 3. resultados
st.header("3. Resultado")
v100, v130 = R["velocidades"][100], R["velocidades"][130]
c = st.columns(4)
c[0].metric("Distancia", f"{fmt_num(R['km'])} km")
c[1].metric("A ~100 km/h", fmt_pesos(v100["med"]["costo"]),
            help=f"Rango: {fmt_pesos(v100['bajo']['costo'])} a {fmt_pesos(v100['alto']['costo'])}")
c[2].metric("A ~130 km/h", fmt_pesos(v130["med"]["costo"]), f"+{R['diferencia_pct']:.0f}%", delta_color="inverse",
            help=f"Rango: {fmt_pesos(v130['bajo']['costo'])} a {fmt_pesos(v130['alto']['costo'])}")
ahorro_h = v100["horas"] - v130["horas"]
c[3].metric("Ir a 130 te cuesta", fmt_pesos(R["diferencia_pesos"]),
            f"y ahorrás ~{ahorro_h * 60:.0f} min de manejo", delta_color="off")
st.caption(f"Litros: {v100['med']['litros']:.1f} a 100 km/h vs {v130['med']['litros']:.1f} a 130 km/h. "
           f"De los {fmt_num(R['km'])} km, {fmt_num(R['km_urbano'])} km son urbanos o lentos (se calculan con el "
           "consumo urbano y la velocidad estimada por OSRM). Tiempos sin paradas ni tránsito.")
st.info("⚖️ **Velocidad legal (Ley 24.449):** 130 km/h sólo está permitido en **autopistas**; en semiautopistas "
        "el máximo es 120 km/h y en rutas comunes 110 km/h (autos). La comparación muestra cuánto cuesta ir más "
        "rápido donde se puede, no una recomendación.")

filas = []
for prov, km in R["km_provincia"].items():
    pr, cl = R["precios"][prov], R["clima"][prov]
    filas.append({"Provincia": prov, "km": round(km), "Precio litro": fmt_pesos(pr["precio"]),
                  "Rango precio": f"{fmt_pesos(pr['bajo'])} – {fmt_pesos(pr['alto'])}",
                  "Litros a 100": round(v100["med"]["litros_prov"][prov], 1),
                  "Litros a 130": round(v130["med"]["litros_prov"][prov], 1),
                  "Temp. máx.": f"{cl['temp_max']:.0f} °C" if not np.isnan(cl["temp_max"]) else "—",
                  "Lluvia": f"{cl.get('prob_lluvia', 0) * 100:.0f}%",
                  "Viento": f"{cl['viento_kmh']:.0f} km/h", "Cómo estimamos el precio": pr["metodo"]})
st.subheader("Por provincia")
st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- pronóstico del precio (modelo de ML)
st.subheader("Precio del litro en los próximos meses (pronóstico del modelo)")
PRON, HIST, VAL = pronosticos(), historico(), validacion()
if PRON is None or PRON.empty:
    st.info("Todavía no hay pronósticos: corré `docker compose run --rm datos python -m src.pipeline --sin-descarga`.")
else:
    pp = PRON[PRON.producto == producto]
    origen = pd.Timestamp(pp.mes_origen.iloc[0]) if len(pp) else None
    meses_obj = sorted(pp.mes_objetivo.unique())
    filas_p, sin_pron = [], []
    for prov in R["km_provincia"]:
        g = pp[pp.provincia == prov].set_index("h")
        if g.empty:
            sin_pron.append(prov)
            continue
        f = {"Provincia": prov, f"{nombre_mes(origen)} (último mes con datos)": fmt_pesos(g.precio_actual.iloc[0])}
        for h, r in g.sort_index().iterrows():
            cambio = (r.precio_pred / r.precio_actual - 1) * 100
            f[f"{nombre_mes(r.mes_objetivo)} (+{h} mes{'es' if h > 1 else ''})"] = (
                f"{fmt_pesos(r.precio_pred)} ({f'{cambio:+.1f}'.replace('.', ',')}%) · rango {fmt_pesos(r.precio_p10)}–{fmt_pesos(r.precio_p90)}")
        filas_p.append(f)
    if filas_p:
        st.dataframe(pd.DataFrame(filas_p), hide_index=True, use_container_width=True)
    if sin_pron:
        st.caption(f"Sin pronóstico propio para {', '.join(sin_pron)}: no tienen una serie mensual con suficientes "
                   "estaciones (mínimo 3 por mes). Para el costo del viaje se usa el cambio mediano del resto del país.")
    if VAL is not None and not VAL.empty:
        def mape(h, m):
            x = VAL[(VAL.h == h) & (VAL.modelo == m)].MAPE_pct
            return f"{x.iloc[0]:.1f}".replace(".", ",") if len(x) else "—"
        modelo = pp.modelo.iloc[0] if len(pp) else "gradient boosting"
        st.caption(
            "**Modelo:** Gradient Boosting (HistGradientBoostingRegressor de scikit-learn), entrenado con la serie mensual de precios por "
            "provincia y producto, la inflación y el dólar. Predice cuánto cambia el precio en 1, 2 y 3 meses. "
            f"**Error medio en la validación** (simulando el pasado sin mirar el futuro): {mape(1, modelo)}% a 1 mes, "
            f"{mape(2, modelo)}% a 2 y {mape(3, modelo)}% a 3; el modelo simple «último precio» erra "
            f"{mape(1, 'ingenuo (último precio)')}%, {mape(2, 'ingenuo (último precio)')}% y "
            f"{mape(3, 'ingenuo (último precio)')}%. El **rango** es el que contuvo al 80% de los errores pasados. "
            "No anticipa saltos bruscos (por ejemplo, una devaluación). Si la fecha del viaje cae en un mes futuro, "
            "el costo de arriba ya aplica este cambio al precio vigente de hoy. "
            f"El último mes completo con datos es {nombre_mes(origen)}: por eso «+1 mes» es el mes en curso.")
    if HIST is not None and filas_p:
        import altair as alt
        provs_g = [f["Provincia"] for f in filas_p]
        tabs = st.tabs(provs_g)
        for tab, prov in zip(tabs, provs_g):
            h_ = HIST[(HIST.provincia == prov) & (HIST.producto == producto) & HIST.confiable]
            h_ = h_[h_.mes > origen - pd.DateOffset(months=24)]
            g = pp[pp.provincia == prov].sort_values("h")
            obs = pd.DataFrame({"mes": h_.mes, "precio": h_.precio_mediana, "serie": "observado (mediana)"})
            pr_ = pd.DataFrame({"mes": [origen] + list(g.mes_objetivo),
                                "precio": [g.precio_actual.iloc[0]] + list(g.precio_pred), "serie": "pronóstico"})
            banda = pd.DataFrame({"mes": [origen] + list(g.mes_objetivo),
                                  "p10": [g.precio_actual.iloc[0]] + list(g.precio_p10),
                                  "p90": [g.precio_actual.iloc[0]] + list(g.precio_p90)})
            base = alt.Chart(banda).mark_area(opacity=0.2, color="#d62728").encode(
                x=alt.X("mes:T", title=None), y=alt.Y("p10:Q", title="$ por litro", scale=alt.Scale(zero=False)),
                y2="p90:Q")
            lineas = alt.Chart(pd.concat([obs, pr_])).mark_line(point=True).encode(
                x="mes:T", y=alt.Y("precio:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("serie:N", title=None, scale=alt.Scale(
                    domain=["observado (mediana)", "pronóstico"], range=["#1f77b4", "#d62728"])),
                strokeDash=alt.condition(alt.datum.serie == "pronóstico", alt.value([5, 4]), alt.value([1, 0])),
                tooltip=[alt.Tooltip("mes:T", format="%m/%Y"), alt.Tooltip("precio:Q", format=",.0f"), "serie:N"])
            with tab:
                st.altair_chart(base + lineas, use_container_width=True)
                st.caption(f"{prov}: precio mediano mensual de los últimos 24 meses y pronóstico con su rango (zona "
                           "sombreada). Los meses con pocas estaciones informando no se muestran.")

st.subheader("Dónde cargar más barato cerca de la ruta")
E = R["estaciones"]
if E is None or E.empty:
    st.info("No encontramos estaciones con precio vigente a menos de 3 km de la ruta.")
else:
    tabla = pd.DataFrame({
        "Bandera": E.get("bandera"), "Dirección": E.get("direccion"), "Localidad": E.get("localidad"),
        "Provincia": E.get("provincia_n"), "Precio litro": E.precio.map(fmt_pesos),
        "Precio informado el": pd.to_datetime(E.fecha).dt.strftime("%d/%m/%Y"),
        "Km del viaje": E.km_del_viaje.round(0).astype(int), "A cuántos km de la ruta": E.dist_ruta_km.round(1)})
    st.dataframe(tabla, hide_index=True, use_container_width=True)

try:
    import pydeck as pdk
    capa_ruta = pdk.Layer("PathLayer", data=[{"path": R["puntos"][["lon", "lat"]].values.tolist()}],
                          get_path="path", get_width=4, width_min_pixels=3, get_color=[30, 100, 200])
    capas = [capa_ruta]
    if E is not None and not E.empty:
        capas.append(pdk.Layer("ScatterplotLayer", data=E[["lon", "lat"]].assign(
                                   precio=E.precio.map(fmt_pesos), bandera=E.get("bandera", "")),
                               get_position="[lon, lat]", get_radius=1500, get_fill_color=[0, 160, 80],
                               pickable=True))
    centro = R["puntos"][["lat", "lon"]].mean()
    st.pydeck_chart(pdk.Deck(layers=capas, initial_view_state=pdk.ViewState(
        latitude=centro.lat, longitude=centro.lon, zoom=5), tooltip={"text": "{bandera}: {precio}"}))
except Exception as e:  # el mapa es un extra
    st.caption(f"(No se pudo dibujar el mapa: {e})")

with st.expander("Supuestos y fuentes (leé esto antes de confiar en el número)"):
    st.markdown(f"""
**Velocidad (100 vs 130 km/h).** consumo(v) = c100 × [(1 − f) + f × ((v + viento de frente)/100)²].
La resistencia del aire crece con el cuadrado de la velocidad; la de los neumáticos casi no cambia.
- c100 = {P.k_med:.2f} × consumo extraurbano de la etiqueta (mediana de {P.n_k} pruebas reales a 100 km/h; rango {P.k_p25:.2f}–{P.k_p75:.2f}).
- f = {P.f_med:.2f} (parte aerodinámica a 100 km/h, de {P.n_f} pruebas a 100 y 130 km/h; rango {P.f_p25:.2f}–{P.f_p75:.2f}).
- Resultado típico: a 130 km/h se gasta ~{((1 - P.f_med) + P.f_med * 1.69 - 1) * 100:.0f}% más por km.

**Clima.** {viaje.AJUSTES_CLIMA['aire']}

{viaje.AJUSTES_CLIMA['lluvia']}

{viaje.AJUSTES_CLIMA['viento']}

**Tramos urbanos.** {viaje.AJUSTES_CLIMA['urbano']}

Fuente del clima por provincia: {', '.join(sorted({c['fuente'] for c in R['clima'].values()}))}.

**Provincias que atraviesa:** {R['fuente_provincias']}. **Ruta:** OSRM (servidor público, OpenStreetMap).

**Precios:** Secretaría de Energía, Res. 314/2016 (CC-BY 4.0). Mediana de estaciones de la provincia con
precio de los últimos 120 días (sin outliers); para fechas futuras se aplica el cambio pronosticado por el
modelo (ver notebook). Desde la Res. 717/2025 muchas estaciones dejaron de informar: en provincias chicas el
precio puede salir de 1-2 estaciones (se avisa en la tabla). El costo asume que cargás en cada provincia lo que
consumís en ella.

**Consumos:** certificados oficiales (Licencias de Configuración Ambiental, Ministerio de Ambiente). La fuente
no informa el año del modelo. No se consideran pendientes ni tránsito.

**Rango:** combina el rango de los parámetros del modelo de velocidad con el rango de precios.
""")
