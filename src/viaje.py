"""Motor de la calculadora: ruta, provincias que atraviesa, clima, precio estimado y costo.

Todas las llamadas a APIs se guardan en data/cache para no repetir pedidos (y respetar
los límites de uso: OSRM público pide no más de 1 pedido por segundo y uso liviano).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from . import config as C
from .consumos import ParamsVelocidad, consumo_velocidad

SES = requests.Session()
SES.headers.update({"User-Agent": C.USER_AGENT})
R_TIERRA = 6371.0


# ----------------------------------------------------------------- utilidades
def _cache(nombre: str, clave: dict, funcion):
    h = hashlib.md5(json.dumps(clave, sort_keys=True).encode()).hexdigest()[:16]
    p = C.CACHE / f"{nombre}_{h}.json"
    if p.exists():
        return json.loads(p.read_text())
    res = funcion()
    p.write_text(json.dumps(res))
    return res


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_TIERRA * np.arcsin(np.sqrt(a))


def rumbo(lat1, lon1, lat2, lon2):
    """Dirección de marcha en grados (0 = norte, 90 = este)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ----------------------------------------------------------------- lugares y ruta
def localidades() -> pd.DataFrame:
    """Localidades de Georef, sin duplicados y con nombre único dentro de cada provincia.

    Georef trae 320 nombres repetidos en la misma provincia. Si están a menos de 5 km son el mismo lugar
    (se deja uno); si están más lejos son lugares distintos (p.ej. dos 'Canning') y se les agrega el
    departamento/partido al nombre para que el usuario elija el correcto.
    """
    L = pd.read_csv(C.GEO / "localidades.csv").dropna(subset=["lat", "lon", "provincia", "nombre"])
    filas = []
    for (prov, nom), g in L.groupby(["provincia", "nombre"], sort=False):
        elegidas = []
        for _, r in g.iterrows():
            if all(haversine(r.lat, r.lon, e.lat, e.lon) > 5 for e in elegidas):
                elegidas.append(r)
        for r in elegidas:
            r = r.copy()
            if len(elegidas) > 1:
                dep = r.get("departamento")
                r["nombre"] = f"{nom} ({dep})" if pd.notna(dep) and str(dep).strip() else f"{nom} ({r.lat:.2f}, {r.lon:.2f})"
            filas.append(r)
    out = pd.DataFrame(filas)
    # si dos lugares distintos tienen además el mismo departamento, se distinguen por coordenadas
    rep = out.duplicated(["provincia", "nombre"], keep=False)
    out.loc[rep, "nombre"] = out.loc[rep].apply(lambda r: f"{r['nombre']} ({r.lat:.2f}, {r.lon:.2f})", axis=1)
    return out.reset_index(drop=True)


def ruta_osrm(o: tuple[float, float], d: tuple[float, float]) -> dict:
    """o, d = (lat, lon). Devuelve distancia (km), duración (h) y puntos [(lat, lon), ...]."""
    def pedir():
        url = f"{C.OSRM}/{o[1]:.5f},{o[0]:.5f};{d[1]:.5f},{d[0]:.5f}"
        r = SES.get(url, params={"overview": "full", "geometries": "geojson", "annotations": "speed"},
                    timeout=60)
        r.raise_for_status()
        j = r.json()
        if j.get("code") != "Ok":
            raise RuntimeError(f"OSRM: {j.get('code')} {j.get('message', '')}")
        rt = j["routes"][0]
        time.sleep(1)  # política de uso del servidor público
        # velocidad estimada por OSRM para cada tramo de la geometría (m/s -> km/h); sirve para saber
        # qué partes del recorrido son urbanas o lentas (calles, avenidas, accesos)
        vel = []
        for leg in rt.get("legs", []):
            vel += [v * 3.6 for v in leg.get("annotation", {}).get("speed", [])]
        return {"km": rt["distance"] / 1000, "horas": rt["duration"] / 3600,
                "puntos": [[c[1], c[0]] for c in rt["geometry"]["coordinates"]], "vel_tramos": vel}
    return _cache("osrm", {"o": [round(x, 4) for x in o], "d": [round(x, 4) for x in d], "v": 2}, pedir)


def muestrear(puntos: list, cada_km: float = 1.0, vel_tramos: list | None = None) -> pd.DataFrame:
    """Puntos cada ~1 km a lo largo de la ruta, con km acumulado, rumbo y velocidad estimada por OSRM."""
    p = np.array(puntos)
    seg = haversine(p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1])
    acum = np.concatenate([[0], np.cumsum(seg)])
    objetivos = np.arange(0, acum[-1], cada_km)
    lat = np.interp(objetivos, acum, p[:, 0])
    lon = np.interp(objetivos, acum, p[:, 1])
    lat, lon = np.append(lat, p[-1, 0]), np.append(lon, p[-1, 1])
    km = np.append(objetivos, acum[-1])
    df = pd.DataFrame({"lat": lat, "lon": lon, "km": km})
    df["km_tramo"] = df.km.diff().shift(-1).fillna(0)
    df["rumbo"] = np.append(rumbo(lat[:-1], lon[:-1], lat[1:], lon[1:]), np.nan)
    df["rumbo"] = df.rumbo.ffill()
    df["vel_osrm"] = np.nan
    if vel_tramos is not None and len(vel_tramos) == len(seg) and len(seg):
        # velocidad del tramo de la geometría en el que cae cada punto (ponderada por distancia en ese km)
        idx = np.clip(np.searchsorted(acum, df.km.to_numpy() + df.km_tramo.to_numpy() / 2, side="right") - 1,
                      0, len(seg) - 1)
        df["vel_osrm"] = np.asarray(vel_tramos, dtype=float)[idx]
    return df


def provincias_de_puntos(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Provincia de cada punto con la API Georef (pedido en lote). Plan B: centroide más cercano."""
    def pedir(lote):
        body = {"ubicaciones": [{"lat": float(a), "lon": float(b), "campos": "provincia.nombre"}
                                for a, b in lote]}
        r = SES.post(f"{C.GEOREF}/ubicacion", json=body, timeout=60)
        r.raise_for_status()
        return [x["ubicacion"]["provincia"]["nombre"] for x in r.json()["resultados"]]
    pts = list(zip(df.lat.round(3), df.lon.round(3)))
    try:
        res = []
        for i in range(0, len(pts), 1000):
            lote = pts[i:i + 1000]
            res += _cache("georef_ubic", {"p": lote}, lambda: pedir(lote))
        return pd.Series(res, index=df.index).ffill().bfill(), "Georef (límites oficiales)"
    except Exception as e:  # sin internet o API caída
        prov = pd.read_csv(C.GEO / "provincias.csv")
        d = haversine(df.lat.to_numpy()[:, None], df.lon.to_numpy()[:, None],
                      prov.centroide_lat.to_numpy()[None, :], prov.centroide_lon.to_numpy()[None, :])
        return (pd.Series(prov.nombre.to_numpy()[d.argmin(1)], index=df.index),
                f"APROXIMADO por centroide más cercano (Georef no respondió: {e.__class__.__name__})")


# ----------------------------------------------------------------- clima
def clima_dia(lat: float, lon: float, fecha: date) -> dict:
    """Pronóstico si la fecha está dentro de 15 días; si no, climatología (promedio de 10 años)."""
    dias = (fecha - date.today()).days
    lat, lon = round(lat, 1), round(lon, 1)
    if 0 <= dias <= 15:
        def pedir():
            r = SES.get(C.OPEN_METEO_FORECAST, params={
                "latitude": lat, "longitude": lon, "timezone": "America/Argentina/Buenos_Aires",
                "hourly": "temperature_2m,precipitation,precipitation_probability,wind_speed_10m,wind_direction_10m",
                "start_date": fecha.isoformat(), "end_date": fecha.isoformat()}, timeout=60)
            r.raise_for_status()
            return r.json()["hourly"]
        h = pd.DataFrame(_cache("meteo_fc", {"lat": lat, "lon": lon, "f": fecha.isoformat(),
                                             "hoy": date.today().isoformat(), "v": 2}, pedir))
        h = h.iloc[8:21]  # horas de manejo diurno (8 a 20 h)
        ang = np.radians(h.wind_direction_10m)
        dir_media = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360
        return {"fuente": "pronóstico Open-Meteo", "temp_max": float(h.temperature_2m.max()),
                "temp_media": float(h.temperature_2m.mean()), "lluvia_mm": float(h.precipitation.sum()),
                # probabilidad de lluvia: promedio horario que da Open-Meteo (si no viene, ¿llueve > 1 mm?)
                "prob_lluvia": (float(h.precipitation_probability.mean()) / 100
                                if "precipitation_probability" in h and h.precipitation_probability.notna().any()
                                else float((h.precipitation.sum() > 1) * 1.0)),
                "viento_kmh": float(h.wind_speed_10m.mean()), "viento_dir": float(dir_media)}

    def pedir_hist():
        fin = date.today().replace(day=1) - pd.Timedelta(days=1)
        ini = fin.replace(year=fin.year - 10)
        r = SES.get(C.OPEN_METEO_ARCHIVE, params={
            "latitude": lat, "longitude": lon, "timezone": "America/Argentina/Buenos_Aires",
            "daily": "temperature_2m_max,temperature_2m_mean,precipitation_sum,wind_speed_10m_max,"
                     "wind_direction_10m_dominant",
            "start_date": ini.isoformat(), "end_date": fin.isoformat()}, timeout=120)
        r.raise_for_status()
        return r.json()["daily"]
    d = pd.DataFrame(_cache("meteo_hist", {"lat": lat, "lon": lon, "mes": date.today().strftime("%Y-%m")},
                            pedir_hist))
    d["time"] = pd.to_datetime(d.time)
    doy = pd.Timestamp(fecha).dayofyear
    dist = np.abs(d.time.dt.dayofyear - doy)
    v = d[np.minimum(dist, 365 - dist) <= 7]
    ang = np.radians(v.wind_direction_10m_dominant.dropna())
    dir_media = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360
    return {"fuente": "climatología Open-Meteo (±7 días, últimos 10 años)",
            "temp_max": float(v.temperature_2m_max.mean()), "temp_media": float(v.temperature_2m_mean.mean()),
            "lluvia_mm": float(v.precipitation_sum.mean()), "prob_lluvia": float((v.precipitation_sum > 1).mean()),
            # SUPUESTO: viento medio diurno ≈ 60% de la máxima diaria (sólo para climatología)
            "viento_kmh": float(v.wind_speed_10m_max.mean() * 0.6), "viento_dir": float(dir_media)}


AJUSTES_CLIMA = {
    "aire": "Aire acondicionado: +0% si la máxima < 22 °C; +4% entre 22 y 27 °C; +7% entre 27 y 32 °C; "
            "+10% arriba de 32 °C. Supuesto propio dentro del rango del Dpto. de Energía de EE.UU. "
            "(fueleconomy.gov: el aire puede bajar el rendimiento más de 25% en calor extremo, sobre todo en "
            "viajes cortos; en ruta el efecto es menor).",
    "lluvia": "Lluvia: el asfalto mojado aumenta la resistencia a la rodadura. Tesis de Chalmers/Volvo (2025): "
              "en ruta muy mojada la resistencia total sube ~5% o más. Aplicamos +5% x probabilidad de lluvia "
              "(promedio de la probabilidad horaria del pronóstico entre las 8 y las 20 h, o la frecuencia "
              "histórica de días con lluvia si la fecha está a más de 15 días).",
    "urbano": "Tramos urbanos o lentos (donde OSRM estima menos de 50 km/h: calles, avenidas, accesos y "
              "travesías por pueblos): se usa el consumo URBANO de la etiqueta oficial, sin importar la velocidad "
              "elegida para la ruta, y el tiempo se calcula con la velocidad de OSRM.",
    "viento": "Viento: se suma la componente de frente (o se resta la de cola) a la velocidad del aire en el "
              "término aerodinámico del modelo físico, tramo por tramo según el rumbo de la ruta.",
}


def factor_aire(temp_max: float) -> float:
    if temp_max is None or np.isnan(temp_max):
        return 0.0
    return 0.0 if temp_max < 22 else 0.04 if temp_max < 27 else 0.07 if temp_max < 32 else 0.10


def factor_lluvia(cl: dict) -> float:
    return 0.05 * cl.get("prob_lluvia", 0.0)


# ----------------------------------------------------------------- precios
_VIG = {"mtime": None, "df": None}


def _vigentes() -> pd.DataFrame:
    """Precios vigentes (se lee una vez y se vuelve a leer sólo si el archivo cambió)."""
    p = C.PROC / "estaciones_vigentes.parquet"
    mt = p.stat().st_mtime
    if _VIG["mtime"] != mt:
        _VIG["df"], _VIG["mtime"] = pd.read_parquet(p), mt
    return _VIG["df"]


def fecha_precios() -> dict:
    """De cuándo son los precios: el más reciente informado y cuántas estaciones tienen precio reciente."""
    v = _vigentes()
    rec = v[v.reciente] if "reciente" in v else v[v.antiguedad_dias <= 120]
    ultima = pd.to_datetime(v["fecha"]).max()
    return {"ultima": ultima, "dias": (pd.Timestamp.now() - ultima).days,
            "estaciones_recientes": int(rec.id_estacion.nunique()) if "id_estacion" in rec else len(rec)}


def precio_provincia(provincia: str, producto: str, fecha: date) -> dict:
    """Precio estimado del litro en la provincia para la fecha (con rango).

    Sólo usa precios RECIENTES (<= 120 días): el archivo de vigentes trae muchos precios viejos de
    estaciones que dejaron de informar (sobre todo desde la Res. 717/2025).
    """
    v = _vigentes()
    v = v[(v.producto_n == producto) & ~v.outlier]
    if "reciente" in v:
        v = v[v.reciente]
    else:
        v = v[v.antiguedad_dias <= 120]
    hs = v["horario"].astype(str).str.lower()
    if hs.str.contains("diurno").any():
        v = v[hs.str.contains("diurno")]
    nac = v.precio
    vp = v[v.provincia_n == provincia]
    if len(vp) >= 3:
        hoy, p25, p75 = float(vp.precio.median()), float(vp.precio.quantile(.25)), float(vp.precio.quantile(.75))
        base_txt = f"mediana de {len(vp)} estaciones con precio reciente"
    elif len(vp) >= 1:
        hoy = float(vp.precio.median())
        # pocas estaciones: el rango sale de la dispersión nacional
        p25, p75 = hoy * float(nac.quantile(.25) / nac.median()), hoy * float(nac.quantile(.75) / nac.median())
        base_txt = f"sólo {len(vp)} estación(es) con precio reciente: MENOS CONFIABLE"
    elif len(nac):
        hoy, p25, p75 = float(nac.median()), float(nac.quantile(.25)), float(nac.quantile(.75))
        base_txt = "sin estaciones con precio reciente en la provincia: se usa la mediana NACIONAL"
    else:
        return {"precio": np.nan, "bajo": np.nan, "alto": np.nan, "metodo": "sin datos"}
    meses = (fecha.year - date.today().year) * 12 + fecha.month - date.today().month
    pr_path = C.PROC / "pronosticos.parquet"
    if meses <= 0 or not pr_path.exists():
        return {"precio": hoy, "bajo": p25, "alto": p75, "metodo": f"precio vigente ({base_txt})"}
    pr_all = pd.read_parquet(pr_path)
    pr_all = pr_all[pr_all.producto == producto]
    if pr_all.empty:
        return {"precio": hoy, "bajo": p25, "alto": p75, "metodo": f"precio vigente sin pronóstico ({base_txt})"}
    pr = pr_all[pr_all.provincia == provincia]
    fuente_pr = "de la provincia"
    if pr.empty:  # provincia sin serie confiable: cambio mediano de las demás provincias
        pr = (pr_all.assign(r=pr_all.precio_pred / pr_all.precio_actual, r10=pr_all.precio_p10 / pr_all.precio_actual,
                            r90=pr_all.precio_p90 / pr_all.precio_actual)
              .groupby("h", as_index=False).agg(r=("r", "median"), r10=("r10", "median"), r90=("r90", "median"),
                                                 mes_origen=("mes_origen", "first"), modelo=("modelo", "first")))
        pr = pr.assign(precio_actual=1.0, precio_pred=pr.r, precio_p10=pr.r10, precio_p90=pr.r90)
        fuente_pr = "mediana del país (la provincia no tiene serie confiable)"
    # Los pronósticos parten del último mes del histórico (mes_origen). Medimos el cambio relativo
    # entre "este mes" y "el mes del viaje" según el modelo, y se lo aplicamos al precio vigente de hoy.
    origen = pd.Timestamp(pr.mes_origen.iloc[0])

    def meses_desde_origen(f):
        return (f.year - origen.year) * 12 + f.month - origen.month

    def rel_en(h, col):
        if h <= 0:
            return 1.0
        f = pr[pr.h == min(h, 3)]
        if f.empty:
            f = pr[pr.h == pr.h.max()]
        f = f.iloc[0]
        return float(f[col] / f.precio_actual)

    h_viaje, h_hoy = meses_desde_origen(fecha), meses_desde_origen(date.today())
    base = rel_en(h_hoy, "precio_pred")
    rel = rel_en(h_viaje, "precio_pred") / base
    rel_lo = rel_en(h_viaje, "precio_p10") / base
    rel_hi = rel_en(h_viaje, "precio_p90") / base
    modelo = pr.modelo.iloc[0]
    metodo = (f"{base_txt}; pronóstico {fuente_pr} a {min(h_viaje, 3)} mes(es) desde {origen:%m/%Y} "
              f"(modelo: {modelo})")
    if h_viaje > 3:
        extra = h_viaje - 3
        infl = 0.02  # SUPUESTO: suba mensual del combustible más allá del horizonte del modelo
        rel, rel_lo, rel_hi = (rel * (1 + infl) ** extra, rel_lo * (1 + infl * .5) ** extra,
                               rel_hi * (1 + infl * 1.5) ** extra)
        metodo += f" + extrapolación de {extra} mes(es) con 2% mensual (SUPUESTO, poco confiable)"
    return {"precio": hoy * rel, "bajo": p25 * min(rel_lo, rel), "alto": p75 * max(rel_hi, rel), "metodo": metodo}


def estaciones_cerca(ruta: pd.DataFrame, producto: str, radio_km: float = 3.0, n: int = 8) -> pd.DataFrame:
    from sklearn.neighbors import BallTree
    v = _vigentes()
    v = v[(v.producto_n == producto) & ~v.outlier & v.lat.notna() & v.lon.notna()
          & (v.lat.abs() > 1) & (v.antiguedad_dias <= 120)]  # sólo precios recientes
    hs = v["horario"].astype(str).str.lower()
    if hs.str.contains("diurno").any():
        v = v[hs.str.contains("diurno")]
    if v.empty:
        return v
    arbol = BallTree(np.radians(ruta[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, idx = arbol.query(np.radians(v[["lat", "lon"]].to_numpy()), k=1)
    v = v.assign(dist_ruta_km=dist[:, 0] * R_TIERRA, km_del_viaje=ruta.km.to_numpy()[idx[:, 0]])
    v = v[v.dist_ruta_km <= radio_km].sort_values("precio")
    cols = [c for c in ["bandera", "empresa", "direccion", "localidad", "provincia_n", "precio", "fecha",
                        "km_del_viaje", "dist_ruta_km", "lat", "lon"] if c in v.columns]
    return v[cols].head(n)


# ----------------------------------------------------------------- cálculo del viaje
VEL_URBANA = 50  # km/h: por debajo de esto (según OSRM) el tramo se considera urbano / lento


def calcular(o: tuple[float, float], d: tuple[float, float], fecha: date, producto: str,
             extraurbano: float | None, params: ParamsVelocidad, c100_usuario: float | None = None,
             velocidades=(100, 130), urbano: float | None = None) -> dict:
    """urbano: consumo urbano (l/100 km) para los tramos urbanos/lentos. Si es None se usa el de ruta."""
    if haversine(o[0], o[1], d[0], d[1]) < 1:
        raise ValueError("El origen y el destino son el mismo lugar: elegí otro destino.")
    r = ruta_osrm(o, d)
    if r["km"] < 1:
        raise ValueError("La ruta encontrada mide menos de 1 km: elegí otro destino.")
    m = muestrear(r["puntos"], vel_tramos=r.get("vel_tramos"))
    m["urbano"] = m.vel_osrm < VEL_URBANA  # NaN (sin dato de OSRM) -> False: todo se toma como ruta
    m["provincia"], fuente_prov = provincias_de_puntos(m)
    km_prov = m.groupby("provincia", sort=False).km_tramo.sum()

    # clima: un punto representativo (el del medio) por provincia
    clima = {}
    for prov, g in m.groupby("provincia", sort=False):
        mid = g.iloc[len(g) // 2]
        try:
            clima[prov] = clima_dia(mid.lat, mid.lon, fecha)
        except Exception as e:
            clima[prov] = {"fuente": f"sin clima ({e.__class__.__name__})", "temp_max": np.nan,
                           "lluvia_mm": np.nan, "prob_lluvia": 0.0, "viento_kmh": 0.0, "viento_dir": 0.0}
    precios = {prov: precio_provincia(prov, producto, fecha) for prov in km_prov.index}

    def litros(v, esc):
        tot = 0.0
        por_prov = {}
        for prov, g in m.groupby("provincia", sort=False):
            cl = clima[prov]
            frente = cl["viento_kmh"] * np.cos(np.radians(cl["viento_dir"] - g.rumbo))
            c = np.array([consumo_velocidad(extraurbano or 0, v, params, w, esc, c100_usuario) for w in frente])
            if urbano is not None:
                c = np.where(g.urbano.to_numpy(), urbano, c)
            aj = 1 + factor_aire(cl["temp_max"]) + factor_lluvia(cl)
            lit = float((g.km_tramo * c / 100).sum() * aj)
            por_prov[prov] = lit
            tot += lit
        return tot, por_prov

    km_urb = float(m.loc[m.urbano, "km_tramo"].sum())
    horas_urb = float((m.loc[m.urbano, "km_tramo"] / m.loc[m.urbano, "vel_osrm"].clip(lower=10)).sum())
    km_ruta = float(m.km_tramo.sum()) - km_urb
    res = {"km": r["km"], "km_urbano": km_urb, "fuente_provincias": fuente_prov, "clima": clima, "precios": precios,
           "km_provincia": km_prov.to_dict(), "puntos": m[["lat", "lon", "km"]], "velocidades": {},
           "horas_osrm": r["horas"]}
    for v in velocidades:
        out = {}
        # Rango: los parámetros "bajo/alto" del modelo no siempre dan menos/más litros (con viento de cola, una
        # parte aerodinámica mayor da MENOS consumo), así que el rango es el mínimo y el máximo de todas las
        # combinaciones de parámetros con el precio bajo y alto.
        esc_lit = {esc: litros(v, esc) for esc in ("med", "bajo", "alto")}
        lit, lp = esc_lit["med"]
        out["med"] = {"litros": lit, "costo": sum(lp[p] * precios[p]["precio"] for p in lp), "litros_prov": lp}
        bajos = [(sum(lp_[p] * precios[p]["bajo"] for p in lp_), l_, lp_) for l_, lp_ in esc_lit.values()]
        altos = [(sum(lp_[p] * precios[p]["alto"] for p in lp_), l_, lp_) for l_, lp_ in esc_lit.values()]
        cb, lb, lpb = min(bajos, key=lambda x: x[0])
        ca, la, lpa = max(altos, key=lambda x: x[0])
        out["bajo"] = {"litros": lb, "costo": cb, "litros_prov": lpb}
        out["alto"] = {"litros": la, "costo": ca, "litros_prov": lpa}
        out["horas"] = horas_urb + km_ruta / v
        res["velocidades"][v] = out
    a, b = res["velocidades"][velocidades[0]]["med"], res["velocidades"][velocidades[1]]["med"]
    res["diferencia_pesos"] = b["costo"] - a["costo"]
    res["diferencia_pct"] = (b["costo"] / a["costo"] - 1) * 100 if a["costo"] else np.nan
    res["estaciones"] = estaciones_cerca(m, producto)
    return res
