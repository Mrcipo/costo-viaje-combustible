"""Base de consumos de vehículos + modelo físico simple para pasar de ~100 a ~130 km/h.

La idea física (explicada simple):
  Para mantener una velocidad constante, el motor vence dos fuerzas principales:
  - Rodadura (neumáticos): casi no depende de la velocidad.
  - Aerodinámica (empujar el aire): crece con el CUADRADO de la velocidad del aire.
  El combustible por km es proporcional a la fuerza total, así que:

      consumo(v) = c100 * [ (1 - f) + f * ((v + viento_en_contra) / 100)^2 ]

  donde c100 es el consumo a 100 km/h y f es la "parte aerodinámica" del consumo a 100 km/h.
  Calibramos con datos:
  - c100 = k * consumo_extraurbano_de_la_etiqueta  (k sale de comparar pruebas reales a 100 km/h)
  - f    = (consumo130/consumo100 - 1) / ((130/100)^2 - 1)  (sale de pruebas a 100 y a 130 km/h)
  Usamos la MEDIANA de las pruebas y el rango intercuartil como incertidumbre.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config as C

PESO_URBANO_NEDC = 0.368  # el ciclo mixto NEDC pondera ~36,8% urbano y ~63,2% extraurbano (por distancia)


def cargar_base() -> pd.DataFrame:
    df = pd.read_csv(C.REF / "consumos_vehiculos.csv")
    df["aproximado"] = df["aproximado"].astype(str).str.lower().eq("true")
    falta_mixto = df["consumo_mixto_l100"].isna()
    df.loc[falta_mixto, "consumo_mixto_l100"] = (
        PESO_URBANO_NEDC * df["consumo_urbano_l100"] + (1 - PESO_URBANO_NEDC) * df["consumo_extraurbano_l100"])
    return df


def cargar_base_oficial() -> pd.DataFrame | None:
    """Base OFICIAL (Ministerio de Ambiente, Licencias de Configuración Ambiental), ya limpia por src/etiqueta.py.

    Devuelve las mismas columnas que la base propia. La fuente no informa el año: anio_desde/hasta vacíos.
    """
    p = C.PROC / "consumos_oficiales.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.usable & d.combustible_n.isin(["nafta", "diesel"])].copy()
    d["marca"] = d["marca"].astype(str).str.upper().str.strip()
    claves = ["marca", "modelo", "motor", "transmision_n", "traccion",
              "consumo_urbano_l100", "consumo_extraurbano_l100", "consumo_mixto_l100"]
    d = d.drop_duplicates(subset=[c for c in claves if c in d])
    pot = d["potencia_kw"].map(lambda x: f" · {x:.0f} kW" if pd.notna(x) else "")
    d["version"] = (d["modelo"].astype(str).str.strip() + " · " + d["transmision_n"].fillna("?").astype(str)
                    + " · " + d["traccion"].fillna("").astype(str) + pot)
    # Si dos filas quedan con la misma etiqueta (otra homologación del mismo modelo), primero se distinguen por
    # motor y, si sigue igual, se muestra su consumo mixto: así el usuario ve que hay dos valores oficiales.
    grupo = ["marca", "familia", "version"]
    rep = d.duplicated(grupo, keep=False)
    d.loc[rep, "version"] = d.loc[rep, "version"] + " · motor " + d.loc[rep, "motor"].astype(str)
    rep = d.duplicated(grupo, keep=False)
    d.loc[rep, "version"] = d.loc[rep, "version"] + d.loc[rep, "consumo_mixto_l100"].map(
        lambda x: f" · mixto {x:.2f} l/100 km")
    d = d.drop_duplicates(grupo)  # lo que queda repetido es idéntico para el usuario
    out = pd.DataFrame({
        "id": "LCA-" + d.index.astype(str),
        "marca": d["marca"],
        "modelo": d["familia"],
        "version": d["version"],
        "anio_desde": np.nan, "anio_hasta": np.nan,
        "combustible": d["combustible_n"],
        "motor": d["motor"], "transmision": d["transmision_n"], "traccion": d["traccion"],
        "potencia_kw": d["potencia_kw"],
        "consumo_urbano_l100": d["consumo_urbano_l100"], "consumo_extraurbano_l100": d["consumo_extraurbano_l100"],
        "consumo_mixto_l100": d["consumo_mixto_l100"], "co2_mixto_gkm": d.get("co2_gkm"),
        "ciclo": "etiqueta oficial (NEDC)", "tipo_fuente": "certificado oficial LCA - Ministerio de Ambiente",
        "fuente_url": C.URL_LCA_PAGINA,
        "aproximado": d["hibrido"].astype(bool),
        "nota": np.where(d["hibrido"], "Híbrido: el modelo de velocidad (pensado para motores a combustión) "
                         "es menos confiable.", ""),
    })
    return out.reset_index(drop=True)


def cargar_pruebas() -> pd.DataFrame:
    return pd.read_csv(C.REF / "consumos_velocidad_constante.csv")


# ----------------------------------------------------------------- embudo de selección
def opciones(df: pd.DataFrame, campo: str, **filtros) -> list:
    d = df
    for k, v in filtros.items():
        if v is not None:
            d = d[d[k] == v]
    if campo == "anio":
        anios = sorted({a for _, r in d.iterrows() for a in range(int(r.anio_desde), int(r.anio_hasta) + 1)})
        return anios
    return sorted(d[campo].dropna().unique().tolist())


def elegir(df: pd.DataFrame, marca: str, modelo: str, version: str | None = None,
           anio: int | None = None, combustible: str | None = None) -> tuple[pd.Series | None, str]:
    """Devuelve la fila elegida y un mensaje. Si la combinación exacta no existe, la más cercana."""
    d = df[(df.marca == marca) & (df.modelo == modelo)]
    if d.empty:
        return None, "No tenemos ese modelo en la base. Podés cargar tu consumo real (opción avanzada)."
    if version is not None and (d.version == version).any():
        d = d[d.version == version]
    if combustible is not None and (d.combustible == combustible).any():
        d = d[d.combustible == combustible]
    if anio is None:
        return d.iloc[0], ""
    dentro = d[(d.anio_desde <= anio) & (d.anio_hasta >= anio)]
    if not dentro.empty:
        return dentro.iloc[0], ""
    dist = np.minimum((d.anio_desde - anio).abs(), (d.anio_hasta - anio).abs())
    fila = d.loc[dist.idxmin()]
    return fila, (f"No tenemos datos para el año {anio}. Usamos la versión más cercana disponible "
                  f"({fila.version}, años {fila.anio_desde}-{fila.anio_hasta}). "
                  "Si tu auto es bastante distinto, cargá tu consumo real.")


# ----------------------------------------------------------------- modelo de velocidad
@dataclass
class ParamsVelocidad:
    k_med: float      # c100 / extraurbano (mediana)
    k_p25: float
    k_p75: float
    f_med: float      # parte aerodinámica a 100 km/h (mediana)
    f_p25: float
    f_p75: float
    n_k: int
    n_f: int

    def dict(self):
        return asdict(self)


def calibrar(base: pd.DataFrame | None = None, pruebas: pd.DataFrame | None = None,
             excluir: str | None = None) -> ParamsVelocidad:
    base = cargar_base() if base is None else base
    pr = cargar_pruebas() if pruebas is None else pruebas
    if excluir is not None:
        pr = pr[pr.modelo + " " + pr.version_probada != excluir]
    pares = pr.merge(base[["id", "consumo_extraurbano_l100"]], left_on="id_base_relacionado",
                     right_on="id", how="inner")
    # k se calibra SOLO con filas de etiqueta oficial (IRAM/AITA) no aproximadas: los catálogos
    # comerciales y el ciclo brasileño dan extraurbanos que no son comparables (ver validar_modelo()).
    oficiales = base.loc[base.ciclo.str.contains("etiqueta", case=False, na=False) & ~base.aproximado, "id"]
    pares = pares[pares.id.isin(oficiales)]
    k = (pares.consumo_100_l100 / pares.consumo_extraurbano_l100).dropna()
    r = (pr.consumo_130_l100 / pr.consumo_100_l100).dropna()
    f = (r - 1) / (1.3 ** 2 - 1)
    return ParamsVelocidad(float(k.median()), float(k.quantile(.25)), float(k.quantile(.75)),
                           float(f.median()), float(f.quantile(.25)), float(f.quantile(.75)), len(k), len(f))


def consumo_velocidad(extraurbano: float, v: float, p: ParamsVelocidad,
                      viento_frontal_kmh: float = 0.0, escenario: str = "med",
                      c100_usuario: float | None = None) -> float:
    """Consumo (l/100 km) a velocidad v. escenario: 'med', 'bajo' o 'alto'."""
    k = {"med": p.k_med, "bajo": p.k_p25, "alto": p.k_p75}[escenario]
    f = {"med": p.f_med, "bajo": p.f_p25, "alto": p.f_p75}[escenario]
    c100 = c100_usuario if c100_usuario is not None else k * extraurbano
    v_aire = max(v + viento_frontal_kmh, 0)
    return c100 * ((1 - f) + f * (v_aire / 100) ** 2)


def _tipo_caja(t) -> str:
    k = str(t).lower()
    return "cvt" if "cvt" in k else "manual" if k.startswith("manual") else "automatica"


def c100_medido(id_base: str, fila: pd.Series | None = None) -> tuple[float | None, str]:
    """Si hay una prueba real a 100 km/h para esa versión, la usamos en lugar de estimarla.

    - Fila de la base propia: se busca por su id.
    - Fila de la base OFICIAL: se busca la versión de la base propia equivalente (misma marca, familia,
      combustible, tipo de caja y consumo extraurbano a menos de 0,15 l/100 km) que tenga prueba real.
    """
    pr = cargar_pruebas().dropna(subset=["consumo_100_l100"])
    if not str(id_base).startswith("LCA-"):
        t = pr[pr.id_base_relacionado == id_base]
    else:
        if fila is None:
            return None, ""
        bp = cargar_base()
        bp = bp[bp.id.isin(pr.id_base_relacionado)]
        cand = bp[(bp.marca.str.upper() == str(fila.marca).upper()) & (bp.modelo.str.upper() == str(fila.modelo).upper())
                  & (bp.combustible == fila.combustible)
                  & (bp.transmision.map(_tipo_caja) == _tipo_caja(fila.get("transmision", "")))
                  & ((bp.consumo_extraurbano_l100 - float(fila.consumo_extraurbano_l100)).abs() <= 0.15)]
        if cand.empty:
            return None, ""
        t = pr[pr.id_base_relacionado == cand.iloc[0].id]
    if t.empty:
        return None, ""
    t = t.iloc[0]
    # Si la prueba indica la potencia (p.ej. '204 cv') y la versión oficial también, tienen que coincidir (±10%)
    import re
    cv = re.search(r"(\d{2,3})\s*cv", str(t.version_probada).lower())
    kw = fila.get("potencia_kw") if fila is not None else None
    if cv and kw is not None and pd.notna(kw) and abs(float(kw) * 1.36 - int(cv.group(1))) / int(cv.group(1)) > 0.10:
        return None, ""
    return float(t.consumo_100_l100), f"medición real a 100 km/h ({t.tipo_fuente}): {t.fuente_url}"


def validar_modelo(base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validación "dejando uno afuera": calibro sin el auto i y predigo su consumo a 100 y 130."""
    base = cargar_base() if base is None else base
    pr = cargar_pruebas()
    filas = []
    for _, t in pr.iterrows():
        nombre = t.modelo + " " + t.version_probada
        p = calibrar(base, pr, excluir=nombre)
        b = base[base.id == t.id_base_relacionado]
        pred100 = p.k_med * float(b.consumo_extraurbano_l100.iloc[0]) if len(b) else np.nan
        pred130_desde_real100 = t.consumo_100_l100 * ((1 - p.f_med) + p.f_med * 1.69)
        filas.append({"vehiculo": f"{t.marca} {nombre}", "real_100": t.consumo_100_l100, "pred_100": pred100,
                      "real_130": t.consumo_130_l100, "pred_130": pred130_desde_real100})
    v = pd.DataFrame(filas)
    v["error_100_%"] = (v.pred_100 / v.real_100 - 1) * 100
    v["error_130_%"] = (v.pred_130 / v.real_130 - 1) * 100
    return v.round(2)
