"""Procesamiento y validación de los datos descargados.

Uso:
    python -m src.procesar            # procesa todo lo que encuentre en data/raw
    python -m src.procesar --muestra  # sólo las primeras ~2 millones de filas del histórico (prueba rápida)

Qué hace:
1. Perfil de calidad de cada CSV (reportes/calidad_*.md): columnas, vacíos, cantidad de
   categorías (avisa si una supera 255), rango de fechas, precios imposibles.
2. Normaliza provincias y productos (los nombres reales varían: mayúsculas, tildes, CABA...).
3. Histórico -> precio vigente por estación y mes -> serie mensual por provincia y producto.
4. Macro: inflación -> índice de precios; dólar -> promedio mensual.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# ----------------------------------------------------------------- normalización
def sin_tildes(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm_col(c: str) -> str:
    c = sin_tildes(c).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", c).strip("_")


# Nombre canónico -> posibles nombres de columna en los CSV de Energía (se validan con los reales).
ALIAS = {
    "id_estacion": ["idempresa", "id_empresa", "idestacion", "id_estacion", "cuit_empresa"],
    "empresa": ["empresa", "razon_social"],
    "bandera": ["empresabandera", "empresa_bandera", "bandera"],
    "direccion": ["direccion", "domicilio"],
    "localidad": ["localidad"],
    "provincia": ["provincia"],
    "producto": ["producto"],
    "id_producto": ["idproducto", "id_producto"],
    "horario": ["tipohorario", "tipo_horario", "horario"],
    "precio": ["precio"],
    "fecha": ["fecha_vigencia", "fechavigencia", "fecha"],
    "indice_tiempo": ["indice_tiempo", "periodo"],
    "lat": ["latitud", "lat"],
    "lon": ["longitud", "lon", "lng"],
}


def mapear_columnas(cols: list[str]) -> dict[str, str]:
    """Devuelve {columna_original: nombre_canonico} para las columnas que reconoce."""
    normal = {c: norm_col(c) for c in cols}
    salida = {}
    for canon, opciones in ALIAS.items():
        for orig, n in normal.items():
            if n in opciones and orig not in salida:
                salida[orig] = canon
                break
    return salida


PROVINCIAS_CANON = {
    "capital federal": "Ciudad Autónoma de Buenos Aires",
    "caba": "Ciudad Autónoma de Buenos Aires",
    "ciudad autonoma de buenos aires": "Ciudad Autónoma de Buenos Aires",
    "ciudad de buenos aires": "Ciudad Autónoma de Buenos Aires",
    "buenos aires": "Buenos Aires", "catamarca": "Catamarca", "chaco": "Chaco", "chubut": "Chubut",
    "cordoba": "Córdoba", "corrientes": "Corrientes", "entre rios": "Entre Ríos", "formosa": "Formosa",
    "jujuy": "Jujuy", "la pampa": "La Pampa", "la rioja": "La Rioja", "mendoza": "Mendoza",
    "misiones": "Misiones", "neuquen": "Neuquén", "rio negro": "Río Negro", "salta": "Salta",
    "san juan": "San Juan", "san luis": "San Luis", "santa cruz": "Santa Cruz", "santa fe": "Santa Fe",
    "santiago del estero": "Santiago del Estero",
    "tierra del fuego": "Tierra del Fuego, Antártida e Islas del Atlántico Sur",
    "tierra del fuego, antartida e islas del atlantico sur": "Tierra del Fuego, Antártida e Islas del Atlántico Sur",
    "tucuman": "Tucumán",
}


def norm_provincia(p) -> str | None:
    if pd.isna(p):
        return None
    k = sin_tildes(p).strip().lower()
    k = re.sub(r"\s+", " ", k)
    if k in PROVINCIAS_CANON:
        return PROVINCIAS_CANON[k]
    if k.startswith("tierra del fuego"):
        return PROVINCIAS_CANON["tierra del fuego"]
    return None  # se reporta como no reconocida


def norm_producto(p) -> str | None:
    """Clave canónica: nafta_comun, nafta_premium, diesel_comun, diesel_premium, gnc."""
    if pd.isna(p):
        return None
    k = sin_tildes(p).lower()
    if "gnc" in k or "gas natural" in k:
        return "gnc"
    if "nafta" in k:
        return "nafta_premium" if "premium" in k or "mas de 95" in k else "nafta_comun"
    if "gas oil" in k or "gasoil" in k or "diesel" in k:
        return "diesel_premium" if "3" in k or "premium" in k else "diesel_comun"
    return None


def parse_fecha(s: pd.Series) -> pd.Series:
    muestra = s.dropna().astype(str).head(1000)
    if muestra.str.match(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$").mean() > 0.9:  # formato del histórico real
        return pd.to_datetime(s, errors="coerce", format="%d/%m/%Y %H:%M")
    f = pd.to_datetime(s, errors="coerce", format="ISO8601")
    if f.isna().mean() > 0.05:  # formato distinto (p.ej. dd/mm/aaaa): probamos más lento y flexible
        f2 = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
        f = f.fillna(f2)
    return f


def parse_precio(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "fi":
        return s.astype(float)
    t = s.astype(str).str.strip().str.replace("$", "", regex=False)
    # "1.234,56" -> 1234.56 ; "1234.56" queda igual
    con_coma = t.str.contains(",", regex=False)
    t = t.where(~con_coma, t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(t, errors="coerce")


# ----------------------------------------------------------------- perfil de calidad
def perfil_calidad(df: pd.DataFrame, nombre: str, extra: list[str] | None = None) -> str:
    lineas = [f"# Calidad de datos: {nombre}", "", f"Filas analizadas: {len(df):,}", "",
              "| columna | tipo | % vacíos | valores distintos | aviso |", "|---|---|---|---|---|"]
    for c in df.columns:
        s = df[c]
        vac = s.isna().mean() * 100
        if s.dtype == object:
            vac = (s.isna() | (s.astype(str).str.strip() == "")).mean() * 100
        nun = s.nunique(dropna=True)
        aviso = []
        if vac >= 99.9:
            aviso.append("COLUMNA VACÍA")
        elif vac > 20:
            aviso.append("muchos vacíos")
        if s.dtype == object and nun > 255:
            aviso.append(">255 categorías (no usar como categórica directa)")
        lineas.append(f"| {c} | {s.dtype} | {vac:.1f} | {nun:,} | {'; '.join(aviso)} |")
    if extra:
        lineas += [""] + extra
    texto = "\n".join(lineas) + "\n"
    (C.REPORTES / f"calidad_{nombre}.md").write_text(texto, encoding="utf-8")
    return texto


# ----------------------------------------------------------------- precios
COLS_UTILES = ["id_estacion", "empresa", "bandera", "direccion", "localidad", "provincia",
               "producto", "horario", "precio", "fecha", "indice_tiempo", "lat", "lon"]


def leer_precios(path: Path, chunksize: int | None = None, nrows: int | None = None):
    enc = "utf-8"
    with open(path, "rb") as f:
        cab = f.readline()
    try:
        cab.decode("utf-8")
    except UnicodeDecodeError:
        enc = "latin-1"
    sep = ";" if cab.count(b";") > cab.count(b",") else ","
    cols = pd.read_csv(path, nrows=0, sep=sep, encoding=enc).columns.tolist()
    mapa = mapear_columnas(cols)
    usar = [c for c, k in mapa.items() if k in COLS_UTILES]
    lector = pd.read_csv(path, sep=sep, encoding=enc, usecols=usar, dtype=str,
                         chunksize=chunksize, nrows=nrows, low_memory=False)
    return lector, mapa, cols


def limpiar_chunk(df: pd.DataFrame, mapa: dict) -> pd.DataFrame:
    df = df.rename(columns=mapa)
    df["precio"] = parse_precio(df["precio"])
    # normalizamos sobre los valores únicos (mucho más rápido que fila por fila)
    df["provincia_n"] = df["provincia"].map({u: norm_provincia(u) for u in df["provincia"].unique()})
    df["producto_n"] = df["producto"].map({u: norm_producto(u) for u in df["producto"].unique()})
    col_f = "fecha" if "fecha" in df else "indice_tiempo"
    df["fecha_original"] = df[col_f]
    df["fecha"] = parse_fecha(df[col_f])
    if "horario" not in df:
        df["horario"] = "sin dato"
    for c in ("lat", "lon"):
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    return df


FECHA_MIN = pd.Timestamp("2016-01-01")
DIAS_RECIENTE = 120  # la Res. 314/2016 empieza en 2016


def marcar_fecha_invalida(df: pd.DataFrame) -> pd.Series:
    """Fechas vacías, imposibles (años 1572, 0008, 9920... vistos en los datos reales) o futuras."""
    return df["fecha"].isna() | (df["fecha"] < FECHA_MIN) | (df["fecha"] > pd.Timestamp.now() + pd.Timedelta(days=1))


def marcar_precio_invalido(df: pd.DataFrame) -> pd.Series:
    """Precios imposibles (errores de carga): <=0, o fuera de un rango absoluto muy amplio.

    El rango fino (outliers relativos a la provincia) se hace después con medianas.
    """
    return (df["precio"].isna()) | (df["precio"] <= 0) | (df["precio"] > 50_000)


def procesar_historico(path: Path, muestra: bool = False) -> None:
    print(f"Procesando histórico {path.name} ({path.stat().st_size/1e6:,.0f} MB)...")
    lector, mapa, cols = leer_precios(path, chunksize=500_000, nrows=2_000_000 if muestra else None)
    parciales, perfil_df = [], []
    n_total = n_invalidos = n_fecha_mala = 0
    ejemplos_fecha = []
    productos_vistos, provincias_no = {}, {}
    for i, ch in enumerate(lector):
        ch = limpiar_chunk(ch, mapa)
        n_total += len(ch)
        if i == 0:
            perfil_df.append(ch.head(200_000))
        inval = marcar_precio_invalido(ch)
        n_invalidos += int(inval.sum())
        fmala = marcar_fecha_invalida(ch)
        n_fecha_mala += int(fmala.sum())
        if fmala.any() and len(ejemplos_fecha) < 8:
            col_f = "fecha_original" if "fecha_original" in ch else "fecha"
            ejemplos_fecha += ch.loc[fmala, col_f].astype(str).head(8 - len(ejemplos_fecha)).tolist()
        for p, n in ch["producto"].value_counts().items():
            productos_vistos[p] = productos_vistos.get(p, 0) + n
        for p, n in ch.loc[ch["provincia_n"].isna(), "provincia"].value_counts().items():
            provincias_no[p] = provincias_no.get(p, 0) + n
        ok = ch[~inval & ~fmala & ch["provincia_n"].notna() & ch["producto_n"].notna()].copy()
        ok["mes"] = ok["fecha"].dt.to_period("M").dt.to_timestamp()
        ok = ok.sort_values("fecha")
        claves = ["id_estacion", "producto_n", "horario", "mes"]
        ult = ok.groupby(claves, as_index=False).last()
        extra_cols = [c for c in ("precio", "fecha", "provincia_n", "localidad", "bandera", "lat", "lon")
                      if c in ult.columns]
        parciales.append(ult[claves + extra_cols])
        print(f"   bloque {i+1}: {n_total:,} filas leídas", flush=True)
    est = pd.concat(parciales, ignore_index=True).sort_values("fecha")
    est = est.groupby(["id_estacion", "producto_n", "horario", "mes"], as_index=False).last()

    extra = ["## Productos encontrados (nombre original -> clave)", ""]
    for p, n in sorted(productos_vistos.items(), key=lambda x: -x[1]):
        extra.append(f"- `{p}` -> `{norm_producto(p)}` ({n:,} filas)")
    extra += ["", f"## Precios inválidos (<=0, vacíos o > 50.000): {n_invalidos:,} de {n_total:,}", "",
              f"## Fechas inválidas (vacías, antes de 2016 o futuras): {n_fecha_mala:,}. "
              f"Ejemplos: {', '.join(map(str, ejemplos_fecha))}", ""]
    if provincias_no:
        extra += ["## Provincias NO reconocidas (se descartan)", ""] + \
                 [f"- `{p}`: {n:,}" for p, n in provincias_no.items()]
    extra += ["", f"Rango de meses: {est['mes'].min():%Y-%m} a {est['mes'].max():%Y-%m}",
              f"Columnas originales: {', '.join(cols)}"]
    perfil_calidad(pd.concat(perfil_df), "precios_historicos", extra)

    est = rellenar_meses(est)
    est = marcar_outliers_estacion(est)
    est.to_parquet(C.PROC / "estaciones_mes.parquet", index=False)
    serie = serie_provincial(est)
    serie.to_parquet(C.PROC / "precios_mensuales.parquet", index=False)
    import json
    limpieza = {
        "filas_leidas": int(n_total), "precios_invalidos_descartados": int(n_invalidos),
        "fechas_invalidas_descartadas": int(n_fecha_mala), "ejemplos_fechas_invalidas": ejemplos_fecha,
        "provincias_no_reconocidas": {str(k): int(v) for k, v in provincias_no.items()},
        "estacion_mes_total": int(len(est)), "estacion_mes_fuera_de_rango_excluidos": int(est.outlier.sum()),
        "estacion_mes_fuera_de_rango_informados": int((est.outlier & est.informado).sum()),
        "provincia_mes_total": int(len(serie)),
        "provincia_mes_no_confiables_excluidos": int((~serie.confiable).sum()),
        "min_estaciones": MIN_ESTACIONES, "meses_arrastre_precio": 3,
    }
    (C.REPORTES / "resumen_limpieza.json").write_text(json.dumps(limpieza, indent=2, ensure_ascii=False))
    cob = (est[est.informado & (est.producto_n == "nafta_comun")].groupby("mes").id_estacion.nunique()
           .rename("estaciones_que_informan"))
    cob.to_frame().to_csv(C.REPORTES / "cobertura_mensual.csv")
    with open(C.REPORTES / "calidad_precios_historicos.md", "a", encoding="utf-8") as f:
        f.write("\n## Cobertura: estaciones que informan nafta súper por mes (últimos 24 meses)\n\n"
                + cob.tail(24).to_string() + "\n\nOJO: la Res. 717/2025 (BO 2/6/2025) derogó la obligación de "
                "informar (Res. 314/2016); desde jul-2025 informan muchas menos estaciones.\n"
                f"Provincia-mes con menos de {MIN_ESTACIONES} estaciones (marcados no confiables): "
                f"{int((~serie.confiable).sum())} de {len(serie)}\n")
    print(f"   OK: {len(est):,} estación-mes, {len(serie):,} provincia-producto-mes")


def rellenar_meses(est: pd.DataFrame, max_meses: int = 3) -> pd.DataFrame:
    """Las estaciones informan cuando cambian el precio: el precio sigue vigente los meses siguientes.

    Completamos hacia adelante hasta `max_meses` (después asumimos que la estación dejó de informar).
    Sólo 3 meses: desde la Res. 717/2025 (2/6/2025), que derogó la obligación de informar (Res. 314/2016),
    muchas estaciones dejaron de informar; arrastrar su último precio más tiempo bajaría la mediana
    artificialmente en un contexto de precios en suba.
    Versión vectorizada (sin bucles por estación) para que ande rápido con millones de filas.
    """
    claves = ["id_estacion", "producto_n", "horario"]
    est = est.copy()
    est["_k"] = est.groupby(claves, sort=False, dropna=False).ngroup()
    est["_m"] = est["mes"].dt.year * 12 + est["mes"].dt.month - 1
    rng = est.groupby("_k")["_m"].agg(["min", "max"])
    largo = (rng["max"] - rng["min"] + 1).to_numpy()
    k_rep = np.repeat(rng.index.to_numpy(), largo)
    inicio = np.repeat(rng["min"].to_numpy(), largo)
    offs = np.arange(largo.sum()) - np.repeat(np.cumsum(largo) - largo, largo)
    grilla = pd.DataFrame({"_k": k_rep, "_m": inicio + offs})
    full = grilla.merge(est, on=["_k", "_m"], how="left")
    full["informado"] = full["precio"].notna()
    cols = [c for c in full.columns if c not in ("_k", "_m", "informado")]
    full[cols] = full.groupby("_k")[cols].ffill(limit=max_meses)
    full = full.dropna(subset=["precio"])
    full["mes"] = pd.to_datetime(dict(year=full["_m"] // 12, month=full["_m"] % 12 + 1, day=1))
    return full.drop(columns=["_k", "_m"]).reset_index(drop=True)


def marcar_outliers_estacion(est: pd.DataFrame, z_lim: float = 5.0) -> pd.DataFrame:
    """z robusto (mediana/MAD) del precio de cada estación contra su provincia ese mes."""
    claves = ["provincia_n", "producto_n", "mes"]
    med = est.groupby(claves)["precio"].transform("median")
    desv = (est["precio"] - med).abs()
    mad = desv.groupby([est[c] for c in claves]).transform("median")
    est["z_robusto"] = 0.6745 * (est["precio"] - med) / mad.replace(0, np.nan)
    est["ratio_mediana"] = est["precio"] / med
    # z robusto alto Y diferencia de más de 10% (las provincias tienen precios muy parejos, así que el MAD
    # es chico y un z alto solo no alcanza: una estación 6% más cara no es un error)
    est["outlier"] = (((est["z_robusto"].abs() > z_lim) & ((est["ratio_mediana"] - 1).abs() > 0.10))
                      | (est["ratio_mediana"] < 0.5) | (est["ratio_mediana"] > 2))
    return est


MIN_ESTACIONES = 3  # menos estaciones que esto en una provincia-mes: no es representativo


def serie_provincial(est: pd.DataFrame, horario: str | None = None) -> pd.DataFrame:
    d = est[~est["outlier"]]
    if horario is None:  # preferimos diurno si existe
        hs = d["horario"].astype(str).str.lower()
        if hs.str.contains("diurno").any():
            d = d[hs.str.contains("diurno")]
    g = d.groupby(["provincia_n", "producto_n", "mes"])["precio"]
    s = pd.concat([g.median().rename("precio_mediana"), g.quantile(0.25).rename("precio_p25"),
                   g.quantile(0.75).rename("precio_p75"), g.size().rename("n_estaciones")], axis=1).reset_index()
    s = s.rename(columns={"provincia_n": "provincia", "producto_n": "producto"})
    pocas = s.n_estaciones < MIN_ESTACIONES
    s["confiable"] = ~pocas
    return s


def procesar_vigentes(path: Path) -> None:
    print("Procesando precios vigentes...")
    df, mapa, cols = leer_precios(path)
    df = limpiar_chunk(df, mapa)
    inval = marcar_precio_invalido(df)
    extra = [f"Precios inválidos: {int(inval.sum()):,} de {len(df):,}",
             f"Fechas de vigencia: {df['fecha'].min()} a {df['fecha'].max()}",
             f"Columnas originales: {', '.join(cols)}",
             "", "## Productos", ""] + [f"- `{p}` -> `{norm_producto(p)}` ({n:,})"
                                         for p, n in df["producto"].value_counts().items()]
    sin_coord = df["lat"].isna() | df["lon"].isna() | (df["lat"].abs() < 1) if "lat" in df else None
    if sin_coord is not None:
        extra.append(f"\nEstaciones sin coordenadas válidas: {int(sin_coord.sum()):,}")
    perfil_calidad(df, "precios_vigentes", extra)
    df = df[~inval & df["provincia_n"].notna() & df["producto_n"].notna()].copy()
    # El archivo "vigentes" incluye estaciones que dejaron de informar hace años: su "precio vigente"
    # es viejo. Los separamos y los outliers se calculan SOLO contra precios recientes.
    df["antiguedad_dias"] = (pd.Timestamp.now() - df["fecha"]).dt.days
    df["reciente"] = df["antiguedad_dias"] <= DIAS_RECIENTE
    rec = marcar_outliers_estacion(df[df.reciente].assign(mes=pd.Timestamp("2000-01-01")))
    df = df.join(rec[["z_robusto", "ratio_mediana", "outlier"]])
    df["outlier"] = df["outlier"].astype("boolean").fillna(False).astype(bool)
    df["estado"] = np.where(~df.reciente, "desactualizado (> 120 días)",
                            np.where(df.outlier, "fuera de rango", "ok"))
    with open(C.REPORTES / "calidad_precios_vigentes.md", "a", encoding="utf-8") as f:
        f.write("\n## Estado de los precios 'vigentes'\n\n" +
                df.groupby(["producto_n", "estado"]).size().unstack(fill_value=0).to_string() + "\n")
    df.to_parquet(C.PROC / "estaciones_vigentes.parquet", index=False)
    print(f"   OK: {len(df):,} filas estación-producto; recientes: {int(df.reciente.sum()):,}; "
          f"fuera de rango: {int(df.outlier.sum())}; desactualizadas: {int((~df.reciente).sum()):,}")


# ----------------------------------------------------------------- macro
def procesar_macro() -> pd.DataFrame:
    print("Procesando inflación y dólar...")
    inf = pd.read_csv(C.RAW / "inflacion_mensual.csv")
    inf["mes"] = pd.to_datetime(inf["fecha"]).dt.to_period("M").dt.to_timestamp()
    inf = inf.groupby("mes", as_index=False)["valor"].last().rename(columns={"valor": "inflacion_mm"})
    inf = inf.sort_values("mes")
    inf["ipc"] = (1 + inf["inflacion_mm"] / 100).cumprod()
    inf["ipc"] = inf["ipc"] / inf["ipc"].iloc[-1] * 100  # base = último mes disponible = 100
    partes = [inf]
    for nombre in ("dolar_oficial", "dolar_blue"):
        p = C.RAW / f"{nombre}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["mes"] = pd.to_datetime(d["fecha"]).dt.to_period("M").dt.to_timestamp()
        partes.append(d.groupby("mes")["venta"].mean().rename(f"usd_{nombre.split('_')[1]}").reset_index())
    macro = partes[0]
    for p in partes[1:]:
        macro = macro.merge(p, on="mes", how="outer")
    macro = macro.sort_values("mes")
    macro.to_csv(C.PROC / "macro_mensual.csv", index=False)
    perfil_calidad(macro, "macro", [f"Base IPC = 100 en {inf['mes'].iloc[-1]:%Y-%m}"])
    print(f"   OK macro: {macro['mes'].min():%Y-%m} a {macro['mes'].max():%Y-%m}")
    return macro


def main(args: list[str]) -> int:
    muestra = "--muestra" in args
    hechos = []
    if (C.RAW / "inflacion_mensual.csv").exists():
        procesar_macro(); hechos.append("macro")
    if (C.RAW / "precios_vigentes.csv").exists():
        procesar_vigentes(C.RAW / "precios_vigentes.csv"); hechos.append("vigentes")
    if (C.RAW / "precios_historicos.csv").exists():
        procesar_historico(C.RAW / "precios_historicos.csv", muestra=muestra); hechos.append("historico")
    print("\n===== RESUMEN PROCESAMIENTO =====")
    print("Procesado:", ", ".join(hechos) or "nada (¿corriste la descarga?)")
    print("Mirá los reportes de calidad en la carpeta reportes/ (calidad_*.md)")
    return 0 if hechos else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
