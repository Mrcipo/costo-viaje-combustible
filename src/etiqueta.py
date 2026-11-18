"""Base OFICIAL de consumos certificados (LCA - Ministerio de Ambiente).

    python -m src.etiqueta      # inspecciona el Excel y arma data/processed/consumos_oficiales.csv

1. Escribe reportes/calidad_etiqueta.md con las hojas, columnas y primeras filas tal cual vienen
   (para verificar la estructura real antes de confiar en el parser).
2. Normaliza columnas (marca, modelo, motor, transmisión, combustible, consumos, CO2, LCA, etiqueta).
3. Valida rangos (consumos entre 2 y 30 l/100 km, urbano >= extraurbano, mixto entre ambos, etc.).
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from . import config as C
from .procesar import norm_col, sin_tildes

XLSX = C.RAW / "etiqueta" / "lca_consumos.xlsx"

CLAVES = {
    "tipo": ["tipo"],
    "marca": ["marca"],
    "modelo": ["modelo"],
    "traccion": ["traccion"],
    "motor": ["motor"],
    "cilindrada": ["cilindrada"],
    "potencia_kw": ["potencia", "pot"],
    "transmision": ["transmision", "caja"],
    "combustible": ["combustible"],
    "norma_emision": ["standard", "norma"],
    "categoria": ["categoria"],
    "co2_gkm": ["co2"],
    "consumo_urbano_l100": ["urbano"],
    "consumo_extraurbano_l100": ["extraurbano", "extra_urbano", "extra"],
    "consumo_mixto_l100": ["mixto", "combinado", "promedio"],
    "lca": ["lca", "licencia"],
    "etiqueta": ["etiqueta"],
    "eficiencia_categoria": ["eficiencia"],
    "grupo_categoria": ["grupo"],
    "fecha": ["fecha", "vigencia"],
    "anio": ["ano", "anio", "modelo_ano"],
}


def _encontrar_encabezado(raw: pd.DataFrame) -> int | None:
    for i in range(min(len(raw), 30)):
        fila = " ".join(norm_col(str(x)) for x in raw.iloc[i].tolist())
        if "marca" in fila and "modelo" in fila and ("urbano" in fila or "consumo" in fila):
            return i
    return None


def _mapear(cols: list[str]) -> dict[str, str]:
    """Columna original -> nombre canónico. Primero coincidencias exactas, después por contenido."""
    n = {c: norm_col(c) for c in cols}
    salida, usados = {}, set()
    # orden importante: 'extraurbano' antes que 'urbano'; 'eficiencia' antes que 'categoria'
    orden = ["consumo_extraurbano_l100", "consumo_urbano_l100", "consumo_mixto_l100", "eficiencia_categoria",
             "grupo_categoria", "co2_gkm", "anio"] + [k for k in CLAVES if k not in (
              "consumo_extraurbano_l100", "consumo_urbano_l100", "consumo_mixto_l100", "eficiencia_categoria",
              "grupo_categoria", "co2_gkm", "anio")]
    for canon in orden:
        for orig, nc in n.items():
            if orig in salida:
                continue
            if any(k == nc or nc.startswith(k) or f"_{k}" in nc or (len(k) > 4 and k in nc) for k in CLAVES[canon]):
                if canon == "consumo_urbano_l100" and "extra" in nc:
                    continue
                if canon == "categoria" and ("eficiencia" in nc or "grupo" in nc):
                    continue
                if canon in usados:
                    continue
                salida[orig] = canon
                usados.add(canon)
                break
    return salida


DOS_PALABRAS_SI_PRIMERA = {"GRAND", "LAND", "GR", "RANGE", "SANTA", "GRAN", "SERIE", "MODEL", "C-ELYSEE",
                           "AMG", "212"}
FAMILIA_ALIAS = {("RENAULT", "DUSTER OROCH"): "OROCH"}  # la Oroch es una pick-up, no una versión del Duster
PREFIJOS_COMERCIALES = {"NUEVO", "NUEVA", "NEW", "ALL", "ALL-NEW"}  # 'NUEVO DUSTER' es el mismo DUSTER
DOS_PALABRAS_SI_SEGUNDA = {"CROSS", "PLUS", "OROCH", "CRUISER", "SPORT", "TRACK", "SW", "JOY"}


def familia_modelo(m) -> str | None:
    """'CRONOS PRESICION 1.8 MT' -> 'CRONOS'; 'COROLLA CROSS' -> 'COROLLA CROSS'; 'GRAND i10' -> 'GRAND I10'.

    El campo Modelo de la fuente a veces trae varias denominaciones juntas ('STRADA ENDURAN CS 1.3. STRADA
    FREEDOM CS 1.3'): tomamos la primera. Regla simple y documentada, no perfecta.
    """
    if pd.isna(m):
        return None
    t = re.sub(r"(?i)^modelo:\s*", "", str(m)).strip()
    t = re.split(r"[.;/,(]| - | – ", t)[0].strip().upper()
    w = t.split()
    while len(w) > 1 and w[0] in PREFIJOS_COMERCIALES:
        w = w[1:]
    # Mercedes: 'CLASE GLC' es lo mismo que 'GLC'; 'AMG CLASE C' -> 'AMG C'
    if len(w) > 1 and w[0] == "CLASE":
        w = w[1:]
    if len(w) > 2 and w[0] == "AMG" and w[1] == "CLASE":
        w = [w[0]] + w[2:]
    if not w:
        return None
    if len(w) > 1 and (w[0] in DOS_PALABRAS_SI_PRIMERA or w[1] in DOS_PALABRAS_SI_SEGUNDA):
        return f"{w[0]} {w[1]}"
    return w[0]


# Marcas que en la fuente aparecen escritas distinto o como sub-marca de otra.
MARCAS_EQUIVALENTES = {"CITROEN": "CITROËN", "DFAC DONGFENG": "DONGFENG", "MERCEDES-AMG": "MERCEDES-BENZ",
                       "212": "BAIC"}  # "212" es la marca todoterreno del grupo BAIC (modelo 212 T01)
PREFIJO_MODELO = {"212": "212", "MERCEDES-AMG": "AMG"}  # para no perder la identidad del modelo


def normalizar_marcas(d: pd.DataFrame) -> pd.DataFrame:
    """Unifica marcas y separa las homologaciones compartidas ('PEUGEOT / CITROËN').

    Una fila 'PEUGEOT / CITROËN' con modelo 'PARTNER/BERLINGO' pasa a ser dos filas:
    PEUGEOT PARTNER y CITROËN BERLINGO (mismo consumo certificado). Se guarda la marca original.
    """
    d = d.copy()
    d["marca_fuente"] = d["marca"]
    d["marca"] = d["marca"].astype(str).str.upper().str.strip()
    filas = []
    for _, r in d.iterrows():
        # 'FORD / SAGW (ELECTRICO)': la segunda es la empresa del sistema eléctrico, no una marca de autos
        marcas = [m.strip() for m in r["marca"].split("/") if m.strip() and "ELECTRIC" not in m.upper()]
        if len(marcas) == 1:
            m0 = marcas[0]
            modelo = str(r["modelo"])
            modelo = re.sub(r"(?i)^MERCEDES-AMG\s+", "AMG ", modelo)
            if m0 in PREFIJO_MODELO:
                modelo = re.sub(rf"(?i)^{re.escape(m0)}\s*", "", modelo).strip()
                pre = PREFIJO_MODELO[m0]
                modelo = modelo if modelo.upper().startswith(pre + " ") else f"{pre} {modelo}"
            filas.append({**r.to_dict(), "marca": MARCAS_EQUIVALENTES.get(m0, m0), "modelo": modelo})
            continue
        partes = [p.strip(" ;") for p in str(r["modelo"]).split("/") if p.strip(" ;")]
        for i, m0 in enumerate(marcas):
            m_norm = MARCAS_EQUIVALENTES.get(m0, m0)
            clave = sin_tildes(m_norm).upper()[:5]
            con_marca = [p for p in partes if clave in sin_tildes(p).upper()]
            parte = con_marca[0] if con_marca else (partes[i] if i < len(partes) else partes[0])
            filas.append({**r.to_dict(), "marca": m_norm, "modelo": parte.split(";")[0].strip()})
    return pd.DataFrame(filas).reset_index(drop=True)


def norm_transmision(t) -> str | None:
    if pd.isna(t):
        return None
    k = str(t).upper()
    if "CVT" in k:
        return "automática CVT"
    if "DCT" in k or "AUTOMATIZADA" in k or k.strip() == "MTA":
        return "automatizada / doble embrague"
    if k.strip().startswith("AT") or "AUTOM" in k or "DHT" in k:
        return "automática"
    if k.strip().startswith("MT") or "MANUAL" in k:
        return "manual"
    return str(t)


def num(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "fi":
        return s.astype(float)
    t = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(t.str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")


def inspeccionar() -> dict[str, pd.DataFrame]:
    hojas = pd.read_excel(XLSX, sheet_name=None, header=None, dtype=object)
    lineas = ["# Base oficial LCA (Ministerio de Ambiente): estructura real", "",
              f"Archivo: `{XLSX.name}` ({XLSX.stat().st_size/1e6:.1f} MB) — hojas: {len(hojas)}", ""]
    for nombre, raw in hojas.items():
        h = _encontrar_encabezado(raw)
        lineas += [f"## Hoja `{nombre}`: {raw.shape[0]:,} filas x {raw.shape[1]} columnas; encabezado en fila {h}", ""]
        muestra = raw.head(12).fillna("").astype(str).map(lambda x: x.replace("|", "/").replace("\n", " ")[:40])
        lineas += ["```", muestra.to_string(index=False), "```"]
        lineas.append("")
    (C.REPORTES / "estructura_etiqueta.md").write_text("\n".join(lineas), encoding="utf-8")
    return hojas


def construir() -> pd.DataFrame:
    hojas = inspeccionar()
    partes = []
    for nombre, raw in hojas.items():
        h = _encontrar_encabezado(raw)
        if h is None:
            print(f"   hoja '{nombre}': sin encabezado reconocible, se ignora")
            continue
        df = raw.iloc[h + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[h].tolist()]
        df = df.loc[:, [c for c in df.columns if c and c.lower() != "nan"]]
        mapa = _mapear(df.columns.tolist())
        print(f"   hoja '{nombre}': columnas reconocidas -> {mapa}")
        df = df.rename(columns=mapa)[list(mapa.values())]
        df["hoja"] = nombre
        partes.append(df)
    if not partes:
        raise RuntimeError("No se reconoció ninguna hoja. Mirá reportes/estructura_etiqueta.md")
    d = pd.concat(partes, ignore_index=True)
    for c in ("consumo_urbano_l100", "consumo_extraurbano_l100", "consumo_mixto_l100", "co2_gkm",
              "cilindrada", "potencia_kw"):
        if c in d:
            d[c] = num(d[c])
    for c in ("marca", "modelo", "motor", "transmision", "combustible", "traccion"):
        if c in d:
            d[c] = d[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).replace({"nan": np.nan, "": np.nan})
    d = d.dropna(subset=["marca", "modelo"])
    if "fecha" in d:
        d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce", dayfirst=True)
    # OJO: la fuente NO informa el año del modelo. Probamos sacarlo del número de LCA y daba años
    # imposibles (1900, 2051): no se inventa. La columna "anio" queda vacía a propósito.
    if "anio" not in d:
        d["anio"] = pd.NA
    # combustible normalizado + híbridos (en los híbridos es normal gastar MENOS en ciudad que en ruta)
    if "combustible" in d:
        k = d["combustible"].astype(str).str.lower()
        d["combustible_n"] = np.select([k.str.contains("gas ?oil|diesel|di[eé]sel", regex=True),
                                        k.str.contains("nafta|gasolina|petrol", regex=True)],
                                       ["diesel", "nafta"], "otro")
        d["hibrido"] = k.str.contains("electric", regex=False)
    d = normalizar_marcas(d)
    # familia del modelo (para el embudo). Si el texto empieza con la marca ("VOLKSWAGEN VENTO GLI"), se saca.
    def _sacar_marca(ma, mo):
        ma_n, mo_s = sin_tildes(str(ma)).upper(), str(mo).strip()
        return mo_s[len(ma_n) + 1:] if sin_tildes(mo_s).upper().startswith(ma_n + " ") else mo_s
    sin_marca = [_sacar_marca(ma, mo) for ma, mo in zip(d["marca"], d["modelo"])]
    fam = pd.Series([familia_modelo(x) for x in sin_marca], index=d.index)
    fam = fam.str.replace(r"^([A-Z0-9]+)-\1$", r"\1", regex=True)  # 'VIRTUS-VIRTUS' -> 'VIRTUS'
    # unificar variantes de escritura ('GR 86' / 'GR86', 'F150' / 'F-150'): clave sin espacios ni guiones
    clave = fam.str.replace(r"[\s-]", "", regex=True)
    mas_comun = fam.groupby([d["marca"], clave]).agg(lambda x: x.value_counts().index[0])
    d["familia"] = [mas_comun.get((ma, k), f) for ma, k, f in zip(d["marca"], clave, fam)]
    d["familia"] = [FAMILIA_ALIAS.get((ma, f), f) for ma, f in zip(d["marca"], d["familia"])]
    d["transmision_n"] = d["transmision"].map(norm_transmision) if "transmision" in d else np.nan
    # validaciones (se marcan, no se borran)
    u, e, m = (d.get("consumo_urbano_l100"), d.get("consumo_extraurbano_l100"), d.get("consumo_mixto_l100"))
    prob = pd.Series("", index=d.index)
    for nombre, s in (("urbano", u), ("extraurbano", e), ("mixto", m)):
        if s is not None:
            prob = prob.where(~(s.isna()), prob + f"sin {nombre}; ")
            prob = prob.where(~((s < 2) | (s > 30)), prob + f"{nombre} fuera de rango; ")
    if u is not None and e is not None:
        prob = prob.where(~(u < e), prob + "urbano < extraurbano; ")
    if m is not None and u is not None and e is not None:
        fuera = (m < np.minimum(u, e) - 0.15) | (m > np.maximum(u, e) + 0.15)
        prob = prob.where(~fuera, prob + "mixto fuera de [extra, urbano]; ")
    d["problemas"] = prob.str.strip()
    solo_hibrido = d["hibrido"] & (d["problemas"] == "urbano < extraurbano;")
    d["usable"] = (d["problemas"] == "") | solo_hibrido
    d["nota"] = np.where(solo_hibrido, "Híbrido: consume menos en ciudad que en ruta (esperable).", "")
    d["fuente"] = "Ministerio de Ambiente - LCA (ciam.ambiente.gob.ar dataset 566)"
    d.to_csv(C.PROC / "consumos_oficiales.csv", index=False)

    rep = ["", "# Calidad", "", f"Filas: {len(d):,} · marcas: {d.marca.nunique()} · modelos: {d.modelo.nunique()}",
           f"Usables en la app: {int(d.usable.sum()):,} (sin problemas, o híbridos con urbano < extraurbano). "
           f"Excluidas: {int((~d.usable).sum()):,}.",
           "El año del modelo NO está en la fuente (se deja vacío; no se infiere).",
           f"Columnas normalizadas: {', '.join(d.columns)}", "",
           "Problemas detectados (filas):", ""]
    rep += [f"- {k}: {v:,}" for k, v in d.problemas.replace("", "ninguno").value_counts().head(15).items()]
    rep += ["", "Valores distintos por columna:", ""]
    rep += [f"- {c}: {d[c].nunique():,}" + (" (>255)" if d[c].nunique() > 255 else "")
            for c in d.columns if d[c].dtype == object]
    with open(C.REPORTES / "estructura_etiqueta.md", "a", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print(f"   OK: {len(d):,} configuraciones, {d.marca.nunique()} marcas -> data/processed/consumos_oficiales.csv")
    return d


if __name__ == "__main__":
    if not XLSX.exists():
        print("Falta el Excel. Corré: docker compose run --rm datos python -m src.descargar etiqueta")
        sys.exit(1)
    construir()
    print("Reporte: reportes/estructura_etiqueta.md")
