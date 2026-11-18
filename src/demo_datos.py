"""Genera datos SINTÉTICOS con el mismo formato que las fuentes reales, para probar el código.

NO son datos reales. Se usan sólo para tests (python -m src.demo_datos <carpeta>).
Incluye a propósito problemas típicos: provincias escritas distinto, precios en 0,
precios absurdos, estaciones sin coordenadas, GNC, horarios diurno/nocturno.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Centroides aproximados (sólo para la demo).
PROV = {
    "Buenos Aires": (-36.68, -60.56), "Ciudad Autónoma de Buenos Aires": (-34.61, -58.45),
    "Catamarca": (-27.34, -66.95), "Chaco": (-26.39, -60.77), "Chubut": (-43.79, -68.53),
    "Córdoba": (-32.14, -63.80), "Corrientes": (-28.77, -57.80), "Entre Ríos": (-32.06, -59.20),
    "Formosa": (-24.89, -59.93), "Jujuy": (-23.32, -65.76), "La Pampa": (-37.13, -65.45),
    "La Rioja": (-29.69, -67.18), "Mendoza": (-34.63, -68.58), "Misiones": (-26.88, -54.65),
    "Neuquén": (-38.64, -70.12), "Río Negro": (-40.41, -67.23), "Salta": (-24.30, -64.81),
    "San Juan": (-30.86, -68.89), "San Luis": (-33.76, -66.03), "Santa Cruz": (-48.82, -69.96),
    "Santa Fe": (-30.71, -60.95), "Santiago del Estero": (-27.78, -63.25),
    "Tierra del Fuego, Antártida e Islas del Atlántico Sur": (-54.36, -67.64),
    "Tucumán": (-26.95, -65.36),
}
LOCALIDADES = [
    ("Ciudad Autónoma de Buenos Aires", "Ciudad Autónoma de Buenos Aires", -34.6037, -58.3816),
    ("La Plata", "Buenos Aires", -34.9214, -57.9544), ("Mar del Plata", "Buenos Aires", -38.0055, -57.5426),
    ("Bahía Blanca", "Buenos Aires", -38.7183, -62.2663), ("Rosario", "Santa Fe", -32.9442, -60.6505),
    ("Santa Fe", "Santa Fe", -31.6333, -60.7000), ("Córdoba", "Córdoba", -31.4201, -64.1888),
    ("Villa Carlos Paz", "Córdoba", -31.4241, -64.4978), ("Mendoza", "Mendoza", -32.8895, -68.8458),
    ("San Luis", "San Luis", -33.2950, -66.3356), ("Neuquén", "Neuquén", -38.9516, -68.0591),
    ("San Carlos de Bariloche", "Río Negro", -41.1335, -71.3103), ("Salta", "Salta", -24.7821, -65.4232),
    ("San Miguel de Tucumán", "Tucumán", -26.8083, -65.2176), ("Paraná", "Entre Ríos", -31.7319, -60.5238),
    ("Santa Rosa", "La Pampa", -36.6167, -64.2833),
]
PRODUCTOS = [(2, "Nafta (súper) entre 92 y 95 Ron", 1.00), (3, "Nafta (premium) de más de 95 Ron", 1.22),
             (19, "Gas Oil Grado 2", 1.02), (21, "Gas Oil Grado 3", 1.25), (6, "GNC", 0.55)]


def generar(carpeta: Path, semilla: int = 7) -> None:
    rng = np.random.default_rng(semilla)
    raw, geo = carpeta / "raw", carpeta / "geo"
    raw.mkdir(parents=True, exist_ok=True); geo.mkdir(parents=True, exist_ok=True)

    meses = pd.date_range("2017-01-01", "2026-10-01", freq="MS")
    # Inflación mensual sintética con un salto grande (tipo dic-2023) y desaceleración después.
    inf = np.full(len(meses), 2.5)
    t = np.arange(len(meses))
    inf += 1.5 * np.sin(t / 9) + rng.normal(0, 0.4, len(meses))
    salto = meses.get_loc(pd.Timestamp("2023-12-01"))
    inf[salto - 8:salto] += 4; inf[salto] = 25.5; inf[salto + 1:salto + 4] = [20, 13, 11]
    inf[salto + 4:] = np.clip(4 - 0.12 * np.arange(len(meses) - salto - 4), 1.5, None) + rng.normal(0, 0.3, len(meses) - salto - 4)
    pd.DataFrame({"fecha": (meses + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
                  "valor": inf.round(1)}).to_csv(raw / "inflacion_mensual.csv", index=False)
    nivel = np.cumprod(1 + inf / 100)
    dias = pd.date_range(meses[0], meses[-1] + pd.offsets.MonthEnd(0), freq="D")
    usd = np.interp(np.arange(len(dias)), np.linspace(0, len(dias) - 1, len(meses)), 16 * nivel * 0.9)
    usd[dias >= "2023-12-13"] *= 2.0  # devaluación
    for n, f in (("oficial", 1.0), ("blue", 1.35)):
        pd.DataFrame({"casa": n, "compra": (usd * f * 0.97).round(2), "venta": (usd * f).round(2),
                      "fecha": dias.strftime("%Y-%m-%d")}).to_csv(raw / f"dolar_{n}.csv", index=False)

    filas, vig = [], []
    eid = 1000
    for prov, (la, lo) in PROV.items():
        fprov = 1 + rng.normal(0, 0.04) + (0.06 if "Tierra" in prov or "Santa Cruz" in prov else 0)
        nombre_prov = "CAPITAL FEDERAL" if prov.startswith("Ciudad") else prov.upper()
        for k in range(8):
            eid += 1
            bandera = rng.choice(["YPF", "SHELL C.A.P.S.A.", "AXION", "PUMA", "BLANCA"], p=[.5, .2, .15, .1, .05])
            fest = fprov * (1 + rng.normal(0, 0.015)) * (1.03 if bandera == "SHELL C.A.P.S.A." else 1)
            elat, elon = la + rng.normal(0, 0.8), lo + rng.normal(0, 0.8)
            for idp, prod, fp in PRODUCTOS:
                if idp == 6 and rng.random() < 0.6:
                    continue
                base = 17.5 * fp * fest * nivel
                for i, m in enumerate(meses):
                    if rng.random() < 0.55:  # informa ese mes (cuando cambia)
                        dia = int(rng.integers(1, 28))
                        precio = round(float(base[i] * (1 + rng.normal(0, 0.01))), 2)
                        if rng.random() < 0.0008:
                            precio = 0.0
                        if rng.random() < 0.0008:
                            precio = round(precio * 100, 2)  # error de carga (falta la coma)
                        for idh, hor in ((2, "Diurno"), (3, "Nocturno")):
                            filas.append({
                                "indice_tiempo": m.strftime("%Y-%m"), "idempresa": eid, "cuit": f"30-{eid}-1",
                                "empresa": f"ESTACION {eid} S.A.", "direccion": f"RUTA {k} KM {eid % 300}",
                                "localidad": "LOCALIDAD DEMO", "provincia": nombre_prov, "region": "DEMO",
                                "idproducto": idp, "producto": prod, "idtipohorario": idh, "tipohorario": hor,
                                "precio": precio, "fecha_vigencia": f"{m:%Y-%m}-{dia:02d} 08:00:00",
                                "idempresabandera": 1, "empresabandera": bandera,
                                "latitud": "" if (eid % 37 == 0) else round(elat, 5),
                                "longitud": "" if (eid % 37 == 0) else round(elon, 5), "geojson": ""})
                vig.append({**filas[-2], "fecha_vigencia": "2026-10-28 08:00:00",
                            "indice_tiempo": "2026-10", "precio": round(float(base[-1]) * 1.02, 2)})
    pd.DataFrame(filas).to_csv(raw / "precios_historicos.csv", index=False)
    pd.DataFrame(vig).to_csv(raw / "precios_vigentes.csv", index=False)

    pd.DataFrame([{"id": f"{i:02d}", "nombre": p, "centroide_lat": c[0], "centroide_lon": c[1]}
                  for i, (p, c) in enumerate(PROV.items())]).to_csv(geo / "provincias.csv", index=False)
    pd.DataFrame([{"id": i, "nombre": n, "provincia": p, "departamento": "", "lat": la, "lon": lo}
                  for i, (n, p, la, lo) in enumerate(LOCALIDADES)]).to_csv(geo / "localidades.csv", index=False)
    print(f"Datos sintéticos en {carpeta}: {len(filas):,} filas históricas, {len(vig):,} vigentes")


if __name__ == "__main__":
    generar(Path(sys.argv[1] if len(sys.argv) > 1 else "data_demo"))
