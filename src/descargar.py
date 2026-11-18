"""Descarga de datos (paso 1 del pipeline: "web scraping / API -> CSV").

Uso (dentro de Docker):
    python -m src.descargar todo         # todo lo necesario
    python -m src.descargar precios      # solo precios vigentes (rápido)
    python -m src.descargar historico    # solo el histórico grande (lento, se puede reanudar)
    python -m src.descargar macro        # inflación y dólar
    python -m src.descargar georef       # provincias y localidades
    python -m src.descargar etiqueta     # intenta bajar la base oficial de la etiqueta vehicular

Cada paso es independiente: si uno falla, los otros siguen. Al final imprime un resumen.
Respetamos las fuentes: un pedido a la vez, pausas cortas, sin forzar bloqueos.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from . import config as C

SES = requests.Session()
SES.headers.update({"User-Agent": C.USER_AGENT})


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def bajar_archivo(url: str, destino: Path, reintentos: int = 30, timeout: int = 120) -> Path:
    """Descarga en streaming con reanudación y reintentos.

    El archivo se va guardando como <nombre>.parcial y SÓLO se renombra al nombre final cuando
    está completo (tamaño igual al que informa el servidor y termina en salto de línea).
    Si se corta, volver a correr el mismo comando retoma desde el .parcial.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    total = None
    for intento in range(1, reintentos + 1):
        ya = parcial.stat().st_size if parcial.exists() else 0
        if total is not None and ya >= total:
            break
        headers = {"Range": f"bytes={ya}-"} if ya else {}
        try:
            with SES.get(url, stream=True, timeout=(30, timeout), headers=headers) as r:
                if r.status_code == 416:  # el servidor dice que ya no hay más bytes
                    break
                r.raise_for_status()
                if ya and r.status_code == 206:
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                        total = int(cr.rsplit("/", 1)[1])
                    modo = "ab"
                else:
                    if ya:
                        _log(f"   el servidor no acepta reanudar: empiezo de cero ({ya/1e6:,.0f} MB descartados)")
                    ya, modo = 0, "wb"
                    cl = r.headers.get("Content-Length")
                    total = int(cl) if cl and cl.isdigit() else None
                bajado, t0, ult = ya, time.time(), 0.0
                with open(parcial, modo) as f:
                    for trozo in r.iter_content(chunk_size=1 << 20):
                        f.write(trozo)
                        bajado += len(trozo)
                        if time.time() - ult > 5:
                            ult = time.time()
                            vel = (bajado - ya) / max(time.time() - t0, 1e-6) / 1e6
                            tot = f"/{total/1e6:,.0f}" if total else ""
                            _log(f"   {destino.name}: {bajado/1e6:,.0f}{tot} MB ({vel:.1f} MB/s)")
            if total is None or parcial.stat().st_size >= total:
                break
            _log(f"   la conexión se cortó en {parcial.stat().st_size/1e6:,.0f}/{total/1e6:,.0f} MB; reintento")
        except (requests.RequestException, OSError) as e:
            _log(f"   intento {intento}/{reintentos} falló: {str(e)[:150]}")
        time.sleep(min(5 * intento, 60))
    tam = parcial.stat().st_size if parcial.exists() else 0
    completo = tam > 0 and (total is None or tam >= total)
    if completo and total is None:  # sin tamaño informado: al menos que no termine a mitad de línea
        with open(parcial, "rb") as f:
            f.seek(-1, 2)
            completo = f.read(1) == b"\n"
    if not completo:
        tot = f" de {total/1e6:,.0f} MB" if total else ""
        raise RuntimeError(f"descarga INCOMPLETA ({tam/1e6:,.0f} MB{tot}). Volvé a correr el mismo comando: "
                           f"retoma desde {parcial.name}")
    parcial.replace(destino)
    _log(f"   OK {destino.name}: {destino.stat().st_size/1e6:,.1f} MB (completo)")
    return destino


def _get_json(url: str, params: dict | None = None, timeout: int = 60):
    for intento in range(1, 4):
        try:
            r = SES.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            _log(f"   intento {intento}/3 {url}: {e}")
            time.sleep(3 * intento)
    raise RuntimeError(f"No se pudo consultar {url}")


def _metadatos(nombre: str, info: dict) -> None:
    p = C.RAW / "_descargas.json"
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta[nombre] = {**info, "descargado": datetime.now().isoformat(timespec="seconds")}
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# ------------------------------------------------------------------ pasos
def paso_precios() -> None:
    _log("Precios VIGENTES por estación (Secretaría de Energía)...")
    p = bajar_archivo(C.URL_PRECIOS_VIGENTES, C.RAW / "precios_vigentes.csv")
    _metadatos("precios_vigentes", {"url": C.URL_PRECIOS_VIGENTES, "bytes": p.stat().st_size})


def paso_historico() -> None:
    _log("Precios HISTÓRICOS (archivo grande; si se corta, volvé a correr y retoma)...")
    p = bajar_archivo(C.URL_PRECIOS_HISTORICOS, C.RAW / "precios_historicos.csv", timeout=120)
    _metadatos("precios_historicos", {"url": C.URL_PRECIOS_HISTORICOS, "bytes": p.stat().st_size})


def paso_macro() -> None:
    _log("Inflación mensual (ArgentinaDatos)...")
    inf = pd.DataFrame(_get_json(C.URL_INFLACION))
    inf.to_csv(C.RAW / "inflacion_mensual.csv", index=False)
    _log(f"   OK inflación: {len(inf)} meses ({inf['fecha'].min()} a {inf['fecha'].max()})")
    for nombre, url in (("dolar_oficial", C.URL_DOLAR_OFICIAL), ("dolar_blue", C.URL_DOLAR_BLUE)):
        time.sleep(1)
        d = pd.DataFrame(_get_json(url))
        d.to_csv(C.RAW / f"{nombre}.csv", index=False)
        _log(f"   OK {nombre}: {len(d)} días ({d['fecha'].min()} a {d['fecha'].max()})")
    _metadatos("macro", {"urls": [C.URL_INFLACION, C.URL_DOLAR_OFICIAL, C.URL_DOLAR_BLUE]})


def paso_georef() -> None:
    _log("Provincias y localidades (Georef)...")
    j = _get_json(f"{C.GEOREF}/provincias", {"campos": "id,nombre,centroide", "max": 30})
    prov = pd.json_normalize(j["provincias"])
    prov.columns = ["centroide_lat" if c == "centroide.lat" else "centroide_lon" if c == "centroide.lon" else c
                    for c in prov.columns]
    prov.to_csv(C.GEO / "provincias.csv", index=False)
    _log(f"   OK provincias: {len(prov)}")
    filas, inicio = [], 0
    while True:
        time.sleep(0.5)
        j = _get_json(f"{C.GEOREF}/localidades", {
            "campos": "id,nombre,provincia.nombre,departamento.nombre,centroide",
            "max": 5000, "inicio": inicio})
        filas += j["localidades"]
        inicio += j["cantidad"]
        if inicio >= j["total"] or j["cantidad"] == 0:
            break
    loc = pd.json_normalize(filas).rename(columns={
        "centroide.lat": "lat", "centroide.lon": "lon",
        "provincia.nombre": "provincia", "departamento.nombre": "departamento"})
    loc.to_csv(C.GEO / "localidades.csv", index=False)
    _log(f"   OK localidades: {len(loc)}")
    _metadatos("georef", {"url": C.GEOREF, "provincias": len(prov), "localidades": len(loc)})


def paso_etiqueta() -> None:
    """Base OFICIAL de consumos certificados (Licencias de Configuración Ambiental, Ministerio de Ambiente).

    El dataset viejo de datos.gob.ar fue dado de baja (404). Los datos vigentes están en el CIAM
    (ciam.ambiente.gob.ar, dataset 566), que enlaza un Excel en geo.ambiente.gob.ar que se actualiza
    con fecha en el nombre. Buscamos el enlace más nuevo en la página; si no aparece, usamos el último conocido.
    """
    import re
    _log("Base oficial de consumos certificados (LCA, Ministerio de Ambiente)...")
    url_xlsx = C.URL_LCA_XLSX_CONOCIDA
    try:
        r = SES.get(C.URL_LCA_PAGINA, timeout=60)
        r.raise_for_status()
        enlaces = re.findall(r'https?://geo\.ambiente\.gob\.ar/[^"\'\s<>]+?\.xlsx', r.text)
        if enlaces:
            url_xlsx = sorted(set(enlaces))[-1]
            _log(f"   enlace encontrado en la página del CIAM: {url_xlsx}")
        else:
            _log("   no encontré el enlace en la página: uso el último conocido")
    except requests.RequestException as e:
        _log(f"   no pude leer la página del CIAM ({e}); uso el enlace conocido")
    dest = C.RAW / "etiqueta"
    dest.mkdir(exist_ok=True)
    p = bajar_archivo(url_xlsx, dest / "lca_consumos.xlsx")
    with open(p, "rb") as f:
        if f.read(2) != b"PK":
            raise RuntimeError("lo que bajó no es un Excel (.xlsx); revisá data/raw/etiqueta/lca_consumos.xlsx")
    _metadatos("etiqueta", {"url": url_xlsx, "pagina": C.URL_LCA_PAGINA, "bytes": p.stat().st_size,
                            "fuente": "Ministerio de Ambiente - UTOEV - Licencias de Configuración Ambiental"})


PASOS = {
    "precios": paso_precios,
    "macro": paso_macro,
    "georef": paso_georef,
    "etiqueta": paso_etiqueta,
    "historico": paso_historico,
}


def main(args: list[str]) -> int:
    pedido = args[0] if args else "todo"
    nombres = list(PASOS) if pedido == "todo" else [pedido]
    resultado = {}
    for n in nombres:
        if n not in PASOS:
            print(f"Paso desconocido: {n}. Opciones: todo, {', '.join(PASOS)}")
            return 2
        try:
            PASOS[n]()
            resultado[n] = "OK"
        except Exception as e:  # seguimos con los demás pasos
            resultado[n] = f"FALLÓ: {e}"
            _log(f"!! {n} falló: {e}")
    print("\n===== RESUMEN DESCARGA =====")
    for n, r in resultado.items():
        print(f"  {n:10s} {r}")
    print("Archivos en data/raw y data/geo. Detalle en data/raw/_descargas.json")
    obligatorios = {"precios", "historico", "macro", "georef"} & set(nombres)
    return 0 if all(resultado[n] == "OK" for n in obligatorios) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
