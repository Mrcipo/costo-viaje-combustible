"""Configuración central: rutas de carpetas y URLs de las fuentes.

Todo se guarda dentro de la carpeta del proyecto (montada como volumen en Docker),
así los datos quedan en tu disco aunque borres el contenedor.
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
# CVC_DATA permite apuntar a otra carpeta (la usamos para los datos de prueba sintéticos).
DATA = Path(os.environ.get("CVC_DATA", RAIZ / "data"))
RAW = DATA / "raw"            # descargas tal cual vienen
PROC = DATA / "processed"     # tablas limpias / agregadas
GEO = DATA / "geo"            # provincias y localidades (Georef)
CACHE = DATA / "cache"        # respuestas de APIs (rutas, clima) para no repetir pedidos
REPORTES = Path(os.environ.get("CVC_REPORTES", RAIZ / "reportes"))  # reportes de calidad y resultados
REF = RAIZ / "data" / "ref"   # base de consumos armada a mano (siempre la misma)
for _p in (RAW, PROC, GEO, CACHE, REPORTES):
    _p.mkdir(parents=True, exist_ok=True)

# Nombre genérico del cliente HTTP (no mandamos datos personales).
USER_AGENT = "tp-costo-viaje-combustible/1.0 (proyecto academico)"

# --- Precios en surtidor (Secretaría de Energía, Res. 314/2016, licencia CC-BY 4.0) ---
URL_PRECIOS_VIGENTES = (
    "http://datos.energia.gob.ar/dataset/1c181390-5045-475e-94dc-410429be4b17/resource/"
    "80ac25de-a44a-4445-9215-090cf55cfda5/download/precios-en-surtidor-resolucin-3142016.csv"
)
URL_PRECIOS_HISTORICOS = (
    "http://datos.energia.gob.ar/dataset/1c181390-5045-475e-94dc-410429be4b17/resource/"
    "f8dda0d5-2a9f-4d34-b79b-4e63de3995df/download/precios-historicos.csv"
)

# --- Macro: ArgentinaDatos ---
URL_INFLACION = "https://api.argentinadatos.com/v1/finanzas/indices/inflacion"
URL_DOLAR_OFICIAL = "https://api.argentinadatos.com/v1/cotizaciones/dolares/oficial"
URL_DOLAR_BLUE = "https://api.argentinadatos.com/v1/cotizaciones/dolares/blue"

# --- Georef (API del Servicio de Normalización de Datos Geográficos) ---
GEOREF = "https://apis.datos.gob.ar/georef/api"

# --- Consumos certificados (Licencias de Configuración Ambiental, LCA) - Ministerio de Ambiente ---
# El dataset de datos.gob.ar fue dado de baja; el vigente está en el CIAM (dataset 566).
URL_LCA_PAGINA = "https://ciam.ambiente.gob.ar/repositorio.php?tid=10&stid=68&dsid=566"
URL_LCA_XLSX_CONOCIDA = ("https://geo.ambiente.gob.ar/documentos/sinia/emisiones_vehiculares/"
                         "LCA_EEUU_EUROPEO_al_21082026.xlsx")

# --- Rutas y clima ---
OSRM = "https://router.project-osrm.org/route/v1/driving"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Producto de surtidor que corresponde a cada combustible del vehículo y calidad.
# Los nombres exactos se validan contra los datos reales en procesar.py (ver reporte de calidad).
PRODUCTOS = {
    ("nafta", "comun"): "Nafta (súper) entre 92 y 95 Ron",
    ("nafta", "premium"): "Nafta (premium) de más de 95 Ron",
    ("diesel", "comun"): "Gas Oil Grado 2",
    ("diesel", "premium"): "Gas Oil Grado 3",
}
