"""Prueba de punta a punta con datos SINTÉTICOS (sin internet).

    python -m pytest -q tests/

Las APIs externas (OSRM, Georef, Open-Meteo) se reemplazan por respuestas falsas.
"""
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

TMP = Path(tempfile.mkdtemp(prefix="cvc_demo_"))
os.environ["CVC_DATA"] = str(TMP / "data")
os.environ["CVC_REPORTES"] = str(TMP / "reportes")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import analisis, config as C, consumos, demo_datos, procesar, viaje  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def datos():
    demo_datos.generar(C.DATA)
    assert procesar.main([]) == 0
    assert analisis.main([]) == 0


def test_calidad_y_normalizacion():
    rep = (C.REPORTES / "calidad_precios_historicos.md").read_text()
    assert "Productos encontrados" in rep
    est = pd.read_parquet(C.PROC / "estaciones_mes.parquet")
    assert "Ciudad Autónoma de Buenos Aires" in set(est.provincia_n)   # venía como CAPITAL FEDERAL
    assert est.outlier.any()                                             # los precios x100 se detectan
    assert (est.loc[est.ratio_mediana > 20, "outlier"]).all()
    serie = pd.read_parquet(C.PROC / "precios_mensuales.parquet")
    assert serie.provincia.nunique() == 24


def test_anomalias_detectan_salto_dic2023():
    a = pd.read_parquet(C.PROC / "anomalias_series.parquet")
    dic = a[(a.mes == "2023-12-01") & (a.producto == "nafta_comun")]
    assert dic.n_metodos.ge(1).mean() > 0.5


def test_validacion_y_pronosticos():
    r = pd.read_csv(C.REPORTES / "validacion_modelos.csv")
    assert set(r.h) == {1, 2, 3}
    assert "ingenuo (último precio)" in set(r.modelo)
    p = pd.read_parquet(C.PROC / "pronosticos.parquet")
    assert (p.precio_p10 <= p.precio_pred).all() and (p.precio_pred <= p.precio_p90).all()


def test_modelo_velocidad():
    p = consumos.calibrar()
    assert 0.95 < p.k_med < 1.4 and 0.2 < p.f_med < 0.8
    c100 = consumos.consumo_velocidad(5.0, 100, p)
    c130 = consumos.consumo_velocidad(5.0, 130, p)
    assert 1.1 < c130 / c100 < 1.7
    assert consumos.consumo_velocidad(5.0, 100, p, viento_frontal_kmh=20) > c100  # viento de frente
    v = consumos.validar_modelo()
    assert v["error_130_%"].abs().median() < 15


def test_embudo_y_mas_cercana():
    b = consumos.cargar_base()
    assert b.consumo_mixto_l100.notna().all()
    assert len(b) >= 20 and b.fuente_url.str.startswith("http").all()
    fila, aviso = consumos.elegir(b, "Renault", "Sandero", "RS 2.0 MT6", 2010)
    assert fila is not None and "más cercana" in aviso


def test_viaje_completo(monkeypatch):
    pts = [[-34.60 + i * 0.05, -58.38 - i * 0.12] for i in range(50)]  # línea hacia el oeste
    monkeypatch.setattr(viaje, "ruta_osrm", lambda o, d: {"km": 700.0, "horas": 7.5, "puntos": pts})
    monkeypatch.setattr(viaje, "provincias_de_puntos", lambda m: (
        pd.Series(["Buenos Aires"] * (len(m) // 2) + ["Córdoba"] * (len(m) - len(m) // 2), index=m.index), "test"))
    monkeypatch.setattr(viaje, "clima_dia", lambda la, lo, f: {
        "fuente": "test", "temp_max": 30.0, "temp_media": 25.0, "lluvia_mm": 0, "prob_lluvia": 0.2,
        "viento_kmh": 15.0, "viento_dir": 270.0})
    p = consumos.calibrar()
    for f in (date.today(), date.today() + timedelta(days=60), date.today() + timedelta(days=200)):
        r = viaje.calcular((-34.6, -58.4), (-31.4, -64.2), f, "nafta_comun", 4.8, p)
        assert r["velocidades"][130]["med"]["costo"] > r["velocidades"][100]["med"]["costo"] > 0
        for v in (100, 130):  # el rango siempre contiene al valor central (también con viento de cola)
            x = r["velocidades"][v]
            assert x["bajo"]["costo"] <= x["med"]["costo"] <= x["alto"]["costo"]
        assert 10 < r["diferencia_pct"] < 60
        assert set(r["km_provincia"]) == {"Buenos Aires", "Córdoba"}


def test_app_streamlit(monkeypatch):
    """Corre la app de punta a punta (sin internet) con el probador oficial de Streamlit."""
    from streamlit.testing.v1 import AppTest
    pts = [[-34.60 + i * 0.05, -58.38 - i * 0.12] for i in range(50)]
    monkeypatch.setattr(viaje, "ruta_osrm", lambda o, d: {"km": 700.0, "horas": 7.5, "puntos": pts})
    monkeypatch.setattr(viaje, "provincias_de_puntos", lambda m: (
        pd.Series(["Buenos Aires"] * (len(m) // 2) + ["Córdoba"] * (len(m) - len(m) // 2), index=m.index), "test"))
    monkeypatch.setattr(viaje, "clima_dia", lambda la, lo, f: {
        "fuente": "test", "temp_max": 18.0, "temp_media": 14.0, "lluvia_mm": 0, "prob_lluvia": 0.0,
        "viento_kmh": 10.0, "viento_dir": 90.0})
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.selectbox[0].select("Renault").run()
    at.selectbox[1].select("Sandero").run()
    at.selectbox[2].select("RS 2.0 MT6").run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4
    assert any("aproximado" in m.value.lower() for m in at.markdown)
    # pronóstico a 1, 2 y 3 meses visible en la app
    assert any("próximos meses" in s.value for s in at.subheader)
    tabla_pron = next(df.value for df in at.dataframe if any("+3 meses" in c for c in df.value.columns))
    assert len(tabla_pron) >= 1 and any("+1 mes" in c for c in tabla_pron.columns)
    assert any("Error medio en la validación" in c.value for c in at.caption)
    # diésel: el filtro deja solo vehículos a gasoil y la calidad ofrece grado 2 / grado 3
    at.radio[0].set_value("diesel").run()
    assert not at.exception, at.exception
    at.selectbox[0].select("Toyota").run()
    at.selectbox[1].select("Hilux").run()
    assert set(at.selectbox[0].options) <= {"Toyota", "Ford"}
    cal = at.radio[1]
    assert "Gasoil" in cal.label and any("grado 3" in o for o in cal.options)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4


def test_marcas_unificadas():
    from src import etiqueta
    d = pd.DataFrame({"marca": ["PEUGEOT / CITROËN", "212", "MERCEDES-AMG", "CITROEN", "FORD / SAGW (ELECTRICO)"],
                      "modelo": ["PARTNER/BERLINGO", "T01", "MERCEDES-AMG GT C", "C4 CACTUS", "KUGA HEV"]})
    n = etiqueta.normalizar_marcas(d)
    pares = set(zip(n.marca, n.modelo))
    assert ("PEUGEOT", "PARTNER") in pares and ("CITROËN", "BERLINGO") in pares
    assert ("BAIC", "212 T01") in pares and ("MERCEDES-BENZ", "AMG GT C") in pares
    assert "CITROEN" not in set(n.marca) and not any("SAGW" in m for m in n.marca)
    assert etiqueta.familia_modelo("NUEVO DUSTER 1.6") == "DUSTER"
    assert etiqueta.familia_modelo("CLASE GLC 300") == "GLC"


def test_localidades_sin_duplicados(tmp_path, monkeypatch):
    geo = tmp_path / "geo"
    geo.mkdir()
    pd.DataFrame([
        {"id": 1, "nombre": "Canning", "provincia": "Buenos Aires", "departamento": "Esteban Echeverría", "lat": -34.857, "lon": -58.498},
        {"id": 2, "nombre": "Canning", "provincia": "Buenos Aires", "departamento": "Ezeiza", "lat": -34.95, "lon": -58.60},
        {"id": 3, "nombre": "Avellaneda", "provincia": "Buenos Aires", "departamento": "Avellaneda", "lat": -34.6638, "lon": -58.3627},
        {"id": 4, "nombre": "Avellaneda", "provincia": "Buenos Aires", "departamento": "Avellaneda", "lat": -34.6627, "lon": -58.3652},
    ]).to_csv(geo / "localidades.csv", index=False)
    monkeypatch.setattr(C, "GEO", geo)
    L = viaje.localidades()
    assert sorted(L.nombre) == ["Avellaneda", "Canning (Esteban Echeverría)", "Canning (Ezeiza)"]


def test_tramos_urbanos_y_validaciones(monkeypatch):
    pts = [[-34.60 + i * 0.05, -58.38 - i * 0.12] for i in range(50)]
    vel = [30.0] * 5 + [90.0] * 44  # los primeros tramos son urbanos (30 km/h según OSRM)
    monkeypatch.setattr(viaje, "ruta_osrm", lambda o, d: {"km": 600.0, "horas": 7.0, "puntos": pts, "vel_tramos": vel})
    monkeypatch.setattr(viaje, "provincias_de_puntos", lambda m: (pd.Series("Buenos Aires", index=m.index), "test"))
    monkeypatch.setattr(viaje, "clima_dia", lambda la, lo, f: {
        "fuente": "test", "temp_max": 18.0, "temp_media": 14.0, "lluvia_mm": 0, "prob_lluvia": 0.0,
        "viento_kmh": 0.0, "viento_dir": 0.0})
    p = consumos.calibrar()
    sin_urb = viaje.calcular((-34.6, -58.38), (-32.0, -64.0), date.today(), "nafta_comun", 5.0, p)
    con_urb = viaje.calcular((-34.6, -58.38), (-32.0, -64.0), date.today(), "nafta_comun", 5.0, p, urbano=9.0)
    assert con_urb["km_urbano"] > 50
    assert con_urb["velocidades"][100]["med"]["litros"] > sin_urb["velocidades"][100]["med"]["litros"]
    # en los tramos urbanos la velocidad elegida no cambia el consumo: la diferencia 100 vs 130 se achica
    assert con_urb["diferencia_pct"] < sin_urb["diferencia_pct"]
    with pytest.raises(ValueError):
        viaje.calcular((-34.6, -58.38), (-34.6, -58.38), date.today(), "nafta_comun", 5.0, p)
