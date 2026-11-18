"""Series de tiempo, anomalías y predicción del precio del litro.

Uso:  python -m src.analisis     (necesita haber corrido src.procesar)

Salidas en data/processed/:
  serie_real.parquet         precio mensual por provincia/producto en pesos corrientes, constantes y USD
  anomalias_series.parquet   saltos de precio detectados (z-score móvil, STL, Isolation Forest)
  anomalias_estaciones.parquet  estaciones con precio fuera de rango respecto de su provincia
  validacion_modelos.csv     error de cada modelo vs. el ingenuo (validación temporal)
  pronosticos.parquet        precio estimado a 1-3 meses por provincia y producto, con rango
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

from . import config as C

warnings.filterwarnings("ignore", category=UserWarning)
PRODUCTOS_ANALISIS = ["nafta_comun", "nafta_premium", "diesel_comun", "diesel_premium"]


# ================================================================ series
def cargar_serie() -> pd.DataFrame:
    s = pd.read_parquet(C.PROC / "precios_mensuales.parquet")
    m = pd.read_csv(C.PROC / "macro_mensual.csv", parse_dates=["mes"])
    s = s[s.producto.isin(PRODUCTOS_ANALISIS)]
    if "confiable" in s:
        s = s[s.confiable]
    s = s[s.mes >= "2017-01-01"]  # en 2016 el sistema casi no tenía estaciones informando
    s = s[s.mes < pd.Timestamp.today().to_period("M").to_timestamp()]  # el mes en curso está incompleto
    s = s.merge(m, on="mes", how="left")
    s = s.sort_values(["provincia", "producto", "mes"])
    s["ipc"] = s.groupby(["provincia", "producto"])["ipc"].ffill()
    s["precio_real"] = s.precio_mediana * 100 / s["ipc"]   # pesos constantes del último mes del IPC
    s["precio_usd"] = s.precio_mediana / s.get("usd_oficial", np.nan)
    s["base_ipc"] = m.loc[m.ipc.round(6) == 100, "mes"].max()
    return s


# ================================================================ anomalías
def anomalias_series(s: pd.DataFrame, z_lim: float = 3.0) -> pd.DataFrame:
    out = []
    for (prov, prod), g in s.groupby(["provincia", "producto"]):
        g = g.set_index("mes").asfreq("MS")
        lr = np.log(g.precio_real)
        dl = lr.diff()
        # 1) z-score móvil: cambio del mes vs. los 12 meses anteriores (sin mirar el futuro)
        mu = dl.shift(1).rolling(12, min_periods=6).mean()
        sd = dl.shift(1).rolling(12, min_periods=6).std()
        z = (dl - mu) / sd
        # 2) STL: tendencia + estacionalidad + residuo; anomalía = residuo grande
        resid = pd.Series(np.nan, index=g.index)
        lr_i = lr.interpolate(limit=2)
        if lr_i.notna().sum() >= 36:
            v = lr_i.dropna()
            r = STL(v, period=12, robust=True).fit().resid
            mad = (r - r.median()).abs().median() or 1e-9
            resid.loc[r.index] = 0.6745 * (r - r.median()) / mad
        out.append(pd.DataFrame({
            "provincia": prov, "producto": prod, "mes": g.index, "precio_nominal": g.precio_mediana,
            "precio_real": g.precio_real, "var_real_%": dl * 100,
            "var_nominal_%": np.log(g.precio_mediana).diff() * 100,
            "z_movil": z, "z_stl": resid, "inflacion_mm": g.inflacion_mm,
            "var_usd_%": np.log(g.usd_oficial).diff() * 100 if "usd_oficial" in g else np.nan}))
    a = pd.concat(out, ignore_index=True)
    # 3) Isolation Forest sobre varias variables a la vez (aprende qué meses son "raros")
    feats = ["var_real_%", "var_nominal_%", "inflacion_mm", "var_usd_%"]
    nac = a.groupby(["producto", "mes"])["var_real_%"].transform("median")
    a["desvio_vs_nacional"] = a["var_real_%"] - nac
    feats.append("desvio_vs_nacional")
    ok = a[feats].notna().all(axis=1)
    a["iso_score"], a["anom_iforest"] = np.nan, False
    iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=0)
    a.loc[ok, "iso_score"] = -iso.fit(a.loc[ok, feats]).score_samples(a.loc[ok, feats])
    a.loc[ok, "anom_iforest"] = iso.predict(a.loc[ok, feats]) == -1
    a["anom_iforest"] = a["anom_iforest"].astype(bool)
    a["anom_z"] = a["z_movil"].abs() > z_lim
    a["anom_stl"] = a["z_stl"].abs() > 3.5
    a["n_metodos"] = a[["anom_z", "anom_stl", "anom_iforest"]].sum(axis=1)
    a["tipo"] = np.select(
        [a["var_usd_%"] > 20, a["var_nominal_%"] > 0, a["var_nominal_%"] < 0],
        ["salto con devaluación", "aumento", "baja"], "sin cambio")
    a["interpretacion"] = a.apply(_interpretar, axis=1)
    return a


def _interpretar(r) -> str:
    if r.n_metodos == 0:
        return ""
    partes = [f"{r['var_nominal_%']:+.1f}% nominal, {r['var_real_%']:+.1f}% real"]
    if pd.notna(r["var_usd_%"]) and abs(r["var_usd_%"]) > 10:
        partes.append(f"dólar {r['var_usd_%']:+.0f}%")
    if pd.notna(r.desvio_vs_nacional) and abs(r.desvio_vs_nacional) > 5:
        partes.append(f"{r.desvio_vs_nacional:+.1f} pp distinto al resto del país (posible dato local)")
    return "; ".join(partes)


def tabla_anomalias(a: pd.DataFrame, min_metodos: int = 2) -> pd.DataFrame:
    """Meses con salto detectado por al menos `min_metodos` métodos, agrupados a nivel país."""
    t = a[a.n_metodos >= min_metodos]
    return (t.groupby(["mes", "producto"])
             .agg(provincias=("provincia", "nunique"), var_nominal_med=("var_nominal_%", "median"),
                  var_real_med=("var_real_%", "median"), var_usd=("var_usd_%", "first"),
                  inflacion=("inflacion_mm", "first"))
             .reset_index().sort_values(["mes", "producto"]).round(1))


def anomalias_estaciones() -> pd.DataFrame:
    partes = []
    p = C.PROC / "estaciones_vigentes.parquet"
    if p.exists():
        v = pd.read_parquet(p)
        v = v[v.outlier & v.get("reciente", True)].assign(fuente="vigentes")
        partes.append(v)
    p = C.PROC / "estaciones_mes.parquet"
    if p.exists():
        e = pd.read_parquet(p)
        e = e[e.outlier & e.informado].assign(fuente="historico")
        partes.append(e)
    if not partes:
        return pd.DataFrame()
    t = pd.concat(partes, ignore_index=True)
    t["motivo"] = np.select(
        [t.ratio_mediana > 20, t.ratio_mediana > 2, t.ratio_mediana < 0.5],
        ["precio ~100x o más: probable error de carga (coma corrida)",
         "más del doble de la mediana provincial", "menos de la mitad de la mediana provincial"],
        "desvío estadístico fuerte (z robusto > 5)")
    return t


# ================================================================ predicción
H = [1, 2, 3]


def armar_features(s: pd.DataFrame) -> pd.DataFrame:
    d = s[["provincia", "producto", "mes", "precio_mediana", "precio_real", "inflacion_mm", "usd_oficial"]].copy()
    d = d.sort_values(["provincia", "producto", "mes"])
    g = d.groupby(["provincia", "producto"])
    d["lp"] = np.log(d.precio_mediana)
    d["dl1"] = g["lp"].diff()
    for k in (1, 2, 3, 6, 12):
        d[f"dl_lag{k}"] = d.groupby(["provincia", "producto"])["dl1"].shift(k - 1)
    d["dl_3m"] = g["lp"].diff(3)
    d["usd_oficial"] = d.groupby(["provincia", "producto"])["usd_oficial"].ffill()
    d["dusd_1"] = np.log(d.usd_oficial).groupby([d.provincia, d.producto]).diff()
    d["dusd_3"] = np.log(d.usd_oficial).groupby([d.provincia, d.producto]).diff(3)
    # Inflación: el INDEC la publica a mediados del mes siguiente, así que al predecir desde el mes t
    # sólo se conoce la de t-1. Usamos esa (rezagada) para no "ver el futuro" y para poder predecir
    # aunque la del último mes todavía no esté publicada.
    inf_lag = g["inflacion_mm"].shift(1)
    inf_lag = inf_lag.groupby([d.provincia, d.producto]).ffill()
    d["inf_1"] = inf_lag / 100
    d["inf_3"] = inf_lag.groupby([d.provincia, d.producto]).transform(lambda x: x.rolling(3, min_periods=1).mean()) / 100
    nac = d.groupby(["producto", "mes"])["lp"].transform("median")
    d["gap_nacional"] = d.lp - nac
    # brecha entre el precio real actual y su promedio de 12 meses (los precios "atrasados" tienden a corregir)
    d["gap_real_12"] = np.log(d.precio_real) - g["precio_real"].transform(lambda x: np.log(x).rolling(12).mean())
    d["mes_del_anio"] = d.mes.dt.month
    for h in H:
        d[f"y_h{h}"] = g["lp"].shift(-h) - d.lp
    return d


FEATS = ["dl_lag1", "dl_lag2", "dl_lag3", "dl_lag6", "dl_lag12", "dl_3m", "dusd_1", "dusd_3",
         "inf_1", "inf_3", "gap_nacional", "gap_real_12", "mes_del_anio"]


def modelos():
    return {
        "ingenuo (último precio)": None,
        "ingenuo + inflación": "inflacion",
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gradient boosting": HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
                                                           min_samples_leaf=30, random_state=0),
    }


def _predecir(nombre, modelo, Xtr, ytr, Xte):
    if modelo is None:
        return np.zeros(len(Xte))
    if modelo == "inflacion":
        return Xte["inf_3"].to_numpy() * Xte["_h"].to_numpy()
    modelo.fit(Xtr[FEATS], ytr)
    return modelo.predict(Xte[FEATS])


def validar(d: pd.DataFrame, n_origenes: int = 24, paso: int = 2) -> pd.DataFrame:
    """Validación temporal "walk-forward": para cada mes de origen entreno SOLO con el pasado
    (cuyo objetivo ya era conocido en ese momento) y predigo 1, 2 y 3 meses adelante."""
    meses = sorted(d.mes.unique())
    origenes = meses[-(n_origenes + max(H)):-max(H):paso]
    filas = []
    for h in H:
        y = f"y_h{h}"
        base = d.dropna(subset=FEATS + [y]).copy()
        base["_h"] = h
        for o in origenes:
            tr = base[base.mes <= pd.Timestamp(o) - pd.DateOffset(months=h)]
            te = base[base.mes == o]
            if len(tr) < 200 or te.empty:
                continue
            for nombre, m in modelos().items():
                pred = _predecir(nombre, m, tr, tr[y], te)
                err = np.exp(pred) / np.exp(te[y].to_numpy()) - 1
                filas += [{"h": h, "origen": o, "modelo": nombre, "provincia": p, "producto": pr,
                           "error_rel": e, "y_real": yr, "y_pred": yp, "inflacion_ref": inf,
                           "precio_real_h": pt * np.exp(yr), "precio_pred_h": pt * np.exp(yp)}
                          for p, pr, e, yr, yp, inf, pt in zip(te.provincia, te.producto, err, te[y], pred,
                                                               te.inf_3, te.precio_mediana)]
    return pd.DataFrame(filas)


def _r2(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    sst = ((real - real.mean()) ** 2).sum()
    return 1 - ((real - pred) ** 2).sum() / sst if sst > 0 else np.nan


def resumen_validacion(v: pd.DataFrame, d: pd.DataFrame | None = None) -> pd.DataFrame:
    """Métricas por horizonte y modelo.

    - MAPE_pct: error porcentual medio del precio del litro.
    - MAE_pesos / RMSE_pesos: error en $ por litro (RMSE castiga más los errores grandes).
    - R2_cambio: R² sobre el CAMBIO porcentual del precio (lo difícil de predecir). El ingenuo da ~0 o negativo.
    - R2_nivel: R² sobre el precio en $; es casi 1 para cualquier modelo porque los precios de 2017 y 2026
      son muy distintos: se informa pero NO sirve para comparar modelos.
    - acierto_direccion_real: ¿acierta si el precio sube más o menos que la inflación? (como el ejercicio de bitcoin)
    """
    filas = []
    for (h, m), g in v.groupby(["h", "modelo"]):
        err_p = g.precio_pred_h - g.precio_real_h
        ref = g.inflacion_ref * g.h
        ok = (np.sign(g.y_pred - ref) == np.sign(g.y_real - ref))
        filas.append({"h": h, "modelo": m, "MAPE_pct": (g.error_rel.abs() * 100).mean(),
                      "mediana_err_pct": (g.error_rel.abs() * 100).median(),
                      "MAE_pesos": err_p.abs().mean(), "RMSE_pesos": np.sqrt((err_p ** 2).mean()),
                      "R2_cambio": _r2(g.y_real, g.y_pred), "R2_nivel": _r2(g.precio_real_h, g.precio_pred_h),
                      "acierto_direccion_real": np.nan if m.startswith("ingenuo") else ok.mean(), "n": len(g)})
    r = pd.DataFrame(filas)
    ing = r[r.modelo == "ingenuo (último precio)"].set_index("h")["MAPE_pct"]
    r["mejora_vs_ingenuo_%"] = (1 - r.MAPE_pct / r.h.map(ing)) * 100
    return r.round(3)


def pronosticar(d: pd.DataFrame, v: pd.DataFrame, resumen: pd.DataFrame) -> pd.DataFrame:
    filas = []
    ultimo = d.mes.max()
    actual = d[d.mes == ultimo]
    for h in H:
        rh = resumen[resumen.h == h].sort_values("MAPE_pct")
        mejor = rh.iloc[0].modelo
        y = f"y_h{h}"
        tr = d.dropna(subset=FEATS + [y]).assign(_h=h)
        te = actual.dropna(subset=FEATS).assign(_h=h)
        if te.empty:
            print(f"   h={h}: sin filas con todas las variables en {ultimo:%Y-%m}; no se pronostica")
            continue
        m = modelos()[mejor]
        pred = _predecir(mejor, m, tr, tr[y], te)
        res = v[(v.h == h) & (v.modelo == mejor)]
        lo, hi = np.log1p(res.error_rel).quantile([0.1, 0.9]) if len(res) else (0, 0)
        for (_, r), yp in zip(te.iterrows(), pred):
            p = r.precio_mediana * np.exp(yp)
            # el error_rel es pred/real-1 => real = pred/(1+err): el intervalo se invierte
            filas.append({"provincia": r.provincia, "producto": r.producto, "mes_origen": ultimo, "h": h,
                          "mes_objetivo": ultimo + pd.DateOffset(months=h), "precio_actual": r.precio_mediana,
                          "precio_pred": p, "precio_p10": p / np.exp(hi), "precio_p90": p / np.exp(lo),
                          "modelo": mejor})
    return pd.DataFrame(filas)


def main(args: list[str]) -> int:
    s = cargar_serie()
    s.to_parquet(C.PROC / "serie_real.parquet", index=False)
    print(f"Serie: {s.provincia.nunique()} provincias, {s.producto.nunique()} productos, "
          f"{s.mes.min():%Y-%m} a {s.mes.max():%Y-%m}")
    a = anomalias_series(s)
    a.to_parquet(C.PROC / "anomalias_series.parquet", index=False)
    t = tabla_anomalias(a)
    t.to_csv(C.REPORTES / "tabla_anomalias.csv", index=False)
    print(f"Anomalías en series: {int((a.n_metodos >= 2).sum())} provincia-mes (>=2 métodos)")
    e = anomalias_estaciones()
    if len(e):
        e.to_parquet(C.PROC / "anomalias_estaciones.parquet", index=False)
        print(f"Estaciones fuera de rango: {len(e):,} registros")
    d = armar_features(s)
    v = validar(d)
    r = resumen_validacion(v)
    r.to_csv(C.REPORTES / "validacion_modelos.csv", index=False)
    print("\nValidación temporal (MAPE % = error porcentual medio; menor es mejor):")
    print(r.to_string(index=False))
    for h in H:
        rh = r[r.h == h].set_index("modelo")
        gana = rh.MAPE_pct.idxmin()
        if gana.startswith("ingenuo (último"):
            print(f"  h={h}: NINGÚN modelo le gana al ingenuo. Se usa el último precio.")
        else:
            print(f"  h={h}: gana '{gana}' ({rh.loc[gana, 'mejora_vs_ingenuo_%']:.0f}% menos error que el ingenuo)")
    pron = pronosticar(d, v, r)
    pron.to_parquet(C.PROC / "pronosticos.parquet", index=False)
    (C.REPORTES / "resumen_analisis.json").write_text(json.dumps({
        "ultimo_mes": str(s.mes.max().date()), "modelos_elegidos": pron.groupby("h").modelo.first().to_dict()},
        indent=2, ensure_ascii=False))
    print(f"\nPronósticos guardados: {len(pron)} filas")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
