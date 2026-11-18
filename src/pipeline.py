"""Corre todo el pipeline de datos en orden: descarga -> procesamiento -> análisis.

    python -m src.pipeline              # todo
    python -m src.pipeline --sin-descarga   # reprocesa lo que ya está en data/raw
    python -m src.pipeline --muestra    # procesa sólo una parte del histórico (prueba rápida)
    python -m src.pipeline --solo-notebook  # sólo vuelve a ejecutar el notebook con lo ya procesado
"""
import sys
import time

from . import analisis, config as C, consumos, descargar, etiqueta, procesar


def ejecutar_notebook() -> None:
    """Corre el notebook de punta a punta y lo guarda CON las salidas (gráficos y tablas), así quien lo abre
    ve los resultados sin tener que correr nada. También deja una copia en HTML en reportes/."""
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    ruta = C.RAIZ / "notebooks" / "TP_costo_viaje_combustible.ipynb"
    nb = nbformat.read(ruta, as_version=4)
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(ruta.parent)}}).execute()
        print("   notebook ejecutado sin errores")
    except Exception as e:  # se guarda igual, con el error visible en la celda
        print(f"   OJO: una celda falló: {str(e)[:300]}")
    nbformat.write(nb, ruta)
    print(f"   guardado con resultados: {ruta.relative_to(C.RAIZ)}")
    try:
        html, _ = HTMLExporter().from_notebook_node(nb)
        (C.REPORTES / "TP_costo_viaje_combustible.html").write_text(html, encoding="utf-8")
        print("   copia HTML: reportes/TP_costo_viaje_combustible.html")
    except Exception as e:  # la copia HTML es un extra
        print(f"   no se pudo generar la copia HTML: {str(e)[:200]}")


def main(args):
    t0 = time.time()
    if "--solo-notebook" in args:
        ejecutar_notebook()
        return 0
    if "--sin-descarga" not in args:
        print("\n######## 1/4 DESCARGA ########")
        descargar.main(["todo"])
    print("\n######## 2/4 PROCESAMIENTO Y CALIDAD ########")
    procesar.main(["--muestra"] if "--muestra" in args else [])
    if etiqueta.XLSX.exists():
        print("\n######## Base oficial de consumos (LCA) ########")
        etiqueta.construir()
    print("\n######## 3/4 CONSUMOS: calibración del modelo 100 vs 130 km/h ########")
    p = consumos.calibrar()
    v = consumos.validar_modelo()
    v.to_csv(C.REPORTES / "validacion_modelo_velocidad.csv", index=False)
    print(f"k (consumo a 100 / extraurbano): mediana {p.k_med:.2f} (rango {p.k_p25:.2f}-{p.k_p75:.2f}, n={p.n_k})")
    print(f"f (parte aerodinámica a 100 km/h): mediana {p.f_med:.2f} (rango {p.f_p25:.2f}-{p.f_p75:.2f}, n={p.n_f})")
    print(f"=> a 130 km/h se gasta ~{((1 - p.f_med) + p.f_med * 1.69 - 1) * 100:.0f}% más que a 100 km/h")
    print(v.to_string(index=False))
    print("\n######## 4/4 SERIES, ANOMALÍAS Y PREDICCIÓN ########")
    if (C.PROC / "precios_mensuales.parquet").exists():
        analisis.main([])
    else:
        print("Falta el histórico procesado: no se puede correr el análisis.")
        return 1
    print("\n######## 5/5 NOTEBOOK: ejecutarlo y guardarlo con resultados (+ copia HTML) ########")
    ejecutar_notebook()
    print(f"\nListo en {(time.time() - t0) / 60:.1f} minutos. Revisá la carpeta reportes/.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
