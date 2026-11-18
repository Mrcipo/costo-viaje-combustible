FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=America/Argentina/Buenos_Aires

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# Capa aparte (así agregarla no reinstala todo lo anterior): lector de Excel para la base oficial de consumos
# y mistune 3.0.2 (con 3.1 la exportación del notebook a HTML de nbconvert 7.16.4 falla).
RUN pip install openpyxl==3.1.5 mistune==3.0.2

# El código y los datos NO se copian: se montan como volumen (la carpeta del proyecto),
# así lo que se descarga queda en tu disco y los cambios de código no obligan a reconstruir.
EXPOSE 8501 8888
CMD ["python", "-m", "src.pipeline"]
