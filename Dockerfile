# Basisimage
FROM python:3.9-slim

# Installiere systemabhängige Pakete
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Erstelle Arbeitsverzeichnis
WORKDIR /app

# Kopiere alle Dateien in das Arbeitsverzeichnis im Container
COPY . /app

# Installiere Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Definiere den Startbefehl
CMD ["python", "notebook.py"]
