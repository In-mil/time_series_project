# Streamlit Dashboard für Cryptocurrency Price Prediction

Eine interaktive Web-Anwendung zur Visualisierung und Interaktion mit dem Ensemble-ML-Modell für Kryptowährungs-Preisvorhersagen.

## Features

### 🎯 Prediction Tab
- **Drei Eingabemethoden:**
  - Random Sample: Generiert realistische Test-Daten
  - Manual Input: Manuelle Eingabe von Zeitreihendaten
  - CSV Upload: Laden von Daten aus CSV-Dateien
- **Echtzeit-Vorhersagen** mit allen Modellen (ANN, GRU, LSTM, Transformer)
- **Ensemble-Vorhersage** als gewichteter Durchschnitt
- **Konfidenz-Metriken** basierend auf Modell-Übereinstimmung

### 📈 Model Comparison Tab
- **Radar Chart**: Normalisierte Darstellung aller Modell-Vorhersagen
- **Box Plot**: Verteilung der Vorhersagen
- **Detaillierte Tabelle**: Abweichungen vom Ensemble-Durchschnitt
- **Visuelle Analyse** der Modell-Performance

### 📊 Analytics Tab
- **Performance-Metriken** für jedes Modell
- **Durchschnittliche Vorhersagen** über alle Anfragen
- **Anzahl der Vorhersagen** pro Modell
- **Vergleichende Visualisierungen**

### 🔍 Recent Predictions Tab
- **Historische Vorhersagen** aus der Datenbank
- **Timeline-Visualisierung** der Ensemble-Vorhersagen
- **Detaillierte Tabelle** mit allen Modell-Vorhersagen
- **Filterbare Anzahl** von Einträgen (10-200)

## Voraussetzungen

1. **FastAPI Service** muss laufen (Standard: `http://localhost:8000`)
2. **Python 3.8+**
3. **Installierte Dependencies** aus `requirements.txt`

## Installation

```bash
# Dependencies installieren
pip install -r requirements.txt

# Oder nur Streamlit
pip install streamlit==1.40.2
```

## Verwendung

### Methode 1: Mit Shell-Script

```bash
./run_streamlit.sh
```

### Methode 2: Direkt mit Streamlit

```bash
streamlit run streamlit_app.py
```

### Methode 3: Mit benutzerdefinierten Einstellungen

```bash
streamlit run streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --browser.gatherUsageStats=false
```

## Zugriff

Nach dem Start öffne deinen Browser und gehe zu:
- **Lokal**: http://localhost:8501
- **Netzwerk**: http://<deine-ip>:8501 (wenn mit 0.0.0.0 gestartet)

## Architektur

```
┌─────────────────┐
│  Streamlit UI   │
│   (Port 8501)   │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│  FastAPI        │
│   (Port 8000)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Models      │
│  + Database     │
└─────────────────┘
```

## API-Endpunkte

Die App nutzt folgende FastAPI-Endpunkte:

- `GET /` - Health Check
- `POST /predict` - Vorhersage erstellen
- `GET /analytics/recent` - Letzte Vorhersagen abrufen
- `GET /analytics/performance` - Performance-Metriken abrufen

## Konfiguration

### API URL ändern
In der Sidebar der App kannst du die API-URL anpassen, falls dein FastAPI-Service auf einem anderen Host/Port läuft.

### Port ändern
Im Script `run_streamlit.sh` oder via Kommandozeile:
```bash
streamlit run streamlit_app.py --server.port=8502
```

## Datenformat

Für manuelle Eingaben oder CSV-Upload:

- **20 Zeitschritte** (Look-back Window)
- **16 Features** pro Zeitschritt (Standard):
  1. Open
  2. High
  3. Low
  4. Close
  5. Volume
  6. MA_5
  7. MA_10
  8. MA_20
  9. EMA_12
  10. EMA_26
  11. RSI
  12. MACD
  13. MACD_Signal
  14. BB_Upper
  15. BB_Middle
  16. BB_Lower

## Troubleshooting

### App startet nicht
```bash
# Prüfe ob Streamlit installiert ist
pip show streamlit

# Neuinstallation
pip install --upgrade streamlit==1.40.2
```

### API Connection Error
```bash
# Prüfe ob FastAPI läuft
curl http://localhost:8000/

# FastAPI starten (falls nicht aktiv)
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

### Keine Predictions in Analytics
- Database muss aktiviert sein im FastAPI-Service
- Mindestens eine Vorhersage muss gemacht worden sein

## Development

### Lokale Anpassungen

Die App ist modular aufgebaut. Hauptbereiche:

1. **Styling** - CSS im `st.markdown()` am Anfang
2. **Tabs** - Vier Haupt-Tabs für verschiedene Funktionen
3. **API Calls** - Alle API-Anfragen nutzen `requests`
4. **Visualisierungen** - Plotly für interaktive Charts

### Neue Features hinzufügen

```python
# Neuen Tab hinzufügen
tab5 = st.tabs(["New Feature"])

with tab5:
    st.header("My New Feature")
    # Dein Code hier
```

## Performance-Tipps

1. **Caching nutzen**: Streamlit cached automatisch zwischen Reruns
2. **API-Aufrufe minimieren**: Nur bei Bedarf refreshen
3. **Große Datenmengen**: Pagination für history verwenden

## Screenshots

Nach dem Start siehst du:
- 📊 Dashboard mit vier Tabs
- 🎨 Interaktive Plotly-Charts
- 📈 Echtzeit-Vorhersagen
- 📉 Historische Daten-Visualisierung

## Support

Bei Fragen oder Problemen:
1. Prüfe die FastAPI-Logs
2. Prüfe Browser-Console für Frontend-Fehler
3. Stelle sicher, dass alle Dependencies installiert sind

---

**Built with:** Streamlit 1.40.2 | Python 3.8+ | Plotly
