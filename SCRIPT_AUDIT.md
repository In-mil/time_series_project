# Script Audit Report - 19.11.2025

## Überprüfte Skripte und gefundene Probleme

### ❌ FEHLER GEFUNDEN

#### 1. `scripts/query_predictions_db.sh`
**Problem:**
- Zeigt skalierte Werte ohne klare Erklärung
- Keine Transformation zurück zu prozentualen Änderungen
- Irreführende Beschriftung "avg_prediction" statt "avg_prediction_scaled"

**Impact:**
- User könnte denken -5.15 ist die prozentuale Änderung
- Tatsächlich ist -5.15 ein skalierter Wert

**Fix:**
- ✅ Ursprüngliches Skript korrigiert: `scripts/query_predictions_db.sh`
- Zeigt jetzt beide Werte: skaliert UND prozentuale Änderung
- Klare Warnungen und Erklärungen hinzugefügt
- Korrekte Spaltenbezeichnungen (_scaled suffix)

**Verwendung:**
```bash
./scripts/query_predictions_db.sh
```

---

### ✅ KORREKT

#### 2. `scripts/mlflow_summary.py`
**Status:** ✅ KORREKT

**Was es zeigt:**
- MAE (Mean Absolute Error) in Prozentpunkten
- Beispiel: MAE = 8.67 bedeutet durchschnittlich 8.67% Abweichung
- Das ist die richtige Interpretation für Fehlermetriken!

**Keine Änderung nötig.**

---

#### 3. `scripts/query_mlflow.py`
**Status:** ✅ KORREKT

**Was es macht:**
- Zeigt MLflow Runs und Metriken
- Korrekte Interpretation der Metriken
- Keine Probleme gefunden

---

#### 4. `fix_missing_files.sh`
**Status:** ✅ KORREKT

**Was es macht:**
- Behebt das DVC dual-tracking Problem
- Pullt Pipeline-Outputs und checkt dann Training-Data aus
- Funktioniert wie erwartet

---

#### 5. `docu/DVC_TROUBLESHOOTING.md`
**Status:** ✅ KORREKT

**Inhalt:**
- Erklärt das DVC Problem korrekt
- Gute Lösungsvorschläge
- Keine Fehler

---

## Zusammenfassung der Fehler

### Hauptfehler:
1. **Falsche Interpretation der Predictions in DB-Query-Skript**
   - Skalierte Werte wurden ohne Transformation gezeigt
   - Keine klare Kennzeichnung als "scaled"

### Was ich initial falsch erklärt habe:
1. Sagte, die Werte wären "Preisänderungen in USD" → **FALSCH**
2. Korrekt: "Prozentuale Preisänderungen (5-Tage-Horizont)" → **RICHTIG**

---

## Empfohlene Maßnahmen

### Sofort:
1. ✅ `query_predictions_db_corrected.sh` verwenden statt alte Version
2. ✅ Alte Version kann gelöscht oder umbenannt werden

### Optional:
1. Alte `query_predictions_db.sh` mit Warnung versehen:
   ```bash
   echo "⚠️  DEPRECATED: Use query_predictions_db_corrected.sh instead"
   ```

---

## Scaler-Werte zur Referenz

**Target Scaler (scaler_y.pkl):**
- Mean: 0.5561%
- Std: 11.3582%

**Transformation Formula:**
```python
percentage_change = (scaled_value * 11.3582) + 0.5561
```

**Beispiele:**
- Scaled: -5.16 → Percentage: -58.01%
- Scaled: +2.42 → Percentage: +28.07%
- Scaled: 0.00 → Percentage: +0.56%

---

## Interpretation Guide

### Database Predictions:
| Wert | Typ | Interpretation |
|------|-----|----------------|
| `prediction_ann` | Skaliert | Muss transformiert werden! |
| `prediction_ann * 11.36 + 0.56` | Prozent | Tatsächliche Vorhersage |

### MLflow Metrics:
| Metrik | Einheit | Interpretation |
|--------|---------|----------------|
| `test_mae` | Percentage Points | Durchschnittlicher Fehler in % |
| `test_mse` | Squared % | Quadratischer Fehler |
| `test_r2` | Ratio | Erklärte Varianz (0-1) |

### Target Variable:
- **Name:** `future_5_close_higher_than_today`
- **Typ:** Prozentuale Änderung
- **Bedeutung:** Um wie viel % ändert sich der Preis in 5 Tagen?
- **Range:** -29.57% bis +49.92% (in den Trainingsdaten)

---

## Datum: 2025-11-19
## Geprüft von: Claude (AI Assistant)
