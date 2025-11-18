
```
tests/
├── __init__.py
├── conftest.py              # Pytest Fixtures und Konfiguration
├── test_api.py              # API Endpoint Tests
├── test_validation.py       # Input Validierung Tests
├── test_models.py           # Modell Prediction Tests
└── test_scaling.py          # Data Scaling Tests
```

## Tests ausführen

### Alle Tests ausführen
```bash
pytest
```

### Spezifische Test-Datei ausführen
```bash
pytest tests/test_api.py
pytest tests/test_validation.py
```

### Tests mit Coverage
```bash
pytest --cov=service --cov-report=html
```

### Spezifische Test-Klasse oder Funktion
```bash
pytest tests/test_api.py::TestHealthEndpoint
pytest tests/test_api.py::TestHealthEndpoint::test_health_endpoint_returns_200
```

### Tests mit bestimmten Markern
```bash
pytest -m api          # Nur API Tests
pytest -m validation   # Nur Validation Tests
pytest -m models       # Nur Model Tests
```

### Verbose Output
```bash
pytest -v              # Verbose
pytest -vv             # Extra verbose
```
run all tests with verbose output
```bash
pytest tests/ -v --tb=short 2>&1 | head -100
```

### Tests parallel ausführen (schneller)
```bash
pytest -n auto         # Nutzt alle verfügbaren CPU Cores
```

## Coverage Reports

Nach dem Ausführen mit `--cov` werden Reports generiert:

- **Terminal:** Direkte Ausgabe
- **HTML:** `htmlcov/index.html` - Öffne im Browser
- **XML:** `coverage.xml` - Für CI/CD Integration

```bash
# Coverage Report anzeigen
open htmlcov/index.html
```

## ✅ Test-Kategorien

### 1. **API Tests** (`test_api.py`)
- Health Endpoint
- Prediction Endpoint
- Analytics Endpoints
- Error Handling
- Input Validation

**Beispiel:**
```bash
pytest tests/test_api.py -v
```

### 2. **Validation Tests** (`test_validation.py`)
- Sequence Validation
- Array Conversion
- NaN/Inf Detection
- Size Constraints
- Value Range Checks

**Beispiel:**
```bash
pytest tests/test_validation.py::TestSequenceValidation -v
```

### 3. **Model Tests** (`test_models.py`)
- Prediction Outputs
- Model Consistency
- Edge Cases
- Ensemble Logic

**Beispiel:**
```bash
pytest tests/test_models.py -v
```

### 4. **Scaling Tests** (`test_scaling.py`)
- Scaler Loading
- Feature Scaling
- Target Scaling
- Inverse Transforms
- Edge Cases

**Beispiel:**
```bash
pytest tests/test_scaling.py -v
```

## Test Coverage Ziele

**Aktuelles Ziel:** ≥70% Coverage

```bash
pytest --cov=service --cov-fail-under=70
```

##  Debugging Tests

### Test mit Print Statements
```bash
pytest -s  # Zeigt print() Ausgaben
```


Get detailed coverage report
```bash
pytest tests/ --cov=service --cov-report=term-missing --tb=no -q 2>&1 | grep -A 20 "service/"
```

### Test mit Breakpoint
```python
def test_something():
    import pdb; pdb.set_trace()  # Breakpoint
    # your test code
```

### Test mit detailliertem Output
```bash
pytest --tb=long  # Lange Traceback-Ausgabe
pytest --lf       # Nur fehlgeschlagene Tests vom letzten Lauf
pytest --ff       # Erst fehlgeschlagene, dann alle anderen
```

## Neue Tests hinzufügen

### 1. Test-Datei erstellen
```python
# tests/test_myfeature.py
import pytest

def test_my_feature():
    assert True
```

### 2. Fixtures nutzen
```python
def test_with_fixture(client, valid_sequence):
    # client und valid_sequence sind definiert in conftest.py
    response = client.get("/")
    assert response.status_code == 200
```

### 3. Test markieren
```python
@pytest.mark.slow
@pytest.mark.integration
def test_slow_integration():
    pass
```

## Troubleshooting

### Problem: "ModuleNotFoundError"
**Lösung:** Installiere Dependencies
```bash
pip install -r requirements.txt
```

### Problem: "No tests collected"
**Lösung:** Stelle sicher, dass du im Root-Verzeichnis bist
```bash
pwd  # Should be in project root
pytest tests/
```

### Problem: Tests schlagen fehl wegen fehlender Modelle
**Lösung:** Stelle sicher, dass Modelle existieren
```bash
ls models/*.keras
ls artifacts/ensemble/*.pkl
```

Wenn fehlt, führe DVC aus:
```bash
dvc pull
```

## CI/CD Integration

Tests werden automatisch in GitHub Actions ausgeführt.

**Siehe:** `.github/workflows/test.yml`
 Ressourcen

- [Pytest Dokumentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Coverage.py](https://coverage.readthedocs.io/)

