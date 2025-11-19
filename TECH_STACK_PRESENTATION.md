# 🛠️ TECH STACK - FOR PRESENTATION

## Complete Technology Stack Overview

---

## 📊 Visual Stack Layers (Copy to Slide)

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│  Streamlit Dashboard  │  FastAPI Swagger UI  │  Grafana         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        API LAYER                                │
│              FastAPI (REST API) - Port 8000                     │
│        Health │ Predict │ Analytics │ Drift │ Metrics          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ML MODEL LAYER                              │
│  ANN  │  GRU  │  LSTM  │  Transformer  │  ENSEMBLE             │
│           TensorFlow 2.20 + Keras 3.12                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING                              │
│  Pandas │ NumPy │ scikit-learn │ StandardScaler                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   STORAGE & PERSISTENCE                         │
│  PostgreSQL 16  │  Google Cloud Storage  │  MLflow Storage     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   MONITORING & OBSERVABILITY                    │
│  Prometheus  │  Grafana  │  AlertManager  │  Evidently AI      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MLOps & CI/CD                              │
│  DVC  │  MLflow  │  GitHub Actions  │  Docker  │  Docker Compose│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                               │
│        Docker Containers  │  Google Cloud Platform              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Tech Stack by Category (For Slides)

### 1. **Machine Learning & AI**
| Technology | Version | Purpose |
|------------|---------|---------|
| **TensorFlow** | 2.20 | Deep learning framework |
| **Keras** | 3.12 | High-level neural network API |
| **scikit-learn** | Latest | Preprocessing, scaling, metrics |
| **Pandas** | 2.3.3 | Data manipulation |
| **NumPy** | 2.3.4 | Numerical computing |

**Why?** Industry-standard ML stack with strong community support

---

### 2. **API & Web Framework**
| Technology | Version | Purpose |
|------------|---------|---------|
| **FastAPI** | 0.121.2 | REST API framework |
| **Uvicorn** | Latest | ASGI server |
| **Pydantic** | Latest | Data validation |
| **Streamlit** | 1.40.2 | Interactive dashboard |

**Why?** FastAPI = Fast, auto-documented, type-safe APIs

---

### 3. **MLOps & Experiment Tracking**
| Technology | Version | Purpose |
|------------|---------|---------|
| **MLflow** | 3.6.0 | Experiment tracking, model registry |
| **DVC** | 3.59.2 | Data version control |
| **GitHub Actions** | - | CI/CD automation |
| **Evidently AI** | 0.7.16 | Data drift detection |

**Why?** Complete MLOps pipeline for reproducibility

---

### 4. **Monitoring & Observability**
| Technology | Version | Purpose |
|------------|---------|---------|
| **Prometheus** | 2.54 | Metrics collection |
| **Grafana** | Latest | Visualization & dashboards |
| **AlertManager** | Latest | Alert routing & notifications |

**Why?** Industry-standard monitoring stack (used by Google, Netflix)

---

### 5. **Database & Storage**
| Technology | Version | Purpose |
|------------|---------|---------|
| **PostgreSQL** | 16 | Prediction logging, analytics |
| **Google Cloud Storage** | - | Artifact storage (DVC backend) |
| **SQLite** | - | MLflow metadata |

**Why?** Reliable, ACID-compliant storage

---

### 6. **DevOps & Infrastructure**
| Technology | Version | Purpose |
|------------|---------|---------|
| **Docker** | Latest | Containerization |
| **Docker Compose** | Latest | Multi-container orchestration |
| **Google Cloud Platform** | - | Cloud hosting |
| **GitHub** | - | Version control |

**Why?** Reproducible, portable deployments

---

### 7. **Testing & Quality**
| Technology | Version | Purpose |
|------------|---------|---------|
| **pytest** | Latest | Testing framework |
| **pytest-cov** | Latest | Code coverage |
| **pytest-asyncio** | Latest | Async testing |

**Why?** Comprehensive test coverage (122 tests, 84%)

---

### 8. **Languages**
| Language | Version | Usage |
|----------|---------|-------|
| **Python** | 3.11+ | Main language (100%) |
| **SQL** | - | Database queries |
| **YAML** | - | Configuration files |
| **Bash** | - | Automation scripts |

---

## 🎨 Visual Tech Stack Diagram (For PowerPoint/Google Slides)

### Option 1: Circular Architecture

```
                    ┌─────────────┐
                    │   FastAPI   │
                    │  (Port 8000)│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │PostgreSQL│       │  Models │       │Prometheus│
   │(Logging) │       │(Predict)│       │(Metrics) │
   └─────────┘       └────┬────┘       └────┬────┘
                           │                  │
                    ┌──────┴──────┐      ┌───▼────┐
                    │   MLflow    │      │Grafana │
                    │  (Tracking) │      │(Dashbd)│
                    └──────┬──────┘      └────────┘
                           │
                      ┌────▼────┐
                      │   DVC   │
                      │  (Data) │
                      └─────────┘
```

### Option 2: Layer-Based Stack

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Layer 1: USER INTERFACE                    ┃
┃  Streamlit │ FastAPI Docs │ Grafana         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                    ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Layer 2: API LAYER                         ┃
┃  FastAPI 0.121 │ Uvicorn │ Pydantic         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                    ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Layer 3: ML MODELS                         ┃
┃  TensorFlow 2.20 │ Keras 3.12               ┃
┃  ANN │ GRU │ LSTM │ Transformer │ Ensemble  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                    ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Layer 4: DATA PROCESSING                   ┃
┃  Pandas 2.3 │ NumPy 2.3 │ scikit-learn      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                    ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Layer 5: STORAGE                           ┃
┃  PostgreSQL 16 │ GCS │ MLflow DB            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                    ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Layer 6: MONITORING                        ┃
┃  Prometheus │ Grafana │ AlertManager        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                    ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Layer 7: MLOps & CI/CD                     ┃
┃  DVC 3.59 │ MLflow 3.6 │ GitHub Actions     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                    ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Layer 8: INFRASTRUCTURE                    ┃
┃  Docker │ Docker Compose │ GCP               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## 📋 Complete Tech Stack Table (For Appendix Slide)

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.11+ | Main programming language |
| **ML Framework** | TensorFlow | 2.20 | Deep learning models |
| **ML API** | Keras | 3.12 | High-level neural network API |
| **Data Processing** | Pandas | 2.3.3 | Data manipulation |
| **Numerical** | NumPy | 2.3.4 | Array operations |
| **Preprocessing** | scikit-learn | Latest | Scaling, metrics |
| **API Framework** | FastAPI | 0.121.2 | REST API server |
| **ASGI Server** | Uvicorn | Latest | Async web server |
| **Dashboard** | Streamlit | 1.40.2 | Interactive UI |
| **Experiment Tracking** | MLflow | 3.6.0 | Model versioning |
| **Data Versioning** | DVC | 3.59.2 | Dataset version control |
| **Drift Detection** | Evidently AI | 0.7.16 | Data drift monitoring |
| **Monitoring** | Prometheus | 2.54 | Metrics collection |
| **Visualization** | Grafana | Latest | Dashboards |
| **Alerting** | AlertManager | Latest | Alert management |
| **Database** | PostgreSQL | 16 | Prediction storage |
| **Cloud Storage** | GCS | - | Artifact storage |
| **Containerization** | Docker | Latest | Container runtime |
| **Orchestration** | Docker Compose | Latest | Multi-container apps |
| **CI/CD** | GitHub Actions | - | Automated pipeline |
| **Testing** | pytest | Latest | Unit/integration tests |
| **Coverage** | pytest-cov | Latest | Code coverage |
| **Cloud Platform** | GCP | - | Cloud infrastructure |

---

## 🎯 Key Tech Stack Highlights (For Slide)

### **Production-Grade Stack**
✅ **Industry Standard** - TensorFlow, FastAPI, PostgreSQL
✅ **Complete MLOps** - DVC, MLflow, GitHub Actions
✅ **Enterprise Monitoring** - Prometheus, Grafana (Netflix/Google stack)
✅ **Container-Native** - Docker, Docker Compose
✅ **Cloud-Ready** - GCP deployment

### **Modern Best Practices**
✅ **API-First** - FastAPI with auto-documentation
✅ **Type-Safe** - Pydantic validation
✅ **Observable** - Prometheus metrics + Grafana dashboards
✅ **Reproducible** - DVC for data, MLflow for models
✅ **Tested** - 122 tests with 84% coverage

---

## 🏗️ Architecture Flow Diagram

```
┌──────────────┐
│ Data Sources │
│ (CSV Files)  │
└──────┬───────┘
       │
       │ DVC (Version Control)
       ↓
┌──────────────────────────────────────────────┐
│     Data Processing (Pandas, NumPy)          │
│  Feature Engineering → 68 Features            │
│  Scaling (StandardScaler)                    │
│  Sequence Creation (20 timesteps)            │
└──────┬───────────────────────────────────────┘
       │
       │ GitHub Actions (CI/CD)
       ↓
┌──────────────────────────────────────────────┐
│  Model Training (TensorFlow/Keras)           │
│  ┌─────┐ ┌─────┐ ┌──────┐ ┌─────────────┐  │
│  │ ANN │ │ GRU │ │ LSTM │ │ Transformer │  │
│  └──┬──┘ └──┬──┘ └───┬──┘ └──────┬──────┘  │
│     └────────┴────────┴───────────┘          │
│              Ensemble                         │
└──────┬───────────────────────────────────────┘
       │
       │ MLflow (Experiment Tracking)
       ↓
┌──────────────────────────────────────────────┐
│  FastAPI Service (Port 8000)                 │
│  ├─ POST /predict                            │
│  ├─ GET  /analytics/recent                   │
│  ├─ GET  /drift/status                       │
│  └─ GET  /metrics (Prometheus)               │
└──────┬───────────────────────────────────────┘
       │
       ├──────────────┬──────────────┬─────────┐
       ↓              ↓              ↓         ↓
┌────────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐
│ PostgreSQL │ │Prometheus │ │Evidently │ │Streamlit │
│ (Logging)  │ │ (Metrics) │ │  (Drift) │ │   (UI)   │
└────────────┘ └─────┬─────┘ └──────────┘ └──────────┘
                     ↓
              ┌──────────┐
              │ Grafana  │
              │(Dashboard)│
              └──────────┘
```

---

## 💡 Why This Tech Stack? (Talking Points)

### **1. TensorFlow + Keras**
> "Industry standard for deep learning. Used by Google, Uber, Airbnb. Supports both research and production."

### **2. FastAPI**
> "Modern Python web framework. Auto-generates API documentation. 3x faster than Flask. Type-safe."

### **3. MLflow + DVC**
> "Complete MLOps solution. DVC for data versioning (Git for datasets). MLflow for experiment tracking (like GitHub for models)."

### **4. Prometheus + Grafana**
> "Netflix and Google's monitoring stack. Prometheus collects metrics. Grafana visualizes. AlertManager notifies."

### **5. Docker + Docker Compose**
> "Containerization ensures 'works on my machine' = works everywhere. Easy deployment to any cloud."

### **6. PostgreSQL**
> "Battle-tested relational database. ACID compliance. Perfect for structured prediction logs."

### **7. GitHub Actions**
> "Free CI/CD for public repos. Automated training on every push. Parallel execution for 75% speedup."

---

## 📊 Tech Stack by Numbers

```
Lines of Code:     ~5,000
Dependencies:      197 packages
Docker Containers: 6 services
API Endpoints:     10+ endpoints
Models:            5 (4 base + 1 ensemble)
Tests:             122 (84% coverage)
Alerts:            21 configured
Monitoring Metrics: 169 tracked
```

---

## 🎨 Color-Coded Stack (For Colorful Slide)

### 🟦 **Data Layer (Blue)**
- Pandas, NumPy, scikit-learn
- PostgreSQL, Google Cloud Storage

### 🟩 **ML Layer (Green)**
- TensorFlow, Keras
- ANN, GRU, LSTM, Transformer

### 🟨 **API Layer (Yellow)**
- FastAPI, Uvicorn
- Streamlit

### 🟧 **MLOps Layer (Orange)**
- DVC, MLflow, GitHub Actions
- Evidently AI

### 🟥 **Monitoring Layer (Red)**
- Prometheus, Grafana, AlertManager

### ⬛ **Infrastructure Layer (Black)**
- Docker, Docker Compose, GCP

---

## 🔄 Data Flow Through Stack

```
1. CSV File (DVC versioned)
      ↓
2. Pandas/NumPy (processing)
      ↓
3. TensorFlow/Keras (training)
      ↓
4. MLflow (logging experiments)
      ↓
5. Docker (containerization)
      ↓
6. FastAPI (serving predictions)
      ↓
7. PostgreSQL (storing results)
      ↓
8. Prometheus (collecting metrics)
      ↓
9. Grafana (visualizing performance)
```

---

## 📦 Docker Services Architecture

```
Docker Compose Stack:
├── postgres-timeseries    (Port 5432)
├── time-series-api        (Port 8000)
├── mlflow-server          (Port 5001)
├── prometheus             (Port 9090)
├── grafana                (Port 3000)
└── alertmanager           (Port 9093)

Total: 6 containerized services
```

---

## 🎯 Tech Stack Slide Template

**Slide Title:** "Production-Ready Tech Stack"

**Content:**

| Layer | Technologies | Purpose |
|-------|--------------|---------|
| 🎯 **ML Models** | TensorFlow 2.20, Keras 3.12 | Deep Learning |
| 🚀 **API** | FastAPI 0.121, Uvicorn | REST API Serving |
| 📊 **Data** | Pandas 2.3, NumPy 2.3 | Processing |
| 🔄 **MLOps** | DVC 3.59, MLflow 3.6 | Versioning |
| 📈 **Monitoring** | Prometheus, Grafana | Observability |
| 💾 **Storage** | PostgreSQL 16, GCS | Persistence |
| 🐳 **Infrastructure** | Docker, GCP | Deployment |
| 🧪 **Testing** | pytest, 122 tests | Quality Assurance |

**Key Point:**
"Industry-standard stack used by companies like Netflix, Google, and Uber"

---

## 🎤 Talking Points for Tech Stack Slide

**Opening:**
> "This project uses a production-grade tech stack that you'd find at companies like Netflix or Google."

**Key Points:**
1. **Modern ML:** "TensorFlow and Keras for state-of-the-art deep learning"
2. **Fast API:** "FastAPI - one of the fastest Python web frameworks, auto-generates documentation"
3. **Complete MLOps:** "DVC and MLflow ensure every experiment and dataset is tracked and reproducible"
4. **Enterprise Monitoring:** "Prometheus and Grafana - the same stack Netflix uses to monitor their services"
5. **Container-Native:** "Everything runs in Docker containers - deploy anywhere in minutes"

**Closing:**
> "This isn't just a school project - it's built with the same tools and practices used in production ML systems at tech companies."

---

## 📄 Alternative: Simple Icon-Based Stack

```
┌────────────────────────────────────────────────┐
│            TECH STACK OVERVIEW                 │
├────────────────────────────────────────────────┤
│                                                │
│  🐍 Python 3.11+                               │
│  🧠 TensorFlow 2.20 + Keras 3.12               │
│  ⚡ FastAPI 0.121 + Uvicorn                    │
│  📊 Pandas 2.3 + NumPy 2.3 + scikit-learn      │
│  🔄 DVC 3.59 + MLflow 3.6                      │
│  📈 Prometheus + Grafana + AlertManager        │
│  💾 PostgreSQL 16 + Google Cloud Storage       │
│  🐳 Docker + Docker Compose                    │
│  🤖 GitHub Actions (CI/CD)                     │
│  🧪 pytest (122 tests, 84% coverage)           │
│                                                │
└────────────────────────────────────────────────┘
```

---

**🎯 Bereit für deine Präsentation!**

Du kannst jetzt:
1. Eine dieser Visualisierungen in deine Slides kopieren
2. Die Tabellen als Referenz verwenden
3. Die "Talking Points" für deine Erklärungen nutzen

Viel Erfolg! 🚀
