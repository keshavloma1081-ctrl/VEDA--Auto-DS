# 🧠 VEDA - Autonomous Data Science System

> Give VEDA a dataset and a goal in plain English. It does the rest.

Python

LangGraph

License

Status

---

## What is VEDA?

VEDA is an **autonomous data science system** powered by 11 AI agents that collaborate to take a raw dataset from ingestion to a trained model, live dashboard, and professional report — with **zero human intervention**.

**You provide:**

- A CSV file
- A goal in plain English: *"predict whether a customer will churn"*

**VEDA delivers:**

- ✅ A trained, evaluated ML model (AUC, F1, Accuracy)
- ✅ A live Streamlit dashboard with predictions
- ✅ A professional HTML report with executive summary
- ✅ SHAP feature explanations
- ✅ Full MLflow experiment tracking

---

## Demo Results on Titanic Dataset

| Metric | Score |

|--------|-------|

| AUC-ROC | 1.0 |

| F1 Score | 0.9992 |

| Accuracy | 0.9989 |

| Precision | 0.9992 |

| Recall | 0.9992 |

| Model | LightGBM |

| Features | 9 |

| Training time | ~30 seconds |

---

## Quick Start

### 1. Clone the repo

```bash

git clone [https://github.com/keshavloma1081-ctrl/VEDA--Auto-DS.git](https://github.com/keshavloma1081-ctrl/VEDA--Auto-DS.git)

cd VEDA--Auto-DS

```

### 2. Create environment

```bash

conda create -n autods python=3.11 -y

conda activate autods

```

### 3. Install dependencies

```bash

pip install -r requirements.txt

```

### 4. Add your Groq API key

Create a `.env` file in the root folder:

Get a free Groq API key at [console.groq.com]([https://console.groq.com](https://console.groq.com))

### 5. Add your dataset

```bash

cp your_data.csv data/

```

### 6. Set your goal in [main.py](http://main.py)

```python

GOAL = "predict customer churn. target: churned"

DATASET = "data/your_data.csv"

```

### 7. Run VEDA

```bash

python [main.py](http://main.py)

```

### 8. View the live dashboard

```bash

streamlit run outputs/veda_[dashboard.py](http://dashboard.py)

```

---

## How It Works

Input: CSV + Goal in plain English | v MasterPlanner <- Groq LLM classifies task, builds execution plan | v DataIngest <- Loads CSV, Parquet, JSON, Excel, SQL | v EDAAgent <- Statistics, correlations, outliers, imbalance | v CleaningAgent <- Null imputation, deduplication, winsorisation | v FeatureEngineering <- Encoding, scaling, datetime extraction | v ModelSelection <- Benchmarks XGBoost, LightGBM, RF, LR, Baseline | v TrainingAgent <- 5-fold CV, MLflow logging, model saved | v EvaluationAgent <- AUC, F1, Accuracy, Precision, Recall | v Explainability <- SHAP values + LLM plain-English explanation | v DashboardAgent <- 4-tab Streamlit app with live prediction form | v ReportAgent <- HTML report with AI executive summary | v Output: Trained Model + Live Dashboard + HTML Report

---

## Agent Roster

| # | Agent | Role |

|---|-------|------|

| 1 | MasterPlanner | Groq LLM classifies ML task and builds execution plan |

| 2 | DataIngest | Loads any data source, infers schema, profiles data |

| 3 | EDAAgent | Full exploratory analysis with statistics and correlations |

| 4 | CleaningAgent | Automated null imputation, outlier treatment, deduplication |

| 5 | FeatureEngineeringAgent | Encoding, scaling, datetime features |

| 6 | ModelSelectionAgent | Benchmarks 5 model families with 3-fold CV |

| 7 | TrainingAgent | Trains winner with 5-fold CV and MLflow logging |

| 8 | EvaluationAgent | Computes all metrics, confusion matrix, pass/fail signal |

| 9 | ExplainabilityAgent | SHAP feature importance and LLM explanation |

| 10 | DashboardAgent | Auto-generates 4-tab Streamlit app |

| 11 | ReportAgent | Professional HTML report with executive summary |

---

## Tech Stack

| Layer | Technology |

|-------|-----------|

| Orchestration | LangGraph, LangChain |

| LLM | Groq — Llama-3.3-70b — free and fast |

| ML Models | XGBoost, LightGBM, Scikit-learn |

| Experiment Tracking | MLflow |

| Explainability | SHAP |

| Dashboard | Streamlit, Plotly |

| Data Processing | Pandas, NumPy, PyArrow |

| State Management | LangGraph MemorySaver |

---

## Project Structure

VEDA--Auto-DS/ ├── veda/ │ ├── core/ │ │ ├── [state.py](http://state.py) # VEDAState — shared state for all agents │ │ ├── [graph.py](http://graph.py) # LangGraph pipeline definition │ │ └── base_[agent.py](http://agent.py) # BaseAgent class all agents inherit from │ ├── agents/ │ │ ├── core_pipeline/ # 9 core ML agents │ │ ├── dashboards/ # Dashboard generation agent │ │ └── reports/ # Report generation agent │ ├── monitors/ # Health monitor agents (coming soon) │ └── idle_pool/ # Replacement agent pool (coming soon) ├── data/ # Your datasets go here ├── outputs/ # All VEDA outputs saved here ├── [main.py](http://main.py) # Entry point ├── requirements.txt # Dependencies └── .env # API keys (not committed)

---

## Roadmap

- Core 11-agent autonomous pipeline
- Multi-source data ingestion
- Automated EDA and data cleaning
- 5-model benchmarking and selection
- Cross-validated training with MLflow tracking
- SHAP explainability
- 4-tab Streamlit dashboard
- HTML report with AI executive summary
- Self-healing agent health monitors
- Idle agent replacement pool
- PDF report generation
- REST API endpoint
- Docker containerisation

---

## Contributing

Contributions welcome. Please open an issue first to discuss what you would like to change.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/new-agent`
3. Commit your changes: `git commit -m "add new agent"`
4. Push to the branch: `git push origin feature/new-agent`
5. Open a Pull Request

---

## Author

**Keshav Kumar** — Data Scientist & ML Engineer

Delhi NCR, India

GitHub: [keshavloma1081-ctrl]([https://github.com/keshavloma1081-ctrl](https://github.com/keshavloma1081-ctrl))

---

## License

MIT License — free to use, modify, and distribute.

---

*VEDA — Vedic Autonomous Data Science Architecture*