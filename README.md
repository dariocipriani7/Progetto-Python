# Student Performance Predictor

Progetto di Machine Learning per la classificazione della performance accademica degli studenti in **High Performance** o **Low Performance**, basata su variabili comportamentali e demografiche — senza usare il voto dell'esame come input.

---

## Obiettivo

Predire se uno studente appartiene alla metà superiore o inferiore della distribuzione dei voti, partendo da 19 variabili che descrivono il suo comportamento (frequenza, ore di studio, tutoraggio) e il suo contesto (famiglia, scuola, pari). Il progetto dimostra che i fattori comportamentali sono predittori significativamente più forti di quelli demografici.

---

## Dataset

- **Fonte:** [Student Performance Factors — Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- **Dimensione originale:** 6607 studenti × 20 colonne
- **Dopo pulizia:** 6378 studenti (229 righe rimosse per valori mancanti, 3.5%)
- **Feature:** 7 numeriche + 13 categoriche
- **Target:** `Exam_Score` (usato solo per costruire la variabile target, poi escluso)

### Colonne principali

| Tipo | Variabili |
|---|---|
| Numeriche | `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`, `Exam_Score` |
| Categoriche | `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Gender` e altri |

---

## Struttura del progetto

```
├── data/
│   ├── StudentPerformanceFactors.csv   # dataset originale
│   ├── dataset_pulito.csv              # dopo dropna()
│   ├── dataset_encoded.csv             # dopo Label Encoding
│   ├── X_train.csv / X_test.csv        # split non scalato (per alberi)
│   ├── X_train_scaled.csv / X_test_scaled.csv  # split scalato (per LR)
│   └── y_train.csv / y_test.csv        # etichette
├── grafici/                            # output grafici salvati dai notebook
├── 01_EDA.ipynb                        # Analisi esplorativa
├── 02_Preprocessing.ipynb              # Preprocessing e feature engineering
├── 03_ML.ipynb                         # Addestramento e valutazione modelli
└── README.md
```

---

## Pipeline

### 01 — EDA (Analisi Esplorativa)

Ispezione della struttura del dataset, analisi della qualità dei dati, distribuzione delle variabili e ricerca di pattern.

**Risultati chiave:**
- Unica correlazione forte: `Attendance` ↔ `Exam_Score` (r = **0.580**). Tutte le altre coppie sotto soglia.
- `Exam_Score` ha distribuzione molto concentrata: media 67.24, std **3.89**, range 55–101. Questo rende impossibile usare una soglia fissa per la classificazione.
- Nessuna differenza di genere: Female 67.27 vs Male 67.24 (irrilevante).
- Effetto tutoraggio: da 66.51 (0 sessioni) a 69.09 (5 sessioni), progressione reale ma modesta.
- Valori mancanti: solo 3 colonne interessate (`Parental_Education_Level` 1.36%, `Teacher_Quality` 1.18%, `Distance_from_Home` 1.01%), rimossi con `dropna()`.

### 02 — Preprocessing

Trasformazione del dataset grezzo in input pronto per i modelli ML.

**Passi:**
1. **Label Encoding** — 13 colonne categoriche convertite in interi con `LabelEncoder` (ordine alfabetico).
2. **Creazione del target** — soglia alla mediana di `Exam_Score` (67.0): `>= 67` → 1 (High Performance), `< 67` → 0 (Low Performance). Classi risultanti: 56.6% / 43.4%. Una soglia fissa a 60 avrebbe prodotto 99% High Performance, rendendo il problema banale.
3. **Separazione X / y** — `Exam_Score` escluso da X per evitare data leakage.
4. **Train/Test split** — 80% training (5102), 20% test (1276), con `stratify=y` per mantenere le proporzioni di classe identiche nei due set.
5. **Normalizzazione** — `StandardScaler` fittato solo sul training set, applicato con `transform()` al test (nessun leakage).

### 03 — Machine Learning

Addestramento e confronto di tre modelli con criterio di selezione che penalizza l'overfitting.

**Modelli:**
- `LogisticRegression` (max_iter=1000)
- `DecisionTreeClassifier` (max_depth=5)
- `RandomForestClassifier` (n_estimators=100)

---

## Risultati

### Confronto modelli

| Modello | Acc. Train | Acc. Test | Gap | Overfitting | Score corretto |
|---|---|---|---|---|---|
| **Logistic Regression** | 89.6% | **88.7%** | 0.9% | No ✓ | **0.887** |
| Decision Tree | 84.5% | 82.4% | 2.1% | No ✓ | 0.824 |
| Random Forest | 100% | 90.8% | 9.2% | Sì ⚠ | 0.886 |

> Lo **score corretto** penalizza i modelli con gap train/test > 5%: `score = acc_test − (gap − 0.05) × 0.5`. Random Forest ottiene accuracy assoluta più alta (90.8%) ma evidenzia overfitting marcato (train = 100%). La Logistic Regression, pur con accuracy leggermente inferiore, è il modello più stabile e generalizzabile.

### Modello selezionato: Logistic Regression

| Metrica | Valore |
|---|---|
| Accuracy test | **88.7%** |
| Cross-validation 5-fold | **89.4% ± 0.5%** |
| Gap train/test | **0.9%** — nessun overfitting |
| Studenti classificati correttamente | **1132 / 1276** |

**Classification report (test set):**

|  | Precision | Recall | F1 |
|---|---|---|---|
| Low Performance (0) | 0.86 | 0.88 | 0.87 |
| High Performance (1) | 0.91 | 0.89 | 0.90 |
| **Media pesata** | **0.89** | **0.89** | **0.89** |

### Feature più influenti

Ordinate per valore assoluto del coefficiente (dati normalizzati — i coefficienti sono direttamente confrontabili):

| Rank | Feature | Coefficiente | Direzione |
|---|---|---|---|
| 1 | `Attendance` | +3.371 | → High Performance |
| 2 | `Hours_Studied` | +2.595 | → High Performance |
| 3 | `Previous_Scores` | +1.048 | → High Performance |
| 4 | `Tutoring_Sessions` | +0.879 | → High Performance |
| 5 | `Peer_Influence` | +0.562 | → High Performance |
| … | `Gender`, `Family_Income`, `School_Type` | ≈ 0 | — |

I fattori comportamentali (`Attendance`, `Hours_Studied`) dominano nettamente sui fattori socio-demografici (`Gender`, `Family_Income`, `School_Type`), che il modello considera praticamente irrilevanti.

---

## Installazione e utilizzo

### Prerequisiti

- Python 3.8+
- Jupyter Notebook o JupyterLab

### Setup

```bash
# Clona il repository
git clone https://github.com/<tuo-username>/<nome-repo>.git
cd <nome-repo>

# Crea un ambiente virtuale (consigliato)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Installa le dipendenze
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Esecuzione

Lancia i notebook nell'ordine corretto — ogni notebook dipende dall'output del precedente:

```bash
jupyter notebook
```

Esegui in sequenza:
1. `01_EDA.ipynb` — produce `data/dataset_pulito.csv` e i grafici in `grafici/`
2. `02_Preprocessing.ipynb` — produce i file CSV in `data/`
3. `03_ML.ipynb` — produce grafici di valutazione in `grafici/`

> **Importante:** il dataset originale `data/StudentPerformanceFactors.csv` deve essere presente prima di eseguire `01_EDA.ipynb`. Scaricalo da [Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) e posizionalo nella cartella `data/`.

---

## Tecnologie

- **Python 3.x**
- **pandas** — manipolazione dati
- **NumPy** — calcoli numerici
- **Matplotlib / Seaborn** — visualizzazioni
- **scikit-learn** — preprocessing, modelli ML, metriche

---

## Limitazioni e possibili miglioramenti

- Il **Label Encoding** assegna codici in ordine alfabetico, non semantico. Per le variabili ordinali (Low/Medium/High) sarebbe più corretto usare `OrdinalEncoder` con ordine esplicito.
- La **soglia mediana** è relativa a questo dataset — non trasferibile direttamente ad altri contesti senza ricalibrazione.
- Il **Random Forest** non è stato ottimizzato con hyperparameter tuning (`max_depth`, `min_samples_leaf`): con una ricerca a griglia potrebbe competere con la LR senza overfitting.
- Possibile estensione: regressione invece di classificazione, per predire il punteggio esatto anziché la categoria.

---

## Autori

Progetto sviluppato come esercitazione di Machine Learning applicato.
