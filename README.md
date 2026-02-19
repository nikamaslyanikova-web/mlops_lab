# MLOps Lab 1 — Telco Customer Churn (Classification)

Лабораторна робота №1: організація проєкту, первинний аналіз даних (EDA), тренування моделі та логування експериментів у **MLflow**.

## Dataset
**Telco Customer Churn (Classification)**  
Цільова змінна: `Churn` (Yes/No).

Датасет зберігається локально (не комітиться в git):
- `data/raw/telco.csv`

- ## Project structure
mlops_lab/
.gitignore
README.md
requirements.txt
data/
raw/
telco.csv
notebooks/
01_eda.ipynb
src/
train.py
artifacts/ # confusion matrix (log as MLflow artifact)
mlruns/ # MLflow локальні runs
models/ # (за потреби) локальні моделі

Встановити залежності: pip install -r requirements.txt

## EDA
Ноутбук з первинним аналізом: notebooks/01_eda.ipynb
Мінімум в EDA: перевірка типів/пропусків, розподіл Churn (баланс класів), базові графіки (наприклад, MonthlyCharges vs Churn), короткі висновки

## Train + MLflow logging

### Скрипт тренування
- `src/train.py`

### Що реалізовано

**1) Завантаження даних**
- читає `data/raw/telco.csv`

**2) Передобробка**
- `TotalCharges` - перетворення у numeric (некоректні/порожні значення - `NaN`)
- `customerID` видаляється 

**3) Розбиття даних**
- `train_test_split(..., stratify=y)` для збереження пропорцій класів у train/test

**4) Модель**
- `RandomForestClassifier(class_weight="balanced")` (корисно через дисбаланс класів `Churn`)

**5) MLflow (логування експериментів)**
- **параметри:** `max_depth`, `n_estimators`, `min_samples_leaf`, тощо  
- **метрики:** `test_accuracy`, `test_f1`, `test_roc_auc`  
- **артефакти:** `confusion_matrix.png`  
- **модель:** збереження моделі всередині run (`mlflow.sklearn.log_model`)

---

### Запуск одного експерименту
```bash
python src/train.py --max-depth 2
python src/train.py --max-depth 4
python src/train.py --max-depth 8
python src/train.py --max-depth 10
python src/train.py --max-depth 14


