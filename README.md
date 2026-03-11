# 🩺 ML Dashboard — Diabetes Health Indicators

> **Расчётно-графическая работа** по дисциплине «Машинное обучение и большие данные»  
> Тема: «Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных»

---

## 📋 Описание проекта

Веб-приложение на **Streamlit** для классификации риска диабета по данным опроса о здоровье (Diabetes Health Indicators Dataset, CDC). Приложение позволяет:

- 📊 Исследовать датасет и его статистику (EDA)
- 📈 Визуализировать зависимости в данных (4+ типа графиков)
- 🤖 Получать предсказания от 6 различных ML-моделей
- 📁 Загружать собственный CSV-файл для пакетного предсказания

**Целевая переменная:** `Diabetes_012`
| Класс | Значение |
|-------|----------|
| 0 | Нет диабета |
| 1 | Предиабет |
| 2 | Диабет 2 типа |

---

## Cтарт

### 1. Клонировать репозиторий
```bash
git clone https://github.com/AlexeyRau/diabetes-ml-dashboard.git
cd diabetes-ml-dashboard
```

### 2. Создать и активировать виртуальное окружение
```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Установить зависимости
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Подготовить данные и обучить модели
Запустите ноутбук:
```bash
jupyter notebook train_models.ipynb
```
После выполнения всех ячеек в папке `models/` появятся 6 файлов `.pkl`.

### 5. Запустить дашборд
```bash
streamlit run app.py
```

---

## 🌐 Деплой (Streamlit Cloud)

Приложение задеплоено на Streamlit Cloud:  
🔗 **[Ссылка на веб-приложение](https://ВАШ_APP.streamlit.app)**  
*(заменить на реальную ссылку после деплоя)*

---

## 🗂️ Структура проекта

```
📦 diabetes-ml-dashboard/
├── 📄 app.py                          # Streamlit-приложение (дашборд)
├── 📄 train_models.ipynb              # Ноутбук: обучение и сериализация моделей
├── 📄 requirements.txt                # Зависимости
├── 📄 README.md                       # Документация проекта
│
├── 📂 models/                         # Сериализованные модели ML
│   ├── dt_classifier_model.pkl        # ML1: Decision Tree
│   ├── gb_classifier_model.pkl        # ML2: Gradient Boosting
│   ├── cb_classifier_model.pkl        # ML3: CatBoost
│   ├── bag_classifier_model.pkl       # ML4: Bagging
│   ├── stack_classifier_model.pkl     # ML5: Stacking
│   └── nn_classifier_model.pkl        # ML6: MLP Neural Network
│
├── 📂 datasets/
│   └── filtered_diabetes_health_indicators.csv
│
└── 📷 photo.jpg                       # Фото разработчика
```

---

## 🤖 Модели машинного обучения

| # | Тип | Модель | Метрика F1-macro |
|---|-----|--------|-----------------|
| ML1 | Классическая | `DecisionTreeClassifier` | ~0.40 |
| ML2 | Бустинг | `GradientBoostingClassifier` | ~0.40 |
| ML3 | Продвинутый бустинг | `CatBoostClassifier` | ~0.40 |
| ML4 | Бэггинг | `BaggingClassifier` | ~0.40 |
| ML5 | Стэкинг | `StackingClassifier` (DT + kNN + LR → LR) | ~0.41 |
| ML6 | Нейронная сеть | `MLPClassifier` (128→64, relu, adam) | ~0.40 |

Все модели обёрнуты в `sklearn.Pipeline` с `StandardScaler`. Сериализация выполнена через `joblib`.  
ML6 дополнительно включает `SelectKBest(f_classif)` для отбора значимых признаков.

---

## 📊 Структура дашборда

| Страница | Содержание |
|----------|------------|
| 👨‍💻 **Разработчик** | ФИО, группа, фото, тема РГР |
| 📂 **Датасет** | Описание предметной области, признаки, EDA, предобработка |
| 📈 **Визуализации** | Гистограммы, boxplot, тепловая карта корреляций, scatter и др. |
| 🤖 **Предсказание** | Выбор модели, ввод данных вручную или загрузка CSV, результат |

---

## 📐 Признаки датасета

| Признак | Описание | Тип |
|---------|----------|-----|
| `HighBP` | Высокое артериальное давление | бинарный |
| `HighChol` | Высокий холестерин | бинарный |
| `CholCheck` | Проверка холестерина за 5 лет | бинарный |
| `BMI` | Индекс массы тела | числовой |
| `Smoker` | Курение (≥100 сигарет за жизнь) | бинарный |
| `Stroke` | Инсульт в анамнезе | бинарный |
| `HeartDiseaseorAttack` | Ишемическая болезнь / инфаркт | бинарный |
| `PhysActivity` | Физическая активность за 30 дней | бинарный |
| `Fruits` | Употребление фруктов ≥1 раза в день | бинарный |
| `Veggies` | Употребление овощей ≥1 раза в день | бинарный |
| `HvyAlcoholConsump` | Злоупотребление алкоголем | бинарный |
| `AnyHealthcare` | Наличие медицинской страховки | бинарный |
| `NoDocbcCost` | Не мог позволить врача из-за стоимости | бинарный |
| `GenHlth` | Общее состояние здоровья | порядковый (1–5) |
| `MentHlth` | Дней с плохим психическим здоровьем | числовой (0–30) |
| `PhysHlth` | Дней с плохим физическим здоровьем | числовой (0–30) |
| `DiffWalk` | Трудности при ходьбе | бинарный |
| `Sex` | Пол (0 — женский, 1 — мужской) | бинарный |
| `Age` | Возрастная группа | порядковый (1–13) |
| `Education` | Уровень образования | порядковый (1–6) |
| `Income` | Уровень дохода | порядковый (1–8) |

---

## 🔧 Зависимости

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
catboost
joblib
```

> Полный список с версиями — в файле `requirements.txt`

---

## 📖 Источники

1. [Streamlit Documentation](https://docs.streamlit.io)
2. [Diabetes Health Indicators Dataset — Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
4. [CatBoost Documentation](https://catboost.ai/docs/)