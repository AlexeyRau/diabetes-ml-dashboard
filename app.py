import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ML Дашборд — Diabetes Health Indicators",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1f4e79; margin-bottom: 0.2rem;}
    .sub-header  {font-size: 1.1rem; color: #555; margin-bottom: 1.5rem;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem; border-radius: 12px; color: white !important;
        text-align: center; margin-bottom: 0.5rem;
    }
    .metric-card * { color: white !important; }
    .info-box {
        background: #f0f4ff; border-left: 4px solid #4c72b0;
        padding: 0.8rem 1.2rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
        color: #1a1a1a !important;
    }
    .info-box * { color: #1a1a1a !important; }
    .info-box b, .info-box strong { color: #1f4e79 !important; }
    .info-box code { color: #c0392b !important; background: #e8eaf6; padding: 1px 4px; border-radius: 3px; }
    .result-positive {
        background: #d4edda; border-left: 5px solid #28a745;
        padding: 1rem; border-radius: 0 8px 8px 0; font-weight: bold;
        color: #155724 !important;
    }
    .result-positive * { color: #155724 !important; }
    .result-negative {
        background: #fff3cd; border-left: 5px solid #ffc107;
        padding: 1rem; border-radius: 0 8px 8px 0; font-weight: bold;
        color: #856404 !important;
    }
    .result-negative * { color: #856404 !important; }
    .result-danger {
        background: #f8d7da; border-left: 5px solid #dc3545;
        padding: 1rem; border-radius: 0 8px 8px 0; font-weight: bold;
        color: #721c24 !important;
    }
    .result-danger * { color: #721c24 !important; }
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #1f4e79 0%, #2e6da4 100%);}
    [data-testid="stSidebar"] * {color: white !important;}
</style>
""", unsafe_allow_html=True)

FEATURE_NAMES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

FEATURE_DESCRIPTIONS = {
    'HighBP':               ('Высокое артериальное давление', 'бинарный (0 — нет, 1 — да)'),
    'HighChol':             ('Высокий холестерин', 'бинарный (0 — нет, 1 — да)'),
    'CholCheck':            ('Проверка холестерина за последние 5 лет', 'бинарный (0 — нет, 1 — да)'),
    'BMI':                  ('Индекс массы тела', 'числовой'),
    'Smoker':               ('Курение (≥100 сигарет за жизнь)', 'бинарный (0 — нет, 1 — да)'),
    'Stroke':               ('Инсульт в анамнезе', 'бинарный (0 — нет, 1 — да)'),
    'HeartDiseaseorAttack': ('Ишемическая болезнь / инфаркт', 'бинарный (0 — нет, 1 — да)'),
    'PhysActivity':         ('Физическая активность за последние 30 дней', 'бинарный (0 — нет, 1 — да)'),
    'Fruits':               ('Употребление фруктов ≥1 раза в день', 'бинарный (0 — нет, 1 — да)'),
    'Veggies':              ('Употребление овощей ≥1 раза в день', 'бинарный (0 — нет, 1 — да)'),
    'HvyAlcoholConsump':    ('Злоупотребление алкоголем', 'бинарный (0 — нет, 1 — да)'),
    'AnyHealthcare':        ('Наличие медицинской страховки', 'бинарный (0 — нет, 1 — да)'),
    'NoDocbcCost':          ('Не мог позволить врача из-за стоимости', 'бинарный (0 — нет, 1 — да)'),
    'GenHlth':              ('Общее состояние здоровья', 'порядковый (1 — отличное … 5 — плохое)'),
    'MentHlth':             ('Дней с плохим психическим здоровьем (за 30 дней)', 'числовой (0–30)'),
    'PhysHlth':             ('Дней с плохим физическим здоровьем (за 30 дней)', 'числовой (0–30)'),
    'DiffWalk':             ('Трудности при ходьбе / подъёме по лестнице', 'бинарный (0 — нет, 1 — да)'),
    'Sex':                  ('Пол', '0 — женский, 1 — мужской'),
    'Age':                  ('Возрастная группа', 'порядковый (1=18–24, …, 13=80+)'),
    'Education':            ('Уровень образования', 'порядковый (1–6)'),
    'Income':               ('Уровень дохода', 'порядковый (1–8)'),
}

MODEL_INFO = {
    'DecisionTree':    {'path': 'models/dt_classifier_model.pkl',   'f1': 0.3950, 'label': 'Дерево решений (ML1)'},
    'GradientBoosting':{'path': 'models/gb_classifier_model.pkl',   'f1': 0.3998, 'label': 'Gradient Boosting (ML2)'},
    'CatBoost':        {'path': 'models/cb_classifier_model.pkl',   'f1': 0.4003, 'label': 'CatBoost (ML3)'},
    'Bagging':         {'path': 'models/bag_classifier_model.pkl',  'f1': 0.3945, 'label': 'Bagging (ML4)'},
    'Stacking':        {'path': 'models/stack_classifier_model.pkl','f1': 0.4055, 'label': 'Stacking (ML5)'},
    'NeuralNetwork':   {'path': 'models/nn_classifier_model.pkl',   'f1': 0.4017, 'label': 'Нейронная сеть MLP (ML6)'},
}

CLASS_LABELS = {0: 'Нет диабета', 1: 'Предиабет', 2: 'Диабет 2 типа'}
CLASS_COLORS = {0: 'result-positive', 1: 'result-negative', 2: 'result-danger'}
CLASS_EMOJI  = {0: '✅', 1: '⚠️', 2: '🔴'}

AGE_GROUPS = {
    1: '18–24', 2: '25–29', 3: '30–34', 4: '35–39', 5: '40–44',
    6: '45–49', 7: '50–54', 8: '55–59', 9: '60–64', 10: '65–69',
    11: '70–74', 12: '75–79', 13: '80+',
}

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    obj = joblib.load(path)
    return obj


def predict_with_model(model_key, model_obj, X):
    if isinstance(model_obj, dict) and 'scaler' in model_obj:
        X_sc = model_obj['scaler'].transform(X)
        preds = model_obj['model'].predict(X_sc)
    else:
        preds = model_obj.predict(X)

    preds = np.array(preds)
    if preds.ndim > 1:
        preds = preds.flatten()
    return preds.astype(int)

@st.cache_data
def load_dataset():
    candidates = [
        'filtered_diabetes_health_indicators.csv',
        '/datasets/filtered_diabetes_health_indicators.csv',
        'data/filtered_diabetes_health_indicators.csv',
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

def page_developer():
    st.markdown('<div class="main-header">👨‍💻 Информация о разработчике</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        photo_found = False
        for photo_path in ['photo.jpg', 'Photo.jpg', 'photo.png', 'Photo.png']:
            if os.path.exists(photo_path):
                st.image(photo_path, caption='Фото разработчика', width=280)
                photo_found = True
                break
        if not photo_found:
            st.markdown("""
            <div style="width:280px;height:320px;background:linear-gradient(135deg,#667eea,#764ba2);
                        border-radius:12px;display:flex;align-items:center;justify-content:center;
                        font-size:6rem;">👤</div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📋 Личные данные")
        st.markdown("""
        <div style="background:#f0f4ff; border-left:4px solid #4c72b0;
                    padding:0.8rem 1.2rem; border-radius:0 8px 8px 0; margin:0.5rem 0;
                    color:#1a1a1a !important;">
        <span style="color:#1a1a1a;"><b style="color:#1f4e79;">ФИО:</b> Рау Алексей Евгеньевич<br>
        <b style="color:#1f4e79;">Группа:</b> ФИТ-231<br>
        <b style="color:#1f4e79;">Дисциплина:</b> Машинное обучение и большие данные</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Тема РГР")
        st.markdown("""
        <div style="background:#f0f4ff; border-left:4px solid #4c72b0;
                    padding:0.8rem 1.2rem; border-radius:0 8px 8px 0; margin:0.5rem 0;">
        <span style="color:#1a1a1a;">
        <b style="color:#1f4e79;">Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML
        и анализа данных</b><br><br>
        Датасет: <i>Diabetes Health Indicators</i> — прогнозирование статуса диабета
        у пациентов (CDC BRFSS 2015)
        </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Стек технологий")
        cols = st.columns(3)
        tech = [
            ('🐍', 'Python 3.11'),
            ('🤖', 'Scikit-learn'),
            ('🔥', 'TensorFlow'),
            ('🚀', 'Streamlit'),
            ('🐼', 'Pandas / NumPy'),
            ('📊', 'Matplotlib / Seaborn'),
        ]
        for i, (emoji, name) in enumerate(tech):
            cols[i % 3].markdown(
                f'<div style="background:linear-gradient(135deg,#667eea,#764ba2);'
                f'padding:1rem 1.5rem;border-radius:12px;text-align:center;margin-bottom:0.5rem;">'
                f'<span style="color:white;font-size:1.4rem;">{emoji}</span><br>'
                f'<b style="color:white;">{name}</b></div>',
                unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📚 Используемые модели ML")
    cols = st.columns(3)
    for i, (key, info) in enumerate(MODEL_INFO.items()):
        cols[i % 3].markdown(
            f"""<div style="background:#f0f4ff; border-left:4px solid #4c72b0;
                padding:0.8rem 1.2rem; border-radius:0 8px 8px 0; margin:0.5rem 0;">
            <span style="color:#1a1a1a;">
            <b style="color:#1f4e79;">{info['label']}</b><br>
            F1-macro (hold-out): <code style="color:#c0392b;background:#e8eaf6;
            padding:1px 5px;border-radius:3px;">{info['f1']:.4f}</code>
            </span></div>""",
            unsafe_allow_html=True)

def page_dataset():
    st.markdown('<div class="main-header">📊 Информация о датасете</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f0f4ff; border-left:4px solid #4c72b0;
                padding:0.8rem 1.2rem; border-radius:0 8px 8px 0; margin:0.5rem 0;">
    <span style="color:#1a1a1a;">
    <b style="color:#1f4e79;">Название:</b> Diabetes Health Indicators (CDC BRFSS 2015)<br>
    <b style="color:#1f4e79;">Источник:</b> Behavioral Risk Factor Surveillance System (CDC, США)<br>
    <b style="color:#1f4e79;">Размер:</b> 229 718 записей, 22 признака<br>
    <b style="color:#1f4e79;">Задача:</b> Мультиклассовая классификация (3 класса)
    </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Целевая переменная")
    col1, col2, col3 = st.columns(3)
    for col, (cls, label, cnt) in zip([col1, col2, col3], [
        (0, 'Нет диабета',  '189 994 (82.7%)'),
        (1, 'Предиабет',    '4 629 (2.0%)'),
        (2, 'Диабет 2 типа','35 095 (15.3%)'),
    ]):
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                    padding:1rem 1.5rem; border-radius:12px; text-align:center; margin-bottom:0.5rem;">
        <div style="font-size:1.8rem; color:white;">{CLASS_EMOJI[cls]}</div>
        <b style="color:white;">Класс {cls}</b><br>
        <span style="color:white;">{label}</span><br>
        <small style="color:rgba(255,255,255,0.85);">{cnt}</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📝 Описание признаков")
    rows = []
    for feat, (desc, dtype) in FEATURE_DESCRIPTIONS.items():
        rows.append({'Признак': feat, 'Описание': desc, 'Тип / шкала': dtype})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 🔧 Предобработка данных")
    st.markdown("""
    <div style="background:#f0f4ff; border-left:4px solid #4c72b0;
                padding:0.8rem 1.2rem; border-radius:0 8px 8px 0; margin:0.5rem 0;">
    <ol style="color:#1a1a1a; margin:0; padding-left:1.2rem;">
    <li style="color:#1a1a1a;"><b style="color:#1f4e79;">Загрузка данных.</b> Датасет загружался из CSV-файла (229 718 строк × 22 столбца).</li>
    <li style="color:#1a1a1a;"><b style="color:#1f4e79;">Обработка пропусков.</b> Пропущенные значения отсутствуют — датасет уже очищен.</li>
    <li style="color:#1a1a1a;"><b style="color:#1f4e79;">Выделение признаков и целевой переменной.</b> X — 21 признак, y — Diabetes_012.</li>
    <li style="color:#1a1a1a;"><b style="color:#1f4e79;">Разбиение.</b> Hold-out 80/20 со стратификацией (stratify=y, random_state=42).</li>
    <li style="color:#1a1a1a;"><b style="color:#1f4e79;">Масштабирование.</b> StandardScaler применялся к обучающей выборке (fit_transform), к тестовой — только transform.</li>
    <li style="color:#1a1a1a;"><b style="color:#1f4e79;">Балансировка классов.</b> SMOTE применялся к обучающей выборке для устранения дисбаланса классов.</li>
    <li style="color:#1a1a1a;"><b style="color:#1f4e79;">K-fold валидация.</b> StratifiedKFold(n_splits=5) для надёжной оценки обобщающей способности.</li>
    </ol>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🔍 EDA (разведочный анализ данных)")
    df = load_dataset()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Строк",     f"{df.shape[0]:,}")
        col2.metric("Признаков", df.shape[1])
        col3.metric("Пропусков", int(df.isnull().sum().sum()))
        col4.metric("Дубликатов", int(df.duplicated().sum()))

        with st.expander("📋 Первые 10 строк датасета"):
            st.dataframe(df.head(10), use_container_width=True)
        with st.expander("📈 Статистика по числовым признакам"):
            st.dataframe(df.describe().round(2), use_container_width=True)
    else:
        st.info("ℹ️ Добавьте файл `filtered_diabetes_health_indicators.csv` в папку с приложением для отображения данных.")

def page_visualizations():
    st.markdown('<div class="main-header">📈 Визуализации данных</div>', unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.warning("⚠️ Добавьте файл `filtered_diabetes_health_indicators.csv` для отображения визуализаций.")
        _demo_visualizations()
        return

    _real_visualizations(df)


def _demo_visualizations():
    st.info("📊 Отображаются демонстрационные графики на синтетических данных")
    rng = np.random.default_rng(42)
    n = 1000
    df_demo = pd.DataFrame({
        'BMI':     rng.normal(28, 6, n).clip(12, 70),
        'Age':     rng.integers(1, 14, n),
        'GenHlth': rng.integers(1, 6, n),
        'MentHlth':rng.integers(0, 31, n),
        'PhysHlth':rng.integers(0, 31, n),
        'Diabetes_012': rng.choice([0, 1, 2], n, p=[0.83, 0.02, 0.15]),
    })
    _real_visualizations(df_demo)


def _real_visualizations(df):
    target = 'Diabetes_012'
    num_cols = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth']
    bin_cols = ['HighBP', 'HighChol', 'Smoker', 'Stroke',
                'HeartDiseaseorAttack', 'PhysActivity', 'DiffWalk']
    num_cols = [c for c in num_cols if c in df.columns]
    bin_cols = [c for c in bin_cols if c in df.columns]

    palette = {0: '#55a868', 1: '#f4a623', 2: '#dd4b39'}

    st.subheader("1️⃣ Распределение целевой переменной (Diabetes_012)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df[target].value_counts().sort_index()
    labels = [CLASS_LABELS[i] for i in counts.index]
    axes[0].bar(labels, counts.values, color=[palette[i] for i in counts.index], edgecolor='white', linewidth=1.5)
    axes[0].set_title("Количество наблюдений по классам")
    axes[0].set_ylabel("Количество")
    for j, v in enumerate(counts.values):
        axes[0].text(j, v + 500, f'{v:,}', ha='center', fontsize=9)
    axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%',
                colors=[palette[i] for i in counts.index], startangle=140,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title("Доля классов")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("2️⃣ Распределения числовых признаков по классам (Box-plot)")
    if num_cols:
        fig, axes = plt.subplots(1, len(num_cols), figsize=(4 * len(num_cols), 5))
        if len(num_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, num_cols):
            data_by_class = [df[df[target] == cls][col].dropna() for cls in [0, 1, 2] if cls in df[target].values]
            bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                            medianprops={'color': 'black', 'linewidth': 2})
            for patch, cls in zip(bp['boxes'], [0, 1, 2]):
                patch.set_facecolor(palette.get(cls, 'gray'))
                patch.set_alpha(0.7)
            ax.set_title(col)
            ax.set_xticklabels(['Норма', 'Предиабет', 'Диабет'], rotation=20, fontsize=8)
            ax.set_ylabel(col)
        plt.suptitle("Box-plot числовых признаков по классам", fontsize=12, y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("3️⃣ Тепловая карта корреляций")
    corr_cols = [c for c in FEATURE_NAMES if c in df.columns] + [target]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, annot_kws={'size': 7},
                linewidths=0.5, square=True)
    ax.set_title("Матрица корреляций признаков (нижний треугольник)", fontsize=13, pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("4️⃣ Доля заболевших диабетом по бинарным признакам")
    if bin_cols:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i, col in enumerate(bin_cols[:8]):
            ct = df.groupby([col, target]).size().unstack(fill_value=0)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            ct_pct.plot(kind='bar', ax=axes[i], stacked=True,
                        color=[palette.get(c, 'gray') for c in ct_pct.columns],
                        legend=(i == 0), rot=0)
            axes[i].set_title(col, fontsize=10)
            axes[i].set_ylabel("% наблюдений")
            axes[i].set_xlabel("")
        for j in range(len(bin_cols), 8):
            axes[j].set_visible(False)
        handles = [plt.Rectangle((0,0),1,1, color=palette[c]) for c in [0,1,2]]
        fig.legend(handles, [CLASS_LABELS[c] for c in [0,1,2]],
                   loc='lower right', frameon=True)
        plt.suptitle("Стековые столбчатые диаграммы: доля классов по бинарным признакам",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if 'BMI' in df.columns and 'Age' in df.columns:
        st.subheader("5️⃣ BMI vs Возраст — scatter plot по классам")
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 5))
        for cls in [0, 1, 2]:
            subset = sample[sample[target] == cls]
            ax.scatter(subset['Age'], subset['BMI'],
                       alpha=0.35, s=15, c=palette[cls], label=CLASS_LABELS[cls])
        ax.set_xlabel("Возрастная группа")
        ax.set_ylabel("BMI")
        ax.set_title("Зависимость BMI от возраста (выборка 5 000 записей)")
        ax.legend()
        ax.set_xticks(range(1, 14))
        ax.set_xticklabels([AGE_GROUPS[i] for i in range(1, 14)], rotation=45, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def page_prediction():
    st.markdown('<div class="main-header">🤖 Предсказание моделей ML</div>', unsafe_allow_html=True)

    model_keys = list(MODEL_INFO.keys())
    selected = st.multiselect(
        "Выберите модели для предсказания:",
        model_keys,
        default=model_keys,
        format_func=lambda k: MODEL_INFO[k]['label']
    )

    tab1, tab2 = st.tabs(["📂 Загрузка CSV", "✍️ Ручной ввод"])

    with tab1:
        st.markdown("""
        <div style="background:#f0f4ff; border-left:4px solid #4c72b0;
                    padding:0.8rem 1.2rem; border-radius:0 8px 8px 0; margin:0.5rem 0;">
        <span style="color:#1a1a1a;">
        Загрузите CSV-файл, содержащий признаки пациентов. Файл должен содержать столбцы:<br>
        <code style="color:#c0392b; background:#e8eaf6; padding:2px 5px; border-radius:3px;
                     font-size:0.85rem;">HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
        HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
        NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income</code>
        </span>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Загрузите CSV-файл с данными пациентов:", type=['csv'])

        if uploaded:
            try:
                df_in = pd.read_csv(uploaded)
                st.write("**Превью загруженных данных:**")
                st.dataframe(df_in.head(), use_container_width=True)

                missing_cols = [c for c in FEATURE_NAMES if c not in df_in.columns]
                if missing_cols:
                    st.error(f"❌ В файле отсутствуют столбцы: {missing_cols}")
                else:
                    X_input = df_in[FEATURE_NAMES].copy()
                    if not ((X_input['BMI'] >= 10) & (X_input['BMI'] <= 100)).all():
                        st.warning("⚠️ Некоторые значения BMI выходят за ожидаемый диапазон (10–100)")
                    _run_predictions(X_input, selected)
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")

    with tab2:
        st.markdown("#### Введите данные пациента")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🩺 Состояние здоровья**")
            high_bp   = st.selectbox("Высокое артериальное давление", [0, 1],
                                      format_func=lambda x: "Нет" if x == 0 else "Да",
                                      key='bp')
            high_chol = st.selectbox("Высокий холестерин", [0, 1],
                                      format_func=lambda x: "Нет" if x == 0 else "Да",
                                      key='chol')
            chol_check= st.selectbox("Проверка холестерина за 5 лет", [0, 1],
                                      format_func=lambda x: "Нет" if x == 0 else "Да",
                                      key='ccheck')
            stroke    = st.selectbox("Инсульт в анамнезе", [0, 1],
                                      format_func=lambda x: "Нет" if x == 0 else "Да",
                                      key='stroke')
            heart_dis = st.selectbox("Ишемическая болезнь / инфаркт", [0, 1],
                                      format_func=lambda x: "Нет" if x == 0 else "Да",
                                      key='heart')
            diff_walk = st.selectbox("Трудности с ходьбой", [0, 1],
                                      format_func=lambda x: "Нет" if x == 0 else "Да",
                                      key='walk')
            gen_hlth  = st.slider("Общее состояние здоровья (1=отлично, 5=плохое)", 1, 5, 3, key='ghlth')

        with col2:
            st.markdown("**📏 Физические показатели**")
            bmi       = st.slider("ИМТ (BMI)", 10.0, 98.0, 26.0, 0.5, key='bmi')
            ment_hlth = st.slider("Дней с плохим психическим здоровьем (30 дней)", 0, 30, 0, key='mhlth')
            phys_hlth = st.slider("Дней с плохим физическим здоровьем (30 дней)", 0, 30, 0, key='phlth')
            sex       = st.selectbox("Пол", [0, 1],
                                      format_func=lambda x: "Женский" if x == 0 else "Мужской",
                                      key='sex')
            age       = st.select_slider("Возрастная группа", options=list(AGE_GROUPS.keys()),
                                          format_func=lambda x: AGE_GROUPS[x], key='age')

            st.markdown("**💊 Образ жизни**")
            smoker      = st.selectbox("Курение (≥100 сигарет)", [0, 1],
                                        format_func=lambda x: "Нет" if x == 0 else "Да", key='smoker')
            phys_act    = st.selectbox("Физическая активность (30 дней)", [0, 1],
                                        format_func=lambda x: "Нет" if x == 0 else "Да", key='phys')
            fruits      = st.selectbox("Фрукты ≥1 раза в день", [0, 1],
                                        format_func=lambda x: "Нет" if x == 0 else "Да", key='fruits')
            veggies     = st.selectbox("Овощи ≥1 раза в день", [0, 1],
                                        format_func=lambda x: "Нет" if x == 0 else "Да", key='veggies')
            hvy_alc     = st.selectbox("Злоупотребление алкоголем", [0, 1],
                                        format_func=lambda x: "Нет" if x == 0 else "Да", key='alc')

        with col3:
            st.markdown("**🏥 Социально-экономические факторы**")
            any_hc      = st.selectbox("Наличие медицинской страховки", [0, 1],
                                        format_func=lambda x: "Нет" if x == 0 else "Да", key='hc')
            no_doc      = st.selectbox("Не мог позволить врача (стоимость)", [0, 1],
                                        format_func=lambda x: "Нет" if x == 0 else "Да", key='nodoc')
            education   = st.slider("Уровень образования (1–6)", 1, 6, 4, key='edu')
            income      = st.slider("Уровень дохода (1–8)", 1, 8, 5, key='income')

            st.markdown("---")
            st.markdown("**ℹ️ Расшифровка ИМТ**")
            if bmi < 18.5:
                st.warning(f"BMI {bmi:.1f} — Недостаточный вес")
            elif bmi < 25:
                st.success(f"BMI {bmi:.1f} — Норма")
            elif bmi < 30:
                st.warning(f"BMI {bmi:.1f} — Избыточный вес")
            else:
                st.error(f"BMI {bmi:.1f} — Ожирение")

        predict_btn = st.button("🔮 Получить предсказание", type="primary", use_container_width=True)

        if predict_btn:
            X_manual = pd.DataFrame([{
                'HighBP': high_bp, 'HighChol': high_chol, 'CholCheck': chol_check,
                'BMI': bmi, 'Smoker': smoker, 'Stroke': stroke,
                'HeartDiseaseorAttack': heart_dis, 'PhysActivity': phys_act,
                'Fruits': fruits, 'Veggies': veggies, 'HvyAlcoholConsump': hvy_alc,
                'AnyHealthcare': any_hc, 'NoDocbcCost': no_doc,
                'GenHlth': gen_hlth, 'MentHlth': ment_hlth, 'PhysHlth': phys_hlth,
                'DiffWalk': diff_walk, 'Sex': sex, 'Age': age,
                'Education': education, 'Income': income,
            }])
            _run_predictions(X_manual, selected, manual=True)


def _run_predictions(X_input, selected_models, manual=False):
    if not selected_models:
        st.warning("Выберите хотя бы одну модель.")
        return

    st.markdown("---")
    st.markdown("### 🔍 Результаты предсказания")

    results = []
    for key in selected_models:
        info = MODEL_INFO[key]
        model = load_model(info['path'])
        if model is None:
            results.append({
                'Модель': info['label'],
                'Предсказание': 'Модель не найдена',
                'Класс': -1,
                'F1 (тест)': info['f1'],
            })
            continue
        try:
            preds = predict_with_model(key, model, X_input)
            results.append({
                'Модель': info['label'],
                'Предсказание': [CLASS_LABELS[p] for p in preds],
                'Класс': preds,
                'F1 (тест)': info['f1'],
            })
        except Exception as e:
            results.append({
                'Модель': info['label'],
                'Предсказание': f'Ошибка: {e}',
                'Класс': -1,
                'F1 (тест)': info['f1'],
            })

    CARD_STYLES = {
        0: ('background:#d4edda; border-left:5px solid #28a745;', 'color:#155724;'),
        1: ('background:#fff3cd; border-left:5px solid #ffc107;', 'color:#856404;'),
        2: ('background:#f8d7da; border-left:5px solid #dc3545;', 'color:#721c24;'),
    }

    if manual and len(X_input) == 1:
        cols = st.columns(min(len(results), 3))
        for i, res in enumerate(results):
            with cols[i % len(cols)]:
                cls = res['Класс']
                pred = res['Предсказание']
                if isinstance(pred, list):
                    cls_val = cls[0]
                    pred_str = pred[0]
                    emoji = CLASS_EMOJI.get(cls_val, '❓')
                    bg_style, text_style = CARD_STYLES.get(cls_val, ('background:#f0f4ff;', 'color:#1a1a1a;'))
                    st.markdown(f"""
                    <div style="{bg_style} padding:1rem; border-radius:0 8px 8px 0; margin-bottom:0.5rem;">
                    <b style="{text_style}">{res['Модель']}</b><br>
                    <span style="{text_style} font-size:1.1rem;">{emoji} <b>{pred_str}</b></span><br>
                    <small style="{text_style} opacity:0.8;">F1 = {res['F1 (тест)']:.4f}</small>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.error(f"**{res['Модель']}**\n{pred}")
    else:
        table_rows = []
        for res in results:
            pred = res['Предсказание']
            if isinstance(pred, list):
                for j, p in enumerate(pred):
                    table_rows.append({
                        'Запись №': j + 1,
                        'Модель': res['Модель'],
                        'Предсказание': p,
                        'F1 (тест)': res['F1 (тест)'],
                    })
            else:
                table_rows.append({
                    'Запись №': '—',
                    'Модель': res['Модель'],
                    'Предсказание': pred,
                    'F1 (тест)': res['F1 (тест)'],
                })
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    valid = [r for r in results if isinstance(r['Предсказание'], list)]
    if valid and len(X_input) == 1:
        st.markdown("#### 📊 Сравнение предсказаний моделей")
        model_labels = [r['Модель'] for r in valid]
        f1_scores    = [r['F1 (тест)'] for r in valid]
        pred_classes = [r['Класс'][0] if len(r['Класс']) > 0 else -1 for r in valid]

        fig, ax = plt.subplots(figsize=(10, 4))
        bar_colors = ['#55a868' if p == 0 else ('#f4a623' if p == 1 else '#dd4b39')
                      for p in pred_classes]
        bars = ax.barh(model_labels, f1_scores, color=bar_colors, alpha=0.8, edgecolor='white')
        ax.set_xlabel("F1-macro (на тестовой выборке)")
        ax.set_title("Качество моделей (цвет = предсказанный класс)")
        ax.set_xlim(0, 1.0)
        for bar, f1 in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{f1:.4f}', va='center', fontsize=9)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#55a868', label='Нет диабета'),
                           Patch(facecolor='#f4a623', label='Предиабет'),
                           Patch(facecolor='#dd4b39', label='Диабет 2 типа')]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

PAGES = {
    "👨‍💻 О разработчике":  page_developer,
    "📊 О датасете":         page_dataset,
    "📈 Визуализации":       page_visualizations,
    "🤖 Предсказание":       page_prediction,
}

with st.sidebar:
    st.markdown("## 🩺 ML Дашборд")
    st.markdown("**Diabetes Health Indicators**")
    st.divider()
    selection = st.radio("Навигация", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.markdown("### 📋 Описание моделей")
    for key, info in MODEL_INFO.items():
        st.markdown(f"**{info['label']}**\nF1: `{info['f1']:.4f}`")
    st.divider()
    st.caption("Выполнено в рамках РГР\n«Машинное обучение и большие данные»")

PAGES[selection]()