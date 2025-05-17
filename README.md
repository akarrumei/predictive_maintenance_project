# Предиктивное обслуживание оборудования (Predictive Maintenance)

## Описание проекта

Этот проект решает задачу бинарной классификации: **предсказание отказа оборудования (0 или 1)** на основе данных о состоянии машины.  
Результаты оформлены в виде интерактивного веб-приложения на **Streamlit**.

---

## Описание датасета

Используется синтетический датасет [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset), содержащий:

- 10 000 записей
- 14 признаков (температура, скорость, крутящий момент, износ и т.д.)
- Целевая переменная: `Machine failure` (0 — всё работает, 1 — произошёл сбой)

---

## Структура проекта

```bash
predictive_maintenance_project/
├── app.py                  # Главный файл (навигация между страницами)
├── analysis_and_model.py   # Загрузка, обучение, визуализация, предсказания
├── presentation.py         # Презентация проекта с помощью reveal_slides
├── requirements.txt        # Все зависимости для запуска
├── README.md               # Этот файл
├── .gitignore              # Игнорируемые файлы Git
├── data/
│   └── predictive_maintenance.csv   # CSV-файл, если используется локально
└── video/
    └── demo.mp4            # (опционально) Видео-демонстрация
```

---

## Как запустить проект

### 1. Клонируй репозиторий

#### Через HTTPS:
```bash
git clone https://github.com/<твой_логин>/predictive_maintenance_project.git
```

#### Или через SSH:
```bash
git clone git@github.com:<твой_логин>/predictive_maintenance_project.git
```

Перейди в папку проекта:
```bash
cd predictive_maintenance_project
```

### 2. Установи зависимости
```bash
pip install -r requirements.txt
```

### 3. Запусти приложение
```bash
streamlit run app.py
```

---

## Используемые технологии

- Python 3
- Streamlit
- Pandas, Seaborn, Matplotlib
- Scikit-learn
- XGBoost
- streamlit-reveal-slides

---

## 

## Что реализовано

- Загрузка и предобработка данных
- Обучение модели (Logistic Regression)
- Accuracy, Confusion Matrix, ROC-кривая
- Интерактивный ввод новых данных
- Презентация проекта на отдельной странице
- Готово к публикации на GitHub

---

## Возможные улучшения

- Добавить выбор разных моделей (XGBoost, RandomForest и др.)
- Кросс-валидация
- Автоматическая балансировка классов
- Сохранение модели (`joblib`, `pickle`)

---

## Лицензия

Проект создан в учебных целях. Используй и модифицируй свободно.
