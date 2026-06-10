# Document Checker Course Project

Прототип системы автоматизированной проверки документов для курсового проекта. Система построена вокруг OCR‑распознавания и текстовой классификации и ориентирована на воспроизводимый CPU‑only запуск.

## Идея проекта

Система принимает изображение документа или PDF, извлекает текст с помощью OCR, обрабатывает его и применяет модель машинного обучения для получения итогового результата.  

Основная прикладная задача — классификация типа документа по тексту, полученному после OCR (например, invoice, resume, email, form и т.д.).

## Цели

Проект нацелен на реализацию демонстрируемого прототипа, который можно показать на защите от начала до конца:

- загрузка документа;
- запуск OCR;
- предобработка текста;
- построение признаков;
- предсказание модели;
- вывод результата через простой интерфейс (Streamlit и FastAPI).

Приоритеты: работоспособность, воспроизводимость, CPU‑only выполнение и наличие измеримых метрик OCR и итоговой ML‑модели.

## Структура проекта

```text
.
├── apps/
│   └── streamlit/           # Streamlit-приложение для локальной демонстрации
├── artifacts/
│   ├── figures/             # Графики и визуализации (при необходимости)
│   ├── metrics/             # Вспомогательные метрики и отчёты
│   ├── models/              # Сохранённые модели текстовой классификации
│   └── ocr_eval/            # Результаты сравнения OCR и ручная разметка
├── data/
│   ├── raw/                 # Исходные данные (малый корпус RVL‑CDIP‑small‑200)
│   └── ocr_rvl_cdip/        # Основной OCR‑корпус для классификации документов
├── notebooks/
│   └── document_classification_baseline.ipynb
├── reports/                 # Итоговый отчёт и вспомогательные материалы
├── scripts/                 # Вспомогательные скрипты подготовки данных, OCR и обучения
├── slides/                  # Материалы презентации
├── src/
│   └── document_checker/    # Основной код проекта (dataset, OCR, pipeline, API, модели)
└── tests/                   # Unit- и integration-тесты
```

## Основные директории

### `data/` — данные проекта

- `raw/` — исходные файлы малого корпуса RVL‑CDIP‑small‑200 (стенд для OCR‑экспериментов).
- `ocr_rvl_cdip/` — основной корпус для OCR и классификации документов (результат OCR по крупной выборке RVL‑CDIP).

### `notebooks/`

- `document_classification_baseline.ipynb` — основной ноутбук с EDA, OCR‑исследованием и базовым текстовым pipeline.

### `src/document_checker/`

- `dataset.py` — работа с малым корпусом RVL‑CDIP‑small‑200: индексация, загрузка, выборки, проверки путей.
- `ocr/` — запуск и обёртки OCR (Tesseract через `pytesseract`), метрики качества OCR (character/word accuracy, Levenshtein, WER).
- `preprocessing/` — предобработка и очистка текста.
- `features/` — задел под извлечение признаков (например, TF‑IDF).
- `models/` — обучение и инференс текстовых моделей, включая загрузку сохранённых артефактов.
- `pipeline/` — документ‑ориентированный pipeline: документ → OCR → обработка текста → предсказание.
- `api/` — слой FastAPI поверх `document_pipeline` для демонстрации backend‑интерфейса.

### `apps/streamlit/`

- `app.py` — Streamlit‑интерфейс для локальной демонстрации пайплайна.

### `artifacts/`

- `models/`:
  - `text_tfidf_logreg.joblib` — базовый пайплайн TF‑IDF + Logistic Regression.
  - `text_tfidf_linearsvc.joblib` — улучшенный пайплайн SimpleTextCleaner → TF‑IDF → LinearSVC.
- `metrics/` — артефакты с метриками и служебными индексами (например, индекс малого корпуса).
- `figures/` — графики и визуализации для отчёта и презентации.
- `ocr_eval/` — результаты сравнения OCR‑конфигураций и ручная разметка ground truth для оценки OCR.

### `scripts/` — вспомогательные скрипты

Для малого корпуса: индексация, инспекция и проверки путей:

- `save_dataset_index.py`
- `inspect_dataset_index.py`
- `check_load_dataset_index.py`
- `check_dataset_paths.py`
- `check_dataset_paths_batch.py`
- `test_subset_dataset.py`

Для OCR: прогоны Tesseract на выборках, подготовка списка для ручной разметки и расчёт OCR‑метрик:

- `run_single_tesseract_sample.py`
- `run_tesseract_batch_sample.py`
- `run_tesseract_three_configs_batch.py`
- `prepare_manual_annotation_list.py`
- `test_ocr_metrics.py`
- `evaluate_ocr_engines.py`

Для большого корпуса: импорт и подготовка `ocr_rvl_cdip`:

- `import_ocr_rvl_cdip_v2.py`
- `download_ocr_rvl_cdip.py`

Прочее:

- `run_fastapi_app.py` — запуск FastAPI‑backend поверх существующего пайплайна.
- `demo_document_pipeline.py` — CLI‑демо end‑to‑end пайплайна.

### `tests/`

Unit- и integration‑тесты для ключевых компонентов: `document_pipeline`, обёрток моделей и FastAPI‑приложения.

- `test_document_pipeline.py`
- `test_text_classifier.py`
- `test_fastapi_app.py`
- `conftest.py`

### `reports/`

Итоговый отчёт и вспомогательные материалы (структура задаётся отдельным `report_draft_structure.md` при необходимости).

### `slides/`

Итоговая презентация (PPTX/HTML‑слайды) и вспомогательные файлы.

## Датасет `ocr_rvl_cdip`

Основной рабочий корпус проекта — `data/ocr_rvl_cdip`. Это локальный датасет, полученный после OCR‑обработки большого поднабора RVL‑CDIP с выравниванием структуры под задачи текстовой классификации документов.

Структура:

```text
data/ocr_rvl_cdip/
├── ocr_dataset.csv
└── images_tif/
    ├── advertisement/
    ├── budget/
    ├── email/
    ├── file_folder/
    ├── form/
    ├── handwritten/
    ├── invoice/
    ├── letter/
    ├── memo/
    ├── news_article/
    ├── presentation/
    ├── questionnaire/
    ├── resume/
    ├── scientific_publication/
    ├── scientific_report/
    └── specification/
```

Файл `ocr_dataset.csv` содержит:

- ~33k строк (после лёгкой очистки небольшой части строк — чуть меньше, для моделирования).
- 4 колонки:
  - `path` — относительный путь к TIFF‑изображению внутри `images_tif`.
  - `label` — тип документа (16 классов).
  - `reference` — стабильный идентификатор документа в формате `<label>/<id>`.
  - `text` — результат OCR по каждому изображению.

Баланс классов относительно ровный: наиболее частые классы содержат порядка двух тысяч примеров, редкие — около тысячи, что позволяет уверенно использовать линейные модели без сложной балансировки.

### История корпуса

Корпус `ocr_rvl_cdip` был собран из более малого внешнего набора, затем заменён на расширенную версию, при этом:

- имя датасета (`ocr_rvl_cdip`),
- структура директорий (`images_tif/<label>/*.tif`),
- формат `ocr_dataset.csv` (`path`, `label`, `reference`, `text`) и смысл поля `reference`

остались теми же. Это позволяет использовать одни и те же ноутбуки и код поверх новой версии корпуса без изменения путей и схем.

### Подготовка датасета

Полный датасет `ocr_rvl_cdip` не хранится в Git, так как это крупный локальный артефакт. Для удобства он размещён отдельным архивом.

Архив доступен по ссылке:  
<https://drive.google.com/file/d/1NZHeYcX1yWXd5IK9_H_olQS21V-9pQD3/view?usp=drive_link>

Локальная подготовка:

```bash
python scripts/download_ocr_rvl_cdip.py
```

Скрипт:

- проверяет наличие `data/ocr_rvl_cdip/ocr_dataset.csv` и `data/ocr_rvl_cdip/images_tif`;
- при отсутствии датасета:
  - скачивает архив `ocr_rvl_cdip.zip` с Google Drive с помощью `gdown`;
  - распаковывает его в каталог `data/`;
  - при необходимости запускает `scripts/import_ocr_rvl_cdip_v2.py` для конвертации PNG → TIFF и сборки `ocr_dataset.csv`;
- валидирует наличие итогового `ocr_dataset.csv` и каталога `images_tif`;
- если датасет уже присутствует локально, повторная загрузка не выполняется.

## OCR‑часть

OCR‑слой реализован на базе Tesseract (через `pytesseract`) и включает:

- обёртки над Tesseract в `src/document_checker/ocr/`:
  - запуск разных конфигураций Tesseract (например, PSM 6, PSM 11 + OEM 1);
  - удобные функции для пакетного прогона по выборкам;
- метрики качества OCR:
  - нормализация текста;
  - расстояние Левенштейна;
  - character accuracy, word accuracy;
  - word error rate (WER).

Для малого корпуса `data/raw/rvl-cdip-small-200` используются скрипты:

- `run_single_tesseract_sample.py`
- `run_tesseract_batch_sample.py`
- `run_tesseract_three_configs_batch.py` — прогоны OCR по выборкам и конфигурациям;
- `prepare_manual_annotation_list.py` — формирование списка документов для ручной разметки ground truth;
- `evaluate_ocr_engines.py` — расчёт метрик OCR по ручной разметке и сравнение конфигураций Tesseract;
- `test_ocr_metrics.py` — локальная проверка корректности OCR‑метрик.

Итоговые результаты сравнения и ручная разметка лежат в `artifacts/ocr_eval/`.

## Текстовый ML‑pipeline

После OCR текст проходит типичный CPU‑дружественный текстовый pipeline:

- очистка и нормализация текста (`preprocessing/`);
- извлечение признаков с помощью TF‑IDF (`features/`);
- линейный классификатор (Logistic Regression или LinearSVC) (`models/`);
- сохранение обученной модели в `artifacts/models/` и повторная загрузка для инференса без переобучения.

Основные экспериментальные шаги реализованы в ноутбуке `notebooks/document_classification_baseline.ipynb`, а производственный вариант — в коде пакета `src/document_checker/`.

## Архитектура пайплайна и backend

Функциональная архитектура пайплайна:

```text
документ (изображение/PDF)
      ↓
       OCR (Tesseract)
      ↓
     текст
      ↓
   очистка и препроцессинг
      ↓
   TF-IDF / признаки
      ↓
   линейный классификатор
      ↓
    предсказание типа документа
```

На основе этого пайплайна реализованы:

- документ‑ориентированный `document_pipeline` в `src/document_checker/pipeline/`;
- минимальный FastAPI‑backend в `src/document_checker/api/fastapi_app.py`;
- Streamlit‑приложение в `apps/streamlit/app.py`.

Это позволяет демонстрировать проект как полноценный прототип: через веб‑интерфейс и через REST‑API.

## Запуск проекта

### Установка зависимостей

Проект использует Poetry для управления зависимостями.

```bash
poetry install
```

### Подготовка данных

См. раздел «Подготовка датасета»:

```bash
python scripts/download_ocr_rvl_cdip.py
```

### Запуск Streamlit‑демо

```bash
bash run_demo.sh
# или явно
poetry run streamlit run apps/streamlit/app.py
```

### Запуск FastAPI‑backend

```bash
poetry run python scripts/run_fastapi_app.py
```

Это поднимает FastAPI‑приложение, которое использует `document_pipeline` и модельные обёртки для обработки загруженных документов.

### Тесты

Для ключевых компонентов реализован набор unit- и integration‑тестов (`tests/`), покрывающих:

- работу `document_pipeline`;
- поведение обёртки над текстовой моделью;
- FastAPI‑приложение (основные маршруты и ошибки).

Запуск тестов:

```bash
poetry run pytest
```

## Отчёт и презентация

Итоговый отчёт и материалы презентации хранятся локально в:

- `reports/` — структура отчёта и финальный документ;
- `slides/` — финальная презентация (PPTX или HTML‑слайды).

## Возможные направления развития

- добавление новых OCR‑конфигураций и движков;
- расширение набора документов и классов;
- углубление анализа ошибок модели;
- развитие backend‑части и улучшение пользовательского интерфейса (валидаторы, логирование, обработка ошибок).