# Document Checker Course Project

Прототип системы автоматизированной проверки документов для курсового проекта.

## Идея проекта

Система принимает изображение документа или PDF, извлекает текст с помощью OCR, обрабатывает его и применяет модель машинного обучения для получения итогового результата.

Основная прикладная задача проекта — классификация типа документа по тексту, полученному после OCR (например, `invoice`, `resume`, `email`, `form` и т.д.).

## Цель

Разработать демонстрируемый CPU-only прототип, который можно показать на защите:

- загрузка документа;
- OCR;
- предобработка текста;
- предсказание модели;
- вывод результата через простой интерфейс.

## Структура проекта

- `data/` — данные проекта:
  - `raw/` — исходные файлы небольшого корпуса RVL‑CDIP‑small‑200 (используется как вспомогательный стенд для OCR-экспериментов).
  - `interim/` — промежуточные данные.
  - `processed/` — подготовленные данные для обучения (зарезервировано под дальнейший пайплайн).
  - `ocr_eval/` — вспомогательные данные для оценки OCR.
  - `ocr_rvl_cdip/` — основной корпус для OCR и классификации документов (результат OCR по крупной выборке RVL‑CDIP).
- `notebooks/` — Jupyter Notebook для EDA, OCR-исследования и экспериментов с ML:
  - `document_classification_baseline.ipynb` — основной ноутбук с текстовым пайплайном и обучением моделей.
- `src/document_checker/` — основной код проекта:
  - `dataset.py` — работа с малым корпусом RVL‑CDIP‑small‑200 (индексация, загрузка, выборки, проверки путей).
  - `ocr/` — запуск и обёртки OCR (`tesseract.py`) и метрики качества OCR (`metrics.py`).
  - `preprocessing/` — заготовка под очистку и подготовку текста.
  - `features/` — заготовка под извлечение признаков.
  - `models/` — заготовка под обучение и инференс ML-моделей.
  - `pipeline/` — заготовка под полный пайплайн обработки документа.
  - `api/` — заготовка под служебный слой для интерфейса и/или backend.
- `apps/streamlit/` — каркас Streamlit-приложения для локальной демонстрации (будет заполняться на финальном этапе).
- `artifacts/` — сохранённые артефакты:
  - `models/` — обученные модели текстовой классификации:
    - `text_tfidf_logreg.joblib` — базовый пайплайн TF‑IDF + Logistic Regression.
    - `text_tfidf_linearsvc.joblib` — улучшенный пайплайн SimpleTextCleaner → TF‑IDF → LinearSVC.
  - `metrics/` — служебные метрики и отчёты (например, индекс малого корпуса).
  - `figures/` — графики и визуализации (зарезервировано).
  - `ocr_eval/` — результаты сравнения OCR-конфигураций и ручная разметка для оценки.
- `scripts/` — вспомогательные скрипты:
  - для малого корпуса: индексация, инспекция и проверки путей (`save_dataset_index.py`, `inspect_dataset_index.py`, `check_load_dataset_index.py`, `check_dataset_paths.py`, `check_dataset_paths_batch.py`, `test_subset_dataset.py`);
  - для OCR: прогоны Tesseract на выборках, подготовка списка для ручной разметки и оценка OCR (`run_single_tesseract_sample.py`, `run_tesseract_batch_sample.py`, `run_tesseract_three_configs_batch.py`, `prepare_manual_annotation_list.py`, `test_ocr_metrics.py`, `evaluate_ocr_engines.py`);
  - для большого корпуса: импорт и подготовка `ocr_rvl_cdip` (`import_ocr_rvl_cdip.py`, `import_ocr_rvl_cdip_v2.py`, `download_ocr_rvl_cdip.py`).
- `tests/` — тесты (каркас).
- `configs/` — конфигурационные файлы (каркас).
- `demo/` — примеры документов для демонстрации (каркас).
- `slides/` — материалы для презентации (каркас).

## Датасет `ocr_rvl_cdip`

Основной рабочий корпус проекта — `ocr_rvl_cdip`. Это локальный датасет, полученный после OCR-обработки большого поднабора RVL‑CDIP с выравниванием структуры под задачи проекта.

Формат данных:

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

- 33014 строк (после лёгкой очистки остаётся 33009 строк для моделирования).
- 4 колонки:
  - `path` — относительный путь к TIFF-изображению внутри `images_tif`.
  - `label` — тип документа (16 классов).
  - `reference` — стабильный идентификатор документа в формате `<label>/<id>`.
  - `text` — результат OCR по каждому изображению.

Баланс классов относительно ровный: самый частый класс (`memo`) содержит около 2400 примеров, самый редкий (`scientific_publication`) — около 1000, что позволяет уверенно использовать линейные модели без сложной балансировки.

### История корпуса

Изначально датасет `ocr_rvl_cdip` был собран из меньшего внешнего корпуса. Позже он был заменён на расширенную версию (`ocr_dataset_v2`), но при этом:

- имя датасета (`ocr_rvl_cdip`),
- структура директорий (`images_tif/<label>/*.tif`),
- формат `ocr_dataset.csv` (`path`, `label`, `reference`, `text`) и смысл поля `reference`

остались теми же. Это гарантирует, что ноутбуки и код продолжают работать поверх новой версии корпуса без изменений в путях и схемах.

## Подготовка датасета

Полный датасет `ocr_rvl_cdip` не хранится в GitHub, потому что это крупный локальный артефакт.

Архив датасета размещён на Google Drive:

<https://drive.google.com/file/d/1OxRYccKmPpnJEAIAMEQ_R4AnvDrjHUf3/view?usp=sharing>

Для локальной подготовки датасета используйте:

```bash
python scripts/download_ocr_rvl_cdip.py
```

Скрипт делает следующее:

1. Проверяет, существует ли уже `data/ocr_rvl_cdip/ocr_dataset.csv` и `data/ocr_rvl_cdip/images_tif`.
2. Если датасета нет:
   - скачивает архив `ocr_rvl_cdip.zip` с Google Drive с помощью `gdown`;
   - распаковывает его в каталог `data/`;
   - при необходимости запускает `scripts/import_ocr_rvl_cdip_v2.py` для конвертации PNG → TIFF и сборки `ocr_dataset.csv`.
3. Валидирует наличие итогового `ocr_dataset.csv` и каталога `images_tif`.

После подготовки ожидается такая структура:

```text
data/ocr_rvl_cdip/
├── ocr_dataset.csv
└── images_tif/
    └── <подкаталоги по 16 классам>
```

Если датасет уже присутствует локально, скрипт обнаружит его и ничего не будет скачивать повторно.

## OCR-часть проекта

OCR-слой проекта реализован на базе Tesseract через `pytesseract` и включает:

- Обёртки над Tesseract в `src/document_checker/ocr/tesseract.py`:
  - `run_tesseract_on_image` — базовая конфигурация.
  - `run_tesseract_on_image_psm6` — режим PSM 6 (один блок текста).
  - `run_tesseract_on_image_sparse_lstm` — режим PSM 11 + OEM 1 (sparse text, LSTM only).
- Метрики качества OCR в `src/document_checker/ocr/metrics.py`:
  - нормализация текста,
  - расстояние Левенштейна,
  - character accuracy,
  - word accuracy,
  - word error rate (WER).

Для малого корпуса `data/raw/rvl-cdip-small-200` реализован набор скриптов:

- `run_single_tesseract_sample.py`, `run_tesseract_batch_sample.py`, `run_tesseract_three_configs_batch.py` — прогоны OCR на подвыборках по нескольким конфигурациям.
- `prepare_manual_annotation_list.py` — формирование списка документов для ручной разметки ground truth.
- `evaluate_ocr_engines.py` — расчёт метрик OCR по ручной разметке и сравнение нескольких конфигураций Tesseract.
- `test_ocr_metrics.py` — локальная проверка корректности метрик.

Итоговые результаты сравнения конфигураций OCR и ручная 