#!/usr/bin/env bash
set -euo pipefail

# Скрипт демо для курсового проекта Document Checker:
# document (image) -> OCR (Tesseract) -> text -> ML (LinearSVC) -> predicted label

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== Document Checker Demo ==="
echo "Project root: $PROJECT_ROOT"
echo

echo "NOTE: предполагается, что Python-окружение уже активировано (Poetry/venv)."
echo "Если нет — активируйте его перед запуском: например,"
echo "  source .venv/bin/activate"
echo

# Пример 1: заранее выбранное изображение презентации
DEMO_IMAGE_1="data/ocr_rvl_cdip/images_tif/presentation/2028818454_2028818503.tif"

if [[ ! -f "$DEMO_IMAGE_1" ]]; then
  echo "Demo image not found: $DEMO_IMAGE_1"
  exit 1
fi

echo "=== Demo #1: single document image ==="
echo "Image path: $DEMO_IMAGE_1"
echo

python scripts/demo_document_pipeline.py "$DEMO_IMAGE_1"

echo
echo "=== Demo finished ==="