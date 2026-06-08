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

# Набор заранее выбранных примеров из разных классов
DEMO_IMAGES=(
  "data/ocr_rvl_cdip/images_tif/presentation/0000001531.tif"
  "data/ocr_rvl_cdip/images_tif/invoice/0000037010.tif"
  "data/ocr_rvl_cdip/images_tif/email/0001451592.tif"
)

idx=1
for img in "${DEMO_IMAGES[@]}"; do
  if [[ ! -f "$img" ]]; then
    echo "Demo image not found: $img"
    exit 1
  fi

  echo "=== Demo #$idx ==="
  echo "Image path: $img"
  echo

  python scripts/demo_document_pipeline.py "$img"

  echo
  idx=$((idx + 1))
done

echo "=== Demo finished ==="