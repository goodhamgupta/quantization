#!/bin/bash
set -e

MODEL_PATH="${1:-tomoro-colqwen3-embed-4b}"
OUTPUT_DIR="${2:-tomoro-colqwen3-embed-4b-autoround}"

echo "=========================================="
echo "Post-Quantization Setup"
echo "=========================================="

# Copy processor files
echo ""
echo "Copying processor files..."
uv run python -c "
from transformers import AutoProcessor
from pathlib import Path

processor = AutoProcessor.from_pretrained('${MODEL_PATH}', trust_remote_code=True)
processor.save_pretrained('${OUTPUT_DIR}')
print('✓ Processor saved')
"

# Copy custom code files
echo ""
echo "Copying custom model files..."
for file in modeling_colqwen3.py configuration_colqwen3.py processing_colqwen3.py; do
    if [ -f "${MODEL_PATH}/${file}" ]; then
        cp "${MODEL_PATH}/${file}" "${OUTPUT_DIR}/${file}"
        echo "  ✓ Copied ${file}"
    fi
done

# Create README
echo ""
echo "Creating README..."
cat > "${OUTPUT_DIR}/README.md" << 'EOF'
# ColQwen3-Embed-4B Quantized (Auto-Round)

4-bit quantized version of [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) using Intel Auto-Round with real Vidore calibration data.

Quantized using Auto-Round: https://github.com/intel/auto-round
EOF

echo "✓ README created"

echo ""
echo "=========================================="
echo "✓ Post-quantization setup complete!"
echo "=========================================="
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Test: uv run python scripts/test_quantized_model.py"
echo "  2. Upload (optional): huggingface-cli upload YOUR_USERNAME/model-name ${OUTPUT_DIR}"
