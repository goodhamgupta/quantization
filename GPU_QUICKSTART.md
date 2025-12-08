# GPU Quickstart Guide

This guide will help you run the ColQwen3 quantization workflow on your GPU instance.

## Prerequisites

- GPU instance with CUDA support
- Python 3.10+
- `uv` package manager installed
- At least 16GB GPU VRAM (for loading both models during testing)
- The `tomoro-colqwen3-embed-4b` model downloaded locally

## Step 1: Environment Setup (2 min)

First, ensure all dependencies are installed:

```bash
# Install dependencies (if not already done)
uv pip install torch transformers auto-round accelerate safetensors pillow datasets

# Verify installation with GPU support
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "from auto_round import AutoRound; print('AutoRound OK')"
uv run python -c "from datasets import load_dataset; print('datasets OK')"
```

Expected output should show:
- PyTorch version (e.g., 2.6.0)
- `CUDA available: True` (important!)
- AutoRound OK
- datasets OK

## Step 2: Run Quantization (15-20 min)

Now run the main quantization script:

```bash
# Basic quantization with default settings (8 Vidore samples, 200 iterations)
uv run python scripts/quantize_model.py

# Or with custom settings:
uv run python scripts/quantize_model.py \
    --model tomoro-colqwen3-embed-4b \
    --output tomoro-colqwen3-embed-4b-autoround \
    --bits 4 \
    --group-size 128 \
    --iters 200 \
    --nsamples 8
```

**Faster RTN mode** (no calibration tuning, ~5 min):
```bash
uv run python scripts/quantize_model.py --iters 0 --group-size 32
```

**Using specific Vidore rows** (e.g., row 51):
```bash
uv run python scripts/quantize_model.py --vidore-indices 51 42 100 200 300 400 500 600
```

### What to expect:

1. Model loading (~1-2 min)
2. Vidore calibration data download and processing (~1 min)
3. Layer config building (few seconds)
4. Quantization process (10-15 min, or ~1 min for RTN mode)
5. Model saving (~1 min)

Output will be saved to `tomoro-colqwen3-embed-4b-autoround/`

## Step 3: Post-Quantization Setup (2 min)

Run the post-processing script to copy processor files and create README:

```bash
bash scripts/post_quantization.sh
```

Or with custom paths:
```bash
bash scripts/post_quantization.sh tomoro-colqwen3-embed-4b tomoro-colqwen3-embed-4b-autoround
```

This will:
- Copy processor files
- Copy custom model code files (modeling_colqwen3.py, etc.)
- Create a README.md in the output directory

## Step 4: Test Quantized Model (10 min)

Test the quantized model against the original:

```bash
uv run python scripts/test_quantized_model.py
```

Or with custom paths:
```bash
uv run python scripts/test_quantized_model.py \
    --original tomoro-colqwen3-embed-4b \
    --quantized tomoro-colqwen3-embed-4b-autoround
```

### What the test does:

1. Loads original model and runs 3 test queries
2. Unloads original model (to free memory)
3. Loads quantized model and runs same queries
4. Compares embeddings using cosine similarity
5. Checks for NaN/Inf values
6. Calculates model size reduction
7. Saves results to `test_results.json`

### Expected results:

- **Cosine similarity:** >0.90 (ideally >0.95)
- **Model size:** ~3GB (down from ~8.3GB)
- **Compression:** ~3x
- **Speedup:** 1.3-2x
- **Quality:** No NaN/Inf values

## Step 5: Optional - Upload to HuggingFace

If you want to share your quantized model:

```bash
# Login to HuggingFace
huggingface-cli login

# Upload the model
huggingface-cli upload YOUR_USERNAME/tomoro-colqwen3-embed-4b-autoround \
    tomoro-colqwen3-embed-4b-autoround
```

## Troubleshooting

### Issue: CUDA Out of Memory

**During quantization:**
```bash
# Use RTN mode (no calibration tuning)
uv run python scripts/quantize_model.py --iters 0 --group-size 32

# Or reduce calibration samples
uv run python scripts/quantize_model.py --nsamples 4
```

**During testing:**
- The test script loads models sequentially (not simultaneously) to minimize memory usage
- If still OOM, you may need to test on a machine with more VRAM

### Issue: Quantization fails with error

The script has a built-in fallback that will automatically try RTN mode if the standard approach fails. Check the output for:
```
⚠️  Standard approach failed: [error]
Trying fallback: RTN mode for VLMs...
```

### Issue: Import errors

Make sure you're using `uv run python` to execute the scripts, not just `python`:
```bash
# Correct
uv run python scripts/quantize_model.py

# May not work if packages aren't in system Python
python scripts/quantize_model.py
```

### Issue: Vidore dataset download is slow

The first run will download the Vidore dataset. Subsequent runs will use the cached version. If it's too slow, you can:
1. Use fewer samples: `--nsamples 4`
2. Pre-download on a faster connection

## Complete Workflow Example

Here's a complete end-to-end example:

```bash
# 1. Setup (already done if you're reading this)
cd /path/to/conversion

# 2. Verify environment
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Run quantization
uv run python scripts/quantize_model.py \
    --model tomoro-colqwen3-embed-4b \
    --output tomoro-colqwen3-embed-4b-autoround \
    --bits 4 \
    --group-size 128 \
    --iters 200 \
    --nsamples 8

# 4. Post-processing
bash scripts/post_quantization.sh

# 5. Test
uv run python scripts/test_quantized_model.py

# 6. Check results
cat test_results.json

# 7. (Optional) Upload
# huggingface-cli login
# huggingface-cli upload YOUR_USERNAME/model-name tomoro-colqwen3-embed-4b-autoround
```

## Quick RTN Mode Example (Fastest)

If you just want to test quickly without full calibration tuning:

```bash
# Quantize (RTN mode - ~5 min)
uv run python scripts/quantize_model.py --iters 0 --group-size 32

# Post-process
bash scripts/post_quantization.sh

# Test
uv run python scripts/test_quantized_model.py
```

RTN mode is faster but may have slightly lower quality (but often still >90% cosine similarity).

## Expected Timeline

- **Phase 1 (Setup):** 2 min (if already installed)
- **Phase 2 (Quantization):** 15-20 min (or 5 min for RTN mode)
- **Phase 3 (Post-processing):** 2 min
- **Phase 4 (Testing):** 10 min
- **Total:** ~25-35 min (or ~15-20 min for RTN mode)

## Files Generated

After completion, you'll have:

```
tomoro-colqwen3-embed-4b-autoround/
├── model.safetensors           # Quantized model weights
├── config.json                 # Model configuration
├── quantization_metadata.json  # Quantization details
├── modeling_colqwen3.py        # Custom model code
├── configuration_colqwen3.py   # Custom config code
├── processing_colqwen3.py      # Custom processor code
├── preprocessor_config.json    # Processor configuration
└── README.md                   # Usage instructions

test_results.json               # Test results and metrics
```

## Next Steps

Once you have a quantized model that meets your quality requirements:

1. ✅ Test with your own specific use cases and documents
2. ✅ Benchmark on your target deployment environment
3. ✅ Compare quality metrics on a larger test set
4. ⏳ Deploy locally or upload to HuggingFace
5. ⏳ (Optional) Export to GGUF format for llama.cpp if needed

## Support

If you encounter issues:
1. Check GPU memory with `nvidia-smi`
2. Verify CUDA is available in PyTorch
3. Try RTN mode as a fallback
4. Check logs for specific error messages
5. Reduce batch size or number of calibration samples

Good luck with your quantization! 🚀
