"""
Exports BGE-small to quantized ONNX - the one missing artifact.
Run ONCE in Colab. Output goes to Drive so it survives session restarts.

Your mlp_weights.npz, issue_centroids.npy, and config.json were already
produced by export_mlp_and_clusters.py and are sitting in Drive. This
script only produces the missing bge_onnx_quantized/ folder.

Requires:
  !pip install optimum[onnxruntime] onnx -q
"""

import os
from pathlib import Path

# Drive path - matches where your other artifacts already are
DRIVE_ROOT = Path("/content/drive/MyDrive/Dataset/embeddings_output")
EXPORT_DIR = DRIVE_ROOT / "lambda_deploy_artifacts"
EXPORT_DIR.mkdir(exist_ok=True)

from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

model_id = "BAAI/bge-small-en-v1.5"
fp32_dir = EXPORT_DIR / "bge_onnx_fp32"
quantized_dir = EXPORT_DIR / "bge_onnx_quantized"

print("Step 1: Exporting BGE-small to ONNX fp32...")
ort_model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
ort_model.save_pretrained(fp32_dir)
tokenizer.save_pretrained(fp32_dir)
print(f"  Saved fp32 to {fp32_dir}")

print("\nStep 2: Quantizing to int8...")
quantizer = ORTQuantizer.from_pretrained(fp32_dir)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir=quantized_dir, quantization_config=qconfig)
tokenizer.save_pretrained(quantized_dir)

onnx_size = os.path.getsize(quantized_dir / "model_quantized.onnx") / (1024 * 1024)
print(f"  Quantized model size: {onnx_size:.1f} MB")
if onnx_size > 60:
    print("  WARNING: >60MB - quantization may not have applied correctly.")
else:
    print("  Size looks correct (expected 30-40MB).")

print("\nFinal artifact list in lambda_deploy_artifacts/:")
for f in sorted(EXPORT_DIR.rglob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(EXPORT_DIR)}: {size_kb:.1f} KB")

print("\nDone. Set ARTIFACT_DIR to this path for local testing:")
print(f"  os.environ['ARTIFACT_DIR'] = '{EXPORT_DIR}'")
