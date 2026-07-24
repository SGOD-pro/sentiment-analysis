"""
Final production Lambda handler.
Sentiment: MLP forward pass, pure numpy, asymmetric threshold applied.
Issue detection: nearest-centroid lookup, only runs when sentiment=negative.
One embedding computed once, feeds both branches.

Package contents (/opt/model in Lambda layer):
  bge_onnx_quantized/       (ONNX encoder + tokenizer, ~35-40MB)
  mlp_weights.npz           (tiny, <1MB)
  issue_centroids.npy       (tiny, K x 384 floats, KB range)
  config.json

requirements.txt:
  onnxruntime
  numpy
  tokenizers
(NOT: torch, transformers, sentence-transformers, sklearn)
"""

import json
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import time
import logging
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        for key in record.__dict__:
            if key not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__ and key not in ("message", "msg"):
                entry[key] = record.__dict__[key]
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)

logger = logging.getLogger("ml_inference")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# In Lambda, this is the layer mount point - /opt/model is correct there,
# do not change it for deployment. For LOCAL testing before you ever touch
# AWS, set the env var to point at wherever export_mlp_and_clusters.py
# actually wrote its output, e.g.:
#   os.environ["ARTIFACT_DIR"] = "/content/drive/MyDrive/Dataset/embeddings_output/lambda_deploy_artifacts"
# Defaults to artifacts/ relative to this script so it resolves correctly locally and when deployed together.
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"))

with open(f"{ARTIFACT_DIR}/config.json") as f:
    CONFIG = json.load(f)

tokenizer = Tokenizer.from_file(f"{ARTIFACT_DIR}/bge_onnx_quantized/tokenizer.json")
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=256)
sess_options = ort.SessionOptions()

sess_options.enable_mem_pattern = False
sess_options.enable_cpu_mem_arena = False
sess_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

onnx_session = ort.InferenceSession(
    f"{ARTIFACT_DIR}/bge_onnx_quantized/model_quantized.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)


mlp_weights = np.load(f"{ARTIFACT_DIR}/mlp_weights.npz")

centroid_path_npz = f"{ARTIFACT_DIR}/issue_centroids.npz"
centroid_path_npy = f"{ARTIFACT_DIR}/issue_centroids.npy"

if os.path.exists(centroid_path_npz):
    npz_data = np.load(centroid_path_npz)
    issue_centroids = {k: npz_data[k] for k in npz_data.files}
elif os.path.exists(centroid_path_npy):
    issue_centroids = {"cross_category_fallback": np.load(centroid_path_npy)}
else:
    issue_centroids = {}


def embed_texts(texts: list[str]) -> np.ndarray:
    t_embed_start = time.perf_counter()

    t_tok_start = time.perf_counter()
    encodings = tokenizer.encode_batch(texts)  # encode_batch pads to same length automatically
    t_tok_end = time.perf_counter()
    duration_tokenizer_ms = (t_tok_end - t_tok_start) * 1000

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    t_run_start = time.perf_counter()
    outputs = onnx_session.run(
        None,
        {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids},
    )
    t_run_end = time.perf_counter()
    duration_session_run_ms = (t_run_end - t_run_start) * 1000

    last_hidden_state = outputs[0]
    cls_embeddings = last_hidden_state[:, 0, :]  # CLS pooling - verify matches training pipeline
    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
    embeddings = cls_embeddings / np.clip(norms, 1e-9, None)

    t_embed_end = time.perf_counter()
    duration_embed_texts_ms = (t_embed_end - t_embed_start) * 1000

    logger.info("tokenizer", extra={"duration_ms": duration_tokenizer_ms})
    logger.info("session.run", extra={"duration_ms": duration_session_run_ms})
    logger.info("embed_texts", extra={"duration_ms": duration_embed_texts_ms})

    return embeddings


def relu(x):
    return np.maximum(0, x)


def mlp_forward(x: np.ndarray) -> np.ndarray:
    """VERIFY key names against your exported mlp_weights.npz - see export script warning."""
    t_start = time.perf_counter()

    w1, b1 = mlp_weights["net.0.weight"], mlp_weights["net.0.bias"]
    w2, b2 = mlp_weights["net.3.weight"], mlp_weights["net.3.bias"]
    w3, b3 = mlp_weights["net.6.weight"], mlp_weights["net.6.bias"]

    h1 = relu(x @ w1.T + b1)
    h2 = relu(h1 @ w2.T + b2)
    logits = h2 @ w3.T + b3

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    t_end = time.perf_counter()
    duration_mlp_forward_ms = (t_end - t_start) * 1000
    logger.info("mlp_forward", extra={"duration_ms": duration_mlp_forward_ms})

    return probs


def apply_asymmetric_threshold(probs: np.ndarray):
    cfg = CONFIG["sentiment"]
    pos_threshold = cfg["positive_margin_threshold"]
    neg_threshold = cfg["negative_margin_threshold"]
    negative_id, neutral_id, positive_id = (
        cfg["negative_class_id"], cfg["neutral_class_id"], cfg["positive_class_id"]
    )

    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    argmax_preds = probs.argmax(axis=1)
    final_preds = argmax_preds.copy()

    positive_uncertain = (argmax_preds == positive_id) & (margins < pos_threshold)
    final_preds[positive_uncertain] = neutral_id
    negative_uncertain = (argmax_preds == negative_id) & (margins < neg_threshold)
    final_preds[negative_uncertain] = neutral_id

    return final_preds, margins


def assign_issue_cluster(embedding: np.ndarray, category: str):
    """
    Nearest-centroid lookup. Only meaningful for negative-sentiment reviews
    (that's what the clusters were built from) - caller is responsible for
    only invoking this when sentiment == negative.
    """
    cfg = CONFIG["issue_detection"]
    
    # Backward compatibility with old config structure
    has_new_config = "per_category_clusters" in cfg
    
    if has_new_config and category in cfg["per_category_clusters"] and category in issue_centroids:
        centroids = issue_centroids[category]
        cluster_source = "per_category"
        cluster_names = cfg["per_category_clusters"][category]["cluster_names"]
        distance_threshold = cfg["per_category_clusters"][category].get("distance_threshold", cfg.get("distance_threshold"))
    else:
        centroids = issue_centroids.get("cross_category_fallback", list(issue_centroids.values())[0] if issue_centroids else np.empty((0, 384)))
        cluster_source = "cross_category_fallback"
        if has_new_config:
            cluster_names = cfg["fallback_clusters"]["cluster_names"]
            distance_threshold = cfg["fallback_clusters"].get("distance_threshold", cfg.get("distance_threshold"))
        else:
            cluster_names = cfg.get("cluster_names", {})
            distance_threshold = cfg.get("distance_threshold")
            
    if centroids.shape[0] == 0:
        return cfg.get("fallback_label", "other"), 0.0, cluster_source

    distances = np.linalg.norm(centroids - embedding, axis=1)
    nearest_idx = int(distances.argmin())
    nearest_distance = float(distances[nearest_idx])

    if distance_threshold is not None and nearest_distance > distance_threshold:
        return cfg.get("fallback_label", "other"), nearest_distance, cluster_source

    cluster_name = cluster_names.get(str(nearest_idx), cfg.get("fallback_label", "other"))
    return cluster_name, nearest_distance, cluster_source


def lambda_handler(event, context):
    """
    Expects: {"texts": ["review 1", "review 2", ...], "categories": ["cat 1", "cat 2", ...]}
    Preprocessing (text_preprocessing.py) must run BEFORE texts reach here.
    """
    t_handler_start = time.perf_counter()
    texts = event.get("texts", [])
    categories = event.get("categories", [])
    if not texts:
        return {"statusCode": 400, "body": json.dumps({"error": "no texts provided"})}

    embeddings = embed_texts(texts)

    sentiment_probs = mlp_forward(embeddings)
    sentiment_preds, margins = apply_asymmetric_threshold(sentiment_probs)

    label_names = CONFIG["sentiment"]["label_names"]
    negative_id = CONFIG["sentiment"]["negative_class_id"]

    t_issue_start = time.perf_counter()
    results = []
    num_negative_reviews = 0
    for i, text in enumerate(texts):
        sentiment_label = label_names[sentiment_preds[i]]
        category = categories[i] if i < len(categories) else ""

        # issue detection ONLY runs for negative sentiment - this is a
        # deliberate product decision (positive/neutral reviews don't
        # have "issues" to categorize), not an oversight
        issue_tag = None
        issue_distance = None
        cluster_source = None
        if sentiment_preds[i] == negative_id:
            num_negative_reviews += 1
            issue_tag, issue_distance, cluster_source = assign_issue_cluster(embeddings[i], category)

        results.append({
            "text": text,
            "sentiment": sentiment_label,
            "sentiment_confidence_margin": float(margins[i]),
            "sentiment_probabilities": {
                label_names[j]: float(p) for j, p in enumerate(sentiment_probs[i])
            },
            "issue_tag": issue_tag,
            "issue_distance": issue_distance,
            "cluster_source": cluster_source,
        })
    t_issue_end = time.perf_counter()
    duration_issue_detection_ms = (t_issue_end - t_issue_start) * 1000
    logger.info("issue_detection", extra={
        "duration_ms": duration_issue_detection_ms,
        "num_negative_reviews": num_negative_reviews,
    })

    t_handler_end = time.perf_counter()
    duration_handler_ms = (t_handler_end - t_handler_start) * 1000
    logger.info("lambda_handler", extra={
        "duration_ms": duration_handler_ms,
        "batch_size": len(texts),
    })

    return {"statusCode": 200, "body": json.dumps({"results": results})}


if __name__ == "__main__":
    test_event = {
        "texts": [
            "this broke after two days, complete waste of money",
            "works exactly as described, very happy with it",
            "it's fine, does the job, nothing special",
        ]
    }
    response = lambda_handler(test_event, None)
    print(json.dumps(json.loads(response["body"]), indent=2))
    print("\nSanity check: does sentiment match what you'd expect for each")
    print("example above? Does the negative one get a real issue_tag (not")
    print("null, not always 'other')? If either looks wrong, do NOT package")
    print("this for Lambda yet - debug locally first, it's much faster to")
    print("iterate here than through a Lambda deploy cycle.")
