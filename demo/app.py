"""
Flask backend for CLIP-GPT2 Image Captioning Demo
IE7615 Group 8 — Spring 2026

Usage (from project root, inside venv):
    source .venv/bin/activate
    python demo/app.py

    Or use the launcher script (no manual activation needed):
    demo/run.sh

Endpoints:
    GET  /              -> serves index.html
    POST /api/caption   -> accepts base64 image, returns captions + metadata
    GET  /api/status    -> returns model load status
    POST /api/evaluate  -> BLEU / METEOR / ROUGE-L / Distinct-2 / Self-BLEU
    POST /api/no_gt_metrics -> CLIPScore, perplexity, grammar (reference-free)
    POST /api/sensitivity   -> beam-width + temperature sweep
"""

# ── Step 1: set HF offline env vars BEFORE importing transformers ─────────────
import os
import sys
from pathlib import Path

_DEMO_DIR_EARLY     = Path(__file__).parent.resolve()
_PROJECT_ROOT_EARLY = _DEMO_DIR_EARLY.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT_EARLY))

_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

def _is_cached(model_id: str) -> bool:
    snap_dir = _HF_CACHE / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not snap_dir.exists():
        return False
    for snap in snap_dir.iterdir():
        if snap.is_dir() and (
            any(snap.glob("*.bin")) or any(snap.glob("*.safetensors"))
        ):
            return True
    return False

if _is_cached("openai/clip-vit-base-patch32") and _is_cached("gpt2"):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"]   = "1"
    os.environ["HF_HUB_OFFLINE"]        = "1"
    print("[demo] HF cache found — offline mode ON")
else:
    print("[demo] Cache incomplete — will download missing models on first run")

# ── Step 2: safe to import transformers now ───────────────────────────────────
import base64
import re
import io
import logging
import time
import types
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer

# ── Project paths & logging ───────────────────────────────────────────────────
DEMO_DIR     = _DEMO_DIR_EARLY
PROJECT_ROOT = _PROJECT_ROOT_EARLY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")
logger.info(f"Offline: {os.environ.get('TRANSFORMERS_OFFLINE') == '1'}")

# ── Constants ─────────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
GPT2_MODEL_NAME = "gpt2"
PREFIX_LENGTH   = 10
CLIP_DIM        = 512
GPT2_DIM        = 768
MAX_NEW_TOKENS  = 30

CHECKPOINTS = {
    "frozen": PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pt",
    "ft4l":   PROJECT_ROOT / "outputs" / "checkpoints" / "best_model_finetuned.pt",
    "lora":   PROJECT_ROOT / "outputs" / "checkpoints" / "best_model_lora.pt",
}

# ── Device ────────────────────────────────────────────────────────────────────
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
logger.info(f"Device: {DEVICE}")

# ── TrainingConfig stub ───────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """Stub that matches the fields used in milestone2 checkpoints."""
    clip_model_name:    str   = CLIP_MODEL_NAME
    gpt2_model_name:    str   = GPT2_MODEL_NAME
    clip_embedding_dim: int   = CLIP_DIM
    gpt2_embedding_dim: int   = GPT2_DIM
    prefix_length:      int   = PREFIX_LENGTH
    max_token_length:   int   = 50
    learning_rate:      float = 5e-4
    batch_size:         int   = 16
    epochs:             int   = 20
    warmup_ratio:       float = 0.1
    weight_decay:       float = 0.01
    beam_width:         int   = 5
    nucleus_top_p:      float = 0.9
    nucleus_temperature: float = 0.8
    early_stopping_patience: int = 3
    log_interval:       int   = 50
    seed:               int   = 42
    device:             str   = DEVICE
    project_root:       str   = str(PROJECT_ROOT)
    data_raw_dir:       str   = ""
    data_processed_dir: str   = ""
    embeddings_dir:     str   = ""
    checkpoints_dir:    str   = str(PROJECT_ROOT / "outputs" / "checkpoints")
    samples_dir:        str   = str(PROJECT_ROOT / "outputs" / "samples")

_main_module = sys.modules.get("__main__", types.ModuleType("__main__"))
if not hasattr(_main_module, "TrainingConfig"):
    _main_module.TrainingConfig = TrainingConfig
    sys.modules["__main__"] = _main_module

# ── Model architecture ────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, clip_dim=512, gpt2_dim=768, prefix_length=10):
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt2_dim = gpt2_dim
        hidden = gpt2_dim * prefix_length
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).view(-1, self.prefix_length, self.gpt2_dim)


class ClipCaptionModel(nn.Module):
    """CLIP prefix conditioning + GPT-2 decoder."""
    def __init__(self, gpt2_model_name: str, tokenizer, prefix_length: int = 10):
        super().__init__()
        self.prefix_length = prefix_length
        self.projection = ProjectionHead(CLIP_DIM, GPT2_DIM, prefix_length)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.gpt2.resize_token_embeddings(len(tokenizer))

    @torch.no_grad()
    def generate(
        self,
        embedding: torch.Tensor,
        strategy: str = "beam",
        num_beams: int = 5,
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_length: int = 40,
    ) -> str:
        self.eval()
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(next(self.parameters()).device)
        prefix = self.projection(embedding)

        gen_kwargs: dict = dict(
            inputs_embeds=prefix,
            max_new_tokens=max_length,
            min_new_tokens=3,
            pad_token_id=tokenizer_global.pad_token_id,
            eos_token_id=tokenizer_global.eos_token_id,
            no_repeat_ngram_size=3,
        )
        if strategy == "greedy":
            gen_kwargs.update(do_sample=False, repetition_penalty=1.5)
        elif strategy == "beam":
            gen_kwargs.update(
                do_sample=False,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0,
            )
        elif strategy == "nucleus":
            gen_kwargs.update(
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                top_k=0,
                repetition_penalty=1.3,
            )

        output_ids = self.gpt2.generate(**gen_kwargs)
        raw = tokenizer_global.decode(output_ids[0], skip_special_tokens=True).strip()
        return raw

    @torch.no_grad()
    def generate_with_scores(
        self,
        embedding: torch.Tensor,
        num_beams: int = 8,
        max_length: int = 40,
    ) -> list:
        """Return top-3 beam hypotheses with log-probability scores."""
        self.eval()
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(next(self.parameters()).device)
        prefix = self.projection(embedding)

        output = self.gpt2.generate(
            inputs_embeds=prefix,
            max_new_tokens=max_length,
            num_beams=num_beams,
            num_return_sequences=min(3, num_beams),
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer_global.pad_token_id,
            eos_token_id=tokenizer_global.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        scores = getattr(output, "sequences_scores", [None] * len(output.sequences))
        return [
            {
                "text": tokenizer_global.decode(seq, skip_special_tokens=True).strip(),
                "score": round(float(sc), 4) if sc is not None else None,
            }
            for seq, sc in zip(output.sequences, scores)
        ]


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _truncate_words(text, max_words):
    words = text.split()
    if len(words) <= max_words:
        return text
    t = " ".join(words[:max_words])
    return t if t[-1] in ".!?" else t.rstrip(",;:") + "..."


def _clean_caption(text, num_sentences=1, max_words=None):
    text = text.strip()
    if not text:
        return text
    if num_sentences and num_sentences > 0:
        sents = _split_sentences(text)
        if sents:
            text = " ".join(sents[:num_sentences])
            if text[-1] not in ".!?":
                text += "."
        elif len(text) > 120:
            t = text[:120]
            sp = t.rfind(" ")
            text = (t[:sp].strip() if sp > 40 else t) + "."
    if max_words and max_words > 0:
        text = _truncate_words(text, max_words)
    return text.strip()


# ── Module-level globals ──────────────────────────────────────────────────────
tokenizer_global = None
clip_processor   = None
clip_model_obj   = None
models           = {}
status           = {"ready": False, "message": "Starting up..."}


def load_all_models() -> None:
    global tokenizer_global, clip_processor, clip_model_obj, models, status

    _g = globals()
    if 'status'           not in _g or _g['status']           is None: _g['status']           = {"ready": False, "message": "Initializing..."}
    if 'models'           not in _g or _g['models']           is None: _g['models']           = {}
    if 'tokenizer_global' not in _g or _g['tokenizer_global'] is None: _g['tokenizer_global'] = None
    if 'clip_processor'   not in _g or _g['clip_processor']   is None: _g['clip_processor']   = None
    if 'clip_model_obj'   not in _g or _g['clip_model_obj']   is None: _g['clip_model_obj']   = None

    try:
        logger.info("Loading GPT-2 tokenizer...")
        tokenizer_global = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
        tokenizer_global.pad_token = tokenizer_global.eos_token

        logger.info("Loading CLIP ViT-B/32...")
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model_obj = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        clip_model_obj.eval()
        for p in clip_model_obj.parameters():
            p.requires_grad = False
        logger.info("CLIP loaded")

        import copy
        logger.info("Pre-loading GPT-2 base weights (shared across all checkpoints)...")
        t_base = time.time()
        _base_model_template = ClipCaptionModel(GPT2_MODEL_NAME, tokenizer_global, PREFIX_LENGTH)
        _base_state = _base_model_template.state_dict()
        logger.info(f"  GPT-2 base ready in {time.time() - t_base:.1f}s")

        for key, ckpt_path in CHECKPOINTS.items():
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint not found, skipping: {ckpt_path}")
                continue

            logger.info(f"Loading {key} from {ckpt_path.name}...")
            t0 = time.time()
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            state = {k.replace("module.", ""): v for k, v in state.items()}

            has_base_model = any('base_model.model.' in k for k in state)
            has_base_layer = any('base_layer.' in k for k in state)
            has_lora_keys  = any('lora_A' in k or 'lora_B' in k for k in state)

            if has_base_model or has_base_layer:
                clean = {}
                for k, v in state.items():
                    k2 = k
                    k2 = k2.replace('gpt2.base_model.model.', 'gpt2.')
                    if 'lora_A' in k2 or 'lora_B' in k2:
                        continue
                    k2 = k2.replace('.base_layer.', '.')
                    clean[k2] = v
                state = clean
                logger.info(f"  {key}: PEFT format -> remapped {len(state)} keys")
            elif has_lora_keys:
                state = {k: v for k, v in state.items()
                         if 'lora_A' not in k and 'lora_B' not in k
                         and 'base_layer' not in k}
                logger.info(f"  {key}: inline LoRA -> kept {len(state)} base keys")
            else:
                logger.info(f"  {key}: standard format -> {len(state)} keys")

            merged_state = dict(_base_state)
            merged_state.update(state)

            import copy
            m = copy.deepcopy(_base_model_template)
            missing, unexpected = m.load_state_dict(merged_state, strict=False)
            real_missing    = [k for k in missing    if 'lora' not in k.lower()]
            real_unexpected = [k for k in unexpected if 'lora' not in k.lower()
                               and 'base_layer' not in k]
            if real_missing:
                logger.warning(f"  {key}: {len(real_missing)} missing -> {real_missing[:2]}")
            if real_unexpected:
                logger.warning(f"  {key}: {len(real_unexpected)} unexpected -> {real_unexpected[:2]}")
            if not real_missing and not real_unexpected:
                logger.info(f"  {key}: weights loaded cleanly")
            m.to(DEVICE).eval()
            models[key] = m
            logger.info(f"  {key} ready in {time.time() - t0:.1f}s")

        if models:
            loaded = list(models.keys())
            status.update(
                ready=True,
                message=f"Ready — {loaded}",
                device=DEVICE,
                dataset="coco_full",
                n_train=94000,
                models_loaded=loaded,
            )
            logger.info(f"All models loaded: {loaded}")
        else:
            status.update(ready=False, message="No checkpoints found", device=DEVICE)

    except Exception:
        status.update(ready=False, message="Load error — see server log")
        logger.exception("Model load failed")


# ── CLIP encode ───────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_image(pil_img: Image.Image) -> torch.Tensor:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    inputs = clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    vision_out = clip_model_obj.vision_model(pixel_values=inputs["pixel_values"])
    pooled = vision_out.pooler_output
    emb = clip_model_obj.visual_projection(pooled)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu()


# ── Normalization metadata ────────────────────────────────────────────────────
def normalization_meta(pil_img: Image.Image) -> dict:
    w, h = pil_img.size
    scale = 224 / min(w, h)
    rw, rh = round(w * scale), round(h * scale)
    cx, cy = (rw - 224) // 2, (rh - 224) // 2
    reasons = []
    if w < 64 or h < 64:                         reasons.append(f"Very small: {w}x{h}")
    if w > 4096 or h > 4096:                     reasons.append(f"Very large: {w}x{h}")
    if max(w, h) / max(min(w, h), 1) > 2.5:      reasons.append(f"Extreme ratio {max(w,h)/max(min(w,h),1):.1f}:1")
    return {
        "original": f"{w}x{h}",
        "resized":  f"{rw}x{rh}",
        "crop":     f"{cx},{cy}+224",
        "hard":     len(reasons) > 0,
        "reasons":  reasons,
    }


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(DEMO_DIR))


# ── Reference-free metrics endpoint ──────────────────────────────────────────
# ── Diversity metrics endpoint (Distinct-2, Self-BLEU) ──────────────────────
@app.route("/api/diversity_metrics", methods=["POST"])
def diversity_metrics():
    """
    Compute Distinct-2 and Self-BLEU for a list of captions.
    Body: { "captions": [str, ...] }
    Returns: { distinct2: float, self_bleu: float, n: int }

    Distinct-2 = unique bigrams / total bigrams across all captions.
    Self-BLEU  = average BLEU-1 of each caption against all others.
                 Lower Self-BLEU = more diverse captions.
    """
    data = request.get_json(force=True) or {}
    captions = [c.strip().lower() for c in data.get("captions", []) if c.strip()]

    if len(captions) < 2:
        return jsonify({"error": "need at least 2 captions"}), 400

    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    for resource in ["punkt"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    tokenized = [nltk.word_tokenize(c) for c in captions]

    # Distinct-2
    all_bigrams, unique_bigrams = [], set()
    for tokens in tokenized:
        for i in range(len(tokens) - 1):
            bg = (tokens[i], tokens[i+1])
            all_bigrams.append(bg)
            unique_bigrams.add(bg)
    distinct2 = len(unique_bigrams) / max(len(all_bigrams), 1)

    # Self-BLEU: each caption vs all others
    sf = SmoothingFunction().method1
    self_bleu_scores = []
    for i, hyp in enumerate(tokenized):
        refs = [t for j, t in enumerate(tokenized) if j != i]
        try:
            score = sentence_bleu(refs, hyp, weights=(1, 0, 0, 0), smoothing_function=sf)
            self_bleu_scores.append(score)
        except Exception:
            pass
    self_bleu = sum(self_bleu_scores) / max(len(self_bleu_scores), 1)

    return jsonify({
        "distinct2":  round(distinct2, 4),
        "self_bleu":  round(self_bleu,  4),
        "n_captions": len(captions),
        "n_bigrams":  len(all_bigrams),
        "n_unique":   len(unique_bigrams),
    })


@app.route("/api/no_gt_metrics", methods=["POST"])
def no_gt_metrics():
    """Reference-free metrics: CLIPScore, CLIP text consistency,
    Perplexity (GPT-2), Grammar score (heuristic)."""
    import math
    from io import BytesIO
    data = request.get_json(force=True)
    img_b64   = data.get("image", "")
    captions  = data.get("captions", [])
    if not captions:
        return jsonify({"error": "no captions"}), 400

    results = []
    for cap in captions:
        entry = {"caption": cap, "clip_score": None,
                 "sbert_sim": None, "perplexity": None, "grammar_score": None}

        # CLIPScore: cos_sim(image_embed, text_embed) * 100
        try:
            from transformers import CLIPProcessor, CLIPModel
            if not hasattr(app, "_clip_for_score"):
                app._clip_for_score = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                app._clip_proc_score = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                app._clip_for_score.eval()
            mdl  = app._clip_for_score
            proc = app._clip_proc_score
            raw = base64.b64decode(img_b64.split(",")[-1])
            pil = Image.open(BytesIO(raw)).convert("RGB")
            inputs = proc(text=[cap], images=pil, return_tensors="pt",
                          padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                out  = mdl(**inputs)
                i_e  = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
                t_e  = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
                sim  = (i_e * t_e).sum().item()
            entry["clip_score"] = round(max(0.0, sim) * 100, 1)
        except Exception as e:
            entry["clip_score"] = None

        # Perplexity via GPT-2 (lower = more fluent)
        try:
            if not hasattr(app, "_ppl_tok"):
                app._ppl_tok = GPT2Tokenizer.from_pretrained("gpt2")
                app._ppl_tok.pad_token = app._ppl_tok.eos_token
                app._ppl_mdl = GPT2LMHeadModel.from_pretrained("gpt2")
                app._ppl_mdl.eval()
            ids = app._ppl_tok.encode(cap, return_tensors="pt")
            with torch.no_grad():
                loss = app._ppl_mdl(ids, labels=ids).loss
            entry["perplexity"] = round(math.exp(loss.item()), 1)
        except Exception:
            entry["perplexity"] = None

        # CLIP text consistency: cosine sim among 3 paraphrases
        try:
            from transformers import CLIPTokenizer
            if clip_model_obj is None:
                raise RuntimeError("CLIP not loaded yet")
            if not hasattr(app, "_clip_tokenizer"):
                app._clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            tokenizer_clip = app._clip_tokenizer
            templates = [
                "a photo of " + cap,
                "an image showing " + cap,
                "this picture shows " + cap,
            ]
            enc = tokenizer_clip(templates, return_tensors="pt",
                                 padding=True, truncation=True, max_length=77)
            enc_cpu = {k: v.to("cpu") for k, v in enc.items()}
            text_model = clip_model_obj.text_model.to("cpu")
            proj       = clip_model_obj.text_projection.to("cpu")
            with torch.no_grad():
                out        = text_model(**enc_cpu)
                text_feats = proj(out.pooler_output)
            clip_model_obj.text_model.to(DEVICE)
            clip_model_obj.text_projection.to(DEVICE)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            sims = []
            n = text_feats.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    sims.append((text_feats[i] * text_feats[j]).sum().item())
            entry["sbert_sim"] = round(sum(sims) / max(len(sims), 1), 3)
        except Exception as e:
            logger.warning(f"CLIP text consistency error: {e}")
            entry["sbert_sim"] = None

        # Grammar score: heuristic
        try:
            tokens = cap.lower().split()
            n = len(tokens)
            score = 100.0
            if n < 4:
                score -= 20
            verb_hints = {"is","are","was","were","has","have","playing",
                          "sitting","standing","holding","walking","running",
                          "eating","looking","wearing","riding","driving"}
            if not any(t in verb_hints for t in tokens):
                score -= 15
            if len(set(tokens)) / max(n, 1) < 0.6:
                score -= 10
            if any(re.search(r"http|www|\d{4,}", t) for t in tokens):
                score -= 25
            entry["grammar_score"] = round(max(0, min(100, score)), 0)
        except Exception:
            entry["grammar_score"] = None

        results.append(entry)

    return jsonify({"results": results})


# ── Serve index.html ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    resp = send_from_directory(str(DEMO_DIR), "index.html")
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route("/images/<path:filename>")
def serve_image(filename):
    images_dir = DEMO_DIR / "images"
    return send_from_directory(str(images_dir), filename)


@app.route("/api/status")
def api_status():
    return jsonify(status)


# ── Caption generation endpoint ───────────────────────────────────────────────
@app.route("/api/caption", methods=["POST"])
def api_caption():
    if not status["ready"]:
        return jsonify({"error": "Models not ready", "status": status}), 503

    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400

    try:
        img_b64 = data["image"]
        if img_b64.startswith("data:"):
            img_b64 = img_b64.split(",", 1)[1]
        pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    model_key     = data.get("model", "lora")
    strategy      = data.get("strategy", "beam")
    beam_width    = int(data.get("beam_width", 5))
    temperature   = float(data.get("temperature", 0.5))
    top_p         = float(data.get("top_p", 0.9))
    num_sentences = int(data.get("num_sentences", 1))
    max_words_raw = int(data.get("max_words", 0))
    _tok_map = {0: 120, 1: 40, 2: 90, 3: 130}
    max_tokens = _tok_map.get(num_sentences, 40)
    if max_words_raw > 0:
        max_tokens = max(max_tokens, min(150, int(max_words_raw * 1.5)))

    # Thumbnail for display in other tabs
    thumb_b64 = ""
    try:
        thumb = pil_img.copy()
        thumb.thumbnail((120, 120))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=70)
        thumb_b64 = "data:image/jpeg;base64," + __import__('base64').b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    meta = normalization_meta(pil_img)

    t_clip = time.time()
    embedding = encode_image(pil_img)
    clip_ms = round((time.time() - t_clip) * 1000)

    results = []
    t_gen = time.time()

    model_keys = list(models.keys()) if model_key == "all" else [model_key]
    strategies = ["beam", "nucleus", "greedy"] if strategy == "all" else [strategy]

    for mk in model_keys:
        m = models.get(mk)
        if m is None:
            for st in strategies:
                results.append({"model": mk, "strategy": st,
                                 "caption": f"[{mk} not loaded]"})
            continue
        for st in strategies:
            try:
                raw_cap = m.generate(
                    embedding,
                    strategy=st,
                    num_beams=beam_width,
                    temperature=temperature,
                    top_p=top_p,
                    max_length=max_tokens,
                )
                caption = _clean_caption(
                    raw_cap,
                    num_sentences=num_sentences,
                    max_words=max_words_raw if max_words_raw > 0 else None,
                )
                results.append({"model": mk, "strategy": st, "caption": caption})
            except Exception as e:
                logger.exception(f"Generate failed for {mk}/{st}")
                results.append({"model": mk, "strategy": st,
                                 "caption": f"[Error: {e}]"})

    gen_ms = round((time.time() - t_gen) * 1000)

    # Top-3 beam hypotheses
    hypotheses = []
    best_key = "lora" if "lora" in models else (model_keys[0] if model_keys else None)
    if best_key and (meta["hard"] or strategy in ("beam", "all")):
        try:
            hypotheses = models[best_key].generate_with_scores(
                embedding, num_beams=max(beam_width, 5)
            )
        except Exception:
            hypotheses = []

    return jsonify({
        "results":    results,
        "hypotheses": hypotheses,
        "meta":       meta,
        "timing":     {"clip_ms": clip_ms, "gen_ms": gen_ms},
        "device":     DEVICE,
        "thumb":      thumb_b64,
    })


# ── Evaluation endpoint (with Distinct-2 and Self-BLEU) ──────────────────────
@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """
    Compute BLEU-1, BLEU-4, METEOR, ROUGE-L, Distinct-2, Self-BLEU
    for a single generated caption against one or more references.

    Distinct-2: ratio of unique bigrams in the hypothesis.
    Self-BLEU:  when multiple captions are provided in 'all_hypotheses',
                compute average BLEU of each caption against the others.
                If only one caption provided, Self-BLEU is reported as N/A.

    Body: {
        "hypothesis": str,
        "references": [str, ...],
        "all_hypotheses": [str, ...]   (optional, for Self-BLEU)
    }
    """
    data = request.get_json(force=True) or {}
    hypothesis    = data.get("hypothesis", "").strip().lower()
    references    = [r.strip().lower() for r in data.get("references", []) if r.strip()]
    all_hyps_raw  = data.get("all_hypotheses", [])
    all_hyps      = [h.strip().lower() for h in all_hyps_raw if h.strip()]

    if not hypothesis:
        return jsonify({"error": "hypothesis is required"}), 400
    if not references:
        return jsonify({"error": "at least one reference is required"}), 400

    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import single_meteor_score
    from rouge_score import rouge_scorer

    for resource in ["punkt", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    hyp_tokens = nltk.word_tokenize(hypothesis)
    ref_tokens = [nltk.word_tokenize(r) for r in references]
    sf = SmoothingFunction().method1

    bleu1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(.25,.25,.25,.25), smoothing_function=sf)

    meteor_scores = []
    for ref in references:
        try:
            s = single_meteor_score(nltk.word_tokenize(ref), hyp_tokens)
            meteor_scores.append(s)
        except Exception:
            pass
    meteor = max(meteor_scores) if meteor_scores else 0.0

    scorer_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = [scorer_rouge.score(ref, hypothesis)["rougeL"].fmeasure for ref in references]
    rouge_l = max(rouge_l_scores) if rouge_l_scores else 0.0

    # Distinct-2: unique bigrams / total bigrams in hypothesis
    def compute_distinct2(tokens):
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        if not bigrams:
            return 0.0
        return round(len(set(bigrams)) / len(bigrams), 4)

    distinct2 = compute_distinct2(hyp_tokens)

    # Self-BLEU: average BLEU of this hypothesis against all other hypotheses
    self_bleu = None
    if len(all_hyps) > 1:
        others = [h for h in all_hyps if h != hypothesis]
        if others:
            other_tokens = [nltk.word_tokenize(h) for h in others]
            self_bleu = round(
                sentence_bleu(other_tokens, hyp_tokens,
                              weights=(.25,.25,.25,.25), smoothing_function=sf),
                4
            )

    return jsonify({
        "bleu1":      round(bleu1,   4),
        "bleu4":      round(bleu4,   4),
        "meteor":     round(meteor,  4),
        "rouge_l":    round(rouge_l, 4),
        "distinct2":  distinct2,
        "self_bleu":  self_bleu,
        "hypothesis": hypothesis,
        "n_references": len(references),
    })


# ── Real-time sensitivity sweep endpoint ──────────────────────────────────────
@app.route("/api/sensitivity", methods=["POST"])
def api_sensitivity():
    """
    Run beam-width sweep and temperature sweep on the given image.
    Body: {
        "image": base64_str,
        "model": "lora"|"frozen"|"ft4l",
        "references": [str, ...]
    }
    Returns: { beam_results: [...], temp_results: [...] }
    """
    if not status["ready"]:
        return jsonify({"error": "Models not ready"}), 503

    data = request.get_json(force=True) or {}
    img_b64    = data.get("image", "")
    model_key  = data.get("model", "lora")
    references = [r.strip().lower() for r in data.get("references", []) if r.strip()]

    if not img_b64:
        return jsonify({"error": "image is required"}), 400

    try:
        raw = img_b64.split(",", 1)[1] if "," in img_b64 else img_b64
        pil_img = Image.open(io.BytesIO(base64.b64decode(raw)))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    embedding = encode_image(pil_img)

    m = models.get(model_key) or models.get("lora") or next(iter(models.values()), None)
    if m is None:
        return jsonify({"error": "No model loaded"}), 503

    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import single_meteor_score
    from rouge_score import rouge_scorer as rs_mod

    for resource in ["punkt", "wordnet"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    sf = SmoothingFunction().method1
    rouge = rs_mod.RougeScorer(["rougeL"], use_stemmer=True)

    def score_caption(cap: str) -> dict:
        if not references:
            return {"bleu4": None, "meteor": None, "rouge_l": None}
        hyp_tok  = nltk.word_tokenize(cap.lower())
        ref_toks = [nltk.word_tokenize(r) for r in references]
        b4  = sentence_bleu(ref_toks, hyp_tok, weights=(.25,.25,.25,.25), smoothing_function=sf)
        met_scores = []
        for ref in references:
            try:
                met_scores.append(single_meteor_score(nltk.word_tokenize(ref), hyp_tok))
            except Exception:
                pass
        met = max(met_scores) if met_scores else 0.0
        rl  = max(rouge.score(ref, cap.lower())["rougeL"].fmeasure for ref in references)
        return {"bleu4": round(b4, 4), "meteor": round(met, 4), "rouge_l": round(rl, 4)}

    BEAM_WIDTHS  = [1, 2, 3, 5, 8, 10]
    TEMPERATURES = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2]

    beam_results = []
    for w in BEAM_WIDTHS:
        try:
            cap = m.generate(embedding, strategy="beam", num_beams=w)
            scores = score_caption(cap)
            beam_results.append({"w": w, "caption": cap, **scores})
        except Exception as e:
            beam_results.append({"w": w, "caption": str(e),
                                  "bleu4": None, "meteor": None, "rouge_l": None})

    temp_results = []
    for t in TEMPERATURES:
        try:
            cap = m.generate(embedding, strategy="nucleus", temperature=t)
            scores = score_caption(cap)
            temp_results.append({"t": t, "caption": cap, **scores})
        except Exception as e:
            temp_results.append({"t": t, "caption": str(e),
                                  "bleu4": None, "meteor": None, "rouge_l": None})

    return jsonify({
        "model":        model_key,
        "beam_results": beam_results,
        "temp_results": temp_results,
        "has_reference": len(references) > 0,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("IE7615 Group 8 -- CLIP+GPT-2 Caption Demo Server")
    logger.info(f"Project root : {PROJECT_ROOT}")
    logger.info(f"Device       : {DEVICE}")
    logger.info(f"Offline mode : {os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'}")
    logger.info("=" * 60)

    import threading

    def _load_bg():
        load_all_models()
        import webbrowser as _wb
        _wb.open("http://127.0.0.1:5000")

    threading.Thread(target=_load_bg, daemon=True).start()

    logger.info("Flask starting — models loading in background")
    logger.info("Browser will open automatically when models are ready")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
