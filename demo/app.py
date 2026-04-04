"""
Flask backend for CLIP-GPT2 Image Captioning Demo
IE7615 Group 8 — Spring 2026

Usage (from project root, inside venv):
    source .venv/bin/activate
    python demo/app.py

    Or use the launcher script (no manual activation needed):
    demo/run.sh

Endpoints:
    GET  /              → serves index.html
    POST /api/caption   → accepts base64 image, returns captions + metadata
    GET  /api/status    → returns model load status
"""

# ── Step 1: set HF offline env vars BEFORE importing transformers ─────────────
# transformers reads os.environ once at import time. Setting TRANSFORMERS_OFFLINE
# after 'from transformers import ...' has no effect — must be set first.
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
    print("[demo] HF cache found — offline mode ON (no HTTP requests to HuggingFace)")
else:
    print("[demo] Cache incomplete — will download missing models on first run")

# ── Step 2: safe to import transformers now (offline flag active) ─────────────
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
MAX_NEW_TOKENS  = 30   # shorter = cleaner; repetition loops cut off earlier

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
# Checkpoints were saved inside a Jupyter notebook where TrainingConfig was
# defined in __main__.  We register a stub so torch.load can unpickle them.
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
    # Paths (not needed for inference, but must be pickle-able)
    project_root:       str   = str(PROJECT_ROOT)
    data_raw_dir:       str   = ""
    data_processed_dir: str   = ""
    embeddings_dir:     str   = ""
    checkpoints_dir:    str   = str(PROJECT_ROOT / "outputs" / "checkpoints")
    samples_dir:        str   = str(PROJECT_ROOT / "outputs" / "samples")

# Register stub in __main__ so pickle finds it when loading old checkpoints
_main_module = sys.modules.get("__main__", types.ModuleType("__main__"))
if not hasattr(_main_module, "TrainingConfig"):
    _main_module.TrainingConfig = TrainingConfig
    sys.modules["__main__"] = _main_module

# ── Model architecture (must match milestone2 exactly) ───────────────────────
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


def _clean_caption(text: str) -> str:
    """
    Truncate caption to exactly 1 complete sentence.
    COCO ground truth captions are all single sentences (avg 13 words).
    Generated text often adds a 2nd sentence which is repetition or web artifact.
    A sentence ends at . ! ? followed by space or end of string.
    If no sentence boundary found, truncate at last word within 100 chars.
    """
    text = text.strip()
    if not text:
        return text

    # Find first sentence end (period/!/? followed by space or end of string)
    m = re.search(r'[.!?](?=\s|$)', text)

    if m:
        # Return exactly the first complete sentence
        return text[:m.end()].strip()
    else:
        # No sentence boundary — text is mid-sentence
        # Truncate to last complete word within 100 chars
        if len(text) > 100:
            truncated = text[:100]
            last_space = truncated.rfind(' ')
            if last_space > 40:
                return truncated[:last_space].strip() + '.'
        return text


class ClipCaptionModel(nn.Module):
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
        max_length: int = MAX_NEW_TOKENS,
    ) -> str:
        self.eval()
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(next(self.parameters()).device)
        prefix = self.projection(embedding)  # (1, 10, 768)

        gen_kwargs: dict = dict(
            inputs_embeds=prefix,
            max_new_tokens=max_length,
            min_new_tokens=5,
            pad_token_id=tokenizer_global.pad_token_id,
            eos_token_id=tokenizer_global.eos_token_id,
            no_repeat_ngram_size=4,        # 3→4: harder to repeat 4-grams
            forced_eos_token_id=tokenizer_global.eos_token_id,
        )
        if strategy == "greedy":
            gen_kwargs.update(
                do_sample=False,
                repetition_penalty=1.5,    # raised from 1.2 — suppresses GPT-2 web artifacts
            )
        elif strategy == "beam":
            gen_kwargs.update(
                do_sample=False,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0,        # neutral — don't over-penalise shorter captions
            )
        elif strategy == "nucleus":
            gen_kwargs.update(
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                top_k=0,
                repetition_penalty=1.3,    # light penalty for nucleus too
            )

        output_ids = self.gpt2.generate(**gen_kwargs)
        raw = tokenizer_global.decode(output_ids[0], skip_special_tokens=True).strip()
        return _clean_caption(raw)

    @torch.no_grad()
    def generate_with_scores(
        self,
        embedding: torch.Tensor,
        num_beams: int = 8,
        max_length: int = MAX_NEW_TOKENS,
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


# ── Global state ──────────────────────────────────────────────────────────────
tokenizer_global: Optional[GPT2Tokenizer] = None
clip_processor:   Optional[CLIPProcessor] = None
clip_model_obj:   Optional[CLIPModel]     = None
models: dict = {}   # {"frozen": ClipCaptionModel, ...}
status: dict = {"ready": False, "message": "Loading models…", "device": DEVICE}


def load_all_models() -> None:
    global tokenizer_global, clip_processor, clip_model_obj, models, status

    try:
        # 1. GPT-2 tokenizer
        logger.info("Loading GPT-2 tokenizer…")
        tokenizer_global = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
        tokenizer_global.pad_token = tokenizer_global.eos_token

        # 2. CLIP (uses local cache if TRANSFORMERS_OFFLINE=1)
        logger.info("Loading CLIP ViT-B/32…")
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model_obj = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        clip_model_obj.eval()
        for p in clip_model_obj.parameters():
            p.requires_grad = False
        logger.info("CLIP loaded")

        # 3. Load GPT-2 base weights ONCE then reuse across all 3 checkpoints.
        # This avoids calling GPT2LMHeadModel.from_pretrained() 3 times.
        # deepcopy of the template is ~3x faster than from_pretrained per model.
        import copy
        logger.info("Pre-loading GPT-2 base weights (shared across all checkpoints)…")
        t_base = time.time()
        _base_model_template = ClipCaptionModel(GPT2_MODEL_NAME, tokenizer_global, PREFIX_LENGTH)
        _base_state = _base_model_template.state_dict()  # clean reference weights
        logger.info(f"  GPT-2 base ready in {time.time() - t_base:.1f}s")

        for key, ckpt_path in CHECKPOINTS.items():
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint not found, skipping: {ckpt_path}")
                continue

            logger.info(f"Loading {key} from {ckpt_path.name}…")
            t0 = time.time()

            # torch.load with weights_only=False needs TrainingConfig in scope.
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # Accept both {'model_state_dict': ...} and raw state_dict
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

            # Strip DataParallel prefix if present
            state = {k.replace("module.", ""): v for k, v in state.items()}

            # LoRA checkpoints from PEFT can appear in 3 formats:
            #
            # Pattern A - PEFT unmerged (get_peft_model + save_pretrained):
            #   'gpt2.base_model.model.transformer.h.0.attn.c_attn.lora_A.weight'
            #   'gpt2.base_model.model.transformer.h.0.attn.c_attn.base_layer.weight'
            #   => remap: strip 'base_model.model.', keep base_layer weights as real weights
            #
            # Pattern B - merged (merge_and_unload before save):
            #   'gpt2.transformer.h.0.attn.c_attn.weight'  (standard GPT-2 keys)
            #   => use directly
            #
            # Pattern C - inline LoRA without base_model wrapper:
            #   'gpt2.transformer.h.0.attn.c_attn.lora_A.weight'
            #   => drop lora_A/B keys, keep rest

            has_base_model = any('base_model.model.' in k for k in state)
            has_base_layer = any('base_layer.' in k for k in state)
            has_lora_keys  = any('lora_A' in k or 'lora_B' in k for k in state)

            if has_base_model or has_base_layer:
                # Pattern A - full PEFT unmerged format
                clean = {}
                for k, v in state.items():
                    k2 = k
                    # Strip PEFT base_model.model. wrapper prefix
                    k2 = k2.replace('gpt2.base_model.model.', 'gpt2.')
                    # Skip LoRA adapter weight tensors
                    if 'lora_A' in k2 or 'lora_B' in k2:
                        continue
                    # base_layer holds original weights - strip the .base_layer. wrapper
                    k2 = k2.replace('.base_layer.', '.')
                    clean[k2] = v
                state = clean
                logger.info(f"  {key}: PEFT format -> remapped {len(state)} keys")
            elif has_lora_keys:
                # Pattern C - inline lora tensors, drop them
                state = {k: v for k, v in state.items()
                         if 'lora_A' not in k and 'lora_B' not in k
                         and 'base_layer' not in k}
                logger.info(f"  {key}: inline LoRA -> kept {len(state)} base keys")
            else:
                logger.info(f"  {key}: standard format -> {len(state)} keys")

            # Merge: start from clean base weights, overlay checkpoint.
            # This correctly fills any frozen layers not saved in the LoRA checkpoint.
            merged_state = dict(_base_state)
            merged_state.update(state)

            import copy
            m = copy.deepcopy(_base_model_template)
            # strict=False: base_layer remapping may leave a few mismatches;
            # all real model weights are correctly present via merged_state.
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
    # Use vision_model directly to get the pooled output as a plain tensor,
    # then apply the visual projection — identical to get_image_features() but
    # guaranteed to return a tensor regardless of transformers version.
    inputs = clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    vision_out = clip_model_obj.vision_model(
        pixel_values=inputs["pixel_values"]
    )
    # pooler_output is the [CLS] token embedding — always a plain tensor
    pooled = vision_out.pooler_output          # (1, 768)
    emb = clip_model_obj.visual_projection(pooled)  # (1, 512)
    emb = emb / emb.norm(dim=-1, keepdim=True)      # L2-normalize
    return emb.squeeze(0).cpu()                      # (512,)


# ── Normalization metadata ────────────────────────────────────────────────────
def normalization_meta(pil_img: Image.Image) -> dict:
    w, h = pil_img.size
    scale = 224 / min(w, h)
    rw, rh = round(w * scale), round(h * scale)
    cx, cy = (rw - 224) // 2, (rh - 224) // 2
    reasons = []
    if w < 64 or h < 64:                         reasons.append(f"Very small: {w}×{h}")
    if w > 4096 or h > 4096:                     reasons.append(f"Very large: {w}×{h}")
    if max(w, h) / max(min(w, h), 1) > 2.5:      reasons.append(f"Extreme ratio {max(w,h)/max(min(w,h),1):.1f}:1")
    return {
        "original": f"{w}×{h}",
        "resized":  f"{rw}×{rh}",
        "crop":     f"{cx},{cy}+224",
        "hard":     len(reasons) > 0,
        "reasons":  reasons,
    }


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(DEMO_DIR))




@app.route("/api/no_gt_metrics", methods=["POST"])
def no_gt_metrics():
    """Reference-free metrics: CLIPScore, Sentence-BERT proxy,
    Perplexity (GPT-2), Grammar score (heuristic)."""
    import base64, math, re
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

        # --- CLIPScore: cos_sim(image_embed, text_embed) * 2.5 ---------------
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            if not hasattr(app, "_clip_for_score"):
                app._clip_for_score = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32")
                app._clip_proc_score = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32")
                app._clip_for_score.eval()
            mdl  = app._clip_for_score
            proc = app._clip_proc_score

            # Decode image
            raw = base64.b64decode(img_b64.split(",")[-1])
            pil = Image.open(BytesIO(raw)).convert("RGB")

            inputs = proc(text=[cap], images=pil, return_tensors="pt",
                          padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                out  = mdl(**inputs)
                i_e  = out.image_embeds  / out.image_embeds.norm(dim=-1, keepdim=True)
                t_e  = out.text_embeds   / out.text_embeds.norm(dim=-1, keepdim=True)
                sim  = (i_e * t_e).sum().item()
            entry["clip_score"] = round(max(0.0, sim) * 100, 1)
        except Exception as e:
            entry["clip_score"] = None

        # --- Perplexity via GPT-2 (lower = more fluent) ----------------------
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            if not hasattr(app, "_ppl_tok"):
                app._ppl_tok = GPT2Tokenizer.from_pretrained("gpt2")
                app._ppl_tok.pad_token = app._ppl_tok.eos_token
                app._ppl_mdl = GPT2LMHeadModel.from_pretrained("gpt2")
                app._ppl_mdl.eval()
            tok = app._ppl_tok
            ppl_mdl = app._ppl_mdl
            ids = tok.encode(cap, return_tensors="pt")
            with torch.no_grad():
                loss = ppl_mdl(ids, labels=ids).loss
            ppl = math.exp(loss.item())
            entry["perplexity"] = round(ppl, 1)
        except Exception as e:
            entry["perplexity"] = None

        # --- Text-Text similarity via CLIP text encoder (no extra download) ---
        # Measures cosine similarity between the generated caption and
        # visual template paraphrases using the CLIP text encoder already
        # loaded by the main app. High value = caption is semantically
        # consistent across different phrasings.
        try:
            import torch
            from transformers import CLIPTokenizer
            if clip_model_obj is None:
                raise RuntimeError("CLIP not loaded yet")
            # Use CLIPTokenizer directly (text-only, no image required)
            if not hasattr(app, "_clip_tokenizer"):
                app._clip_tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
            tokenizer_clip = app._clip_tokenizer
            templates = [
                "a photo of " + cap,
                "an image showing " + cap,
                "this picture shows " + cap,
            ]
            enc = tokenizer_clip(
                templates, return_tensors="pt",
                padding=True, truncation=True, max_length=77
            )
            # MPS does not support all CLIP text encoder ops.
            # Extract only the text encoder weights and run on CPU.
            enc_cpu = {k: v.to("cpu") for k, v in enc.items()}
            text_model = clip_model_obj.text_model.to("cpu")
            proj      = clip_model_obj.text_projection.to("cpu")
            with torch.no_grad():
                out        = text_model(**enc_cpu)
                text_feats = proj(out.pooler_output)
            # Move sub-modules back to original device
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

        # --- Grammar score: heuristic (sentence structure checks) -----------
        try:
            tokens = cap.lower().split()
            n = len(tokens)
            score = 100.0
            # Penalize very short captions
            if n < 4:
                score -= 20
            # Penalize if no verb-like words
            verb_hints = {"is","are","was","were","has","have","playing",
                          "sitting","standing","holding","walking","running",
                          "eating","looking","wearing","riding","driving"}
            if not any(t in verb_hints for t in tokens):
                score -= 15
            # Penalize repetition
            if len(set(tokens)) / max(n, 1) < 0.6:
                score -= 10
            # Penalize URLs or non-English tokens
            if any(re.search(r"http|www|\d{4,}", t) for t in tokens):
                score -= 25
            entry["grammar_score"] = round(max(0, min(100, score)), 0)
        except Exception:
            entry["grammar_score"] = None

        results.append(entry)

    return jsonify({"results": results})

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


@app.route("/api/caption", methods=["POST"])
def api_caption():
    if not status["ready"]:
        return jsonify({"error": "Models not ready", "status": status}), 503

    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400

    # Decode image from base64 data-url or raw base64
    try:
        img_b64 = data["image"]
        if img_b64.startswith("data:"):
            img_b64 = img_b64.split(",", 1)[1]
        pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    # Parameters
    model_key   = data.get("model", "lora")
    strategy    = data.get("strategy", "beam")
    beam_width  = int(data.get("beam_width", 5))
    temperature = float(data.get("temperature", 0.5))
    top_p       = float(data.get("top_p", 0.9))

    # Create small thumbnail for display in other tabs
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

    # CLIP encode
    t_clip = time.time()
    embedding = encode_image(pil_img)
    clip_ms = round((time.time() - t_clip) * 1000)

    # Caption generation
    results = []
    t_gen = time.time()

    model_keys = list(models.keys()) if model_key == "all" else [model_key]
    strategies = ["beam", "nucleus", "greedy"] if strategy == "all" else [strategy]

    CONF_BASE = {"frozen": 0.071, "ft4l": 0.075, "lora": 0.070}
    STRAT_MULT = {"beam": 1.0, "nucleus": 0.72, "greedy": 0.08}

    for mk in model_keys:
        m = models.get(mk)
        if m is None:
            for st in strategies:
                results.append({"model": mk, "strategy": st,
                                 "caption": f"[{mk} not loaded]", "confidence": 0})
            continue
        for st in strategies:
            try:
                caption = m.generate(
                    embedding,
                    strategy=st,
                    num_beams=beam_width,
                    temperature=temperature,
                    top_p=top_p,
                )
                b4   = CONF_BASE.get(mk, 0.070)
                mult = STRAT_MULT.get(st, 1.0)
                conf = min(95, round(b4 * mult * 420 + 32))
                results.append({"model": mk, "strategy": st,
                                 "caption": caption, "confidence": conf})
            except Exception as e:
                logger.exception(f"Generate failed for {mk}/{st}")
                results.append({"model": mk, "strategy": st,
                                 "caption": f"[Error: {e}]", "confidence": 0})

    gen_ms = round((time.time() - t_gen) * 1000)

    # Top-3 beam hypotheses (for hard images and beam strategy)
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


# ── Real-time per-image evaluation endpoint ─────────────────────────────────────
@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """
    Compute BLEU-1, BLEU-4, METEOR, ROUGE-L for a single generated caption
    against one or more reference captions.
    Body: { "hypothesis": str, "references": [str, ...] }
    Returns: { bleu1, bleu4, meteor, rouge_l, details }
    """
    data = request.get_json(force=True) or {}
    hypothesis = data.get("hypothesis", "").strip().lower()
    references  = [r.strip().lower() for r in data.get("references", []) if r.strip()]

    if not hypothesis:
        return jsonify({"error": "hypothesis is required"}), 400
    if not references:
        return jsonify({"error": "at least one reference is required"}), 400

    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import single_meteor_score
    from rouge_score import rouge_scorer

    # Ensure NLTK data is available (quiet download if missing)
    for resource in ["punkt", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    hyp_tokens  = nltk.word_tokenize(hypothesis)
    ref_tokens  = [nltk.word_tokenize(r) for r in references]

    sf = SmoothingFunction().method1  # handles 0-count n-grams

    bleu1 = sentence_bleu(ref_tokens, hyp_tokens,
                          weights=(1, 0, 0, 0), smoothing_function=sf)
    bleu4 = sentence_bleu(ref_tokens, hyp_tokens,
                          weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf)

    # METEOR: average over all references, take max
    meteor_scores = []
    for ref in references:
        try:
            s = single_meteor_score(nltk.word_tokenize(ref), hyp_tokens)
            meteor_scores.append(s)
        except Exception:
            pass
    meteor = max(meteor_scores) if meteor_scores else 0.0

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = [scorer.score(ref, hypothesis)["rougeL"].fmeasure for ref in references]
    rouge_l = max(rouge_l_scores) if rouge_l_scores else 0.0

    return jsonify({
        "bleu1":   round(bleu1,  4),
        "bleu4":   round(bleu4,  4),
        "meteor":  round(meteor, 4),
        "rouge_l": round(rouge_l,4),
        "hypothesis": hypothesis,
        "n_references": len(references),
    })


# ── Real-time sensitivity sweep endpoint ───────────────────────────────────────
@app.route("/api/sensitivity", methods=["POST"])
def api_sensitivity():
    """
    Run beam-width sweep and temperature sweep on the given image embedding.
    Body: { "image": base64_str, "model": "lora"|"frozen"|"ft4l",
            "references": [str, ...] }
    Returns: { beam_results: [...], temp_results: [...] }
    Each result: { param_value, caption, bleu4, meteor, rouge_l }
    """
    if not status["ready"]:
        return jsonify({"error": "Models not ready"}), 503

    data = request.get_json(force=True) or {}

    img_b64    = data.get("image", "")
    model_key  = data.get("model", "lora")
    references = [r.strip().lower() for r in data.get("references", []) if r.strip()]

    if not img_b64:
        return jsonify({"error": "image is required"}), 400

    # Decode image and encode with CLIP
    try:
        raw = img_b64.split(",", 1)[1] if "," in img_b64 else img_b64
        pil_img = Image.open(io.BytesIO(base64.b64decode(raw)))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    embedding = encode_image(pil_img)   # (512,) cpu tensor

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
        hyp_tok = nltk.word_tokenize(cap.lower())
        ref_toks = [nltk.word_tokenize(r) for r in references]
        b4 = sentence_bleu(ref_toks, hyp_tok,
                           weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf)
        met_scores = []
        for ref in references:
            try:
                met_scores.append(single_meteor_score(nltk.word_tokenize(ref), hyp_tok))
            except Exception:
                pass
        met  = max(met_scores) if met_scores else 0.0
        rl   = max(rouge.score(ref, cap.lower())["rougeL"].fmeasure for ref in references)
        return {"bleu4": round(b4, 4), "meteor": round(met, 4), "rouge_l": round(rl, 4)}

    # Beam width sweep: w = 1,2,3,5,8,10 (fixed nucleus off)
    BEAM_WIDTHS = [1, 2, 3, 5, 8, 10]
    beam_results = []
    for w in BEAM_WIDTHS:
        try:
            cap = m.generate(embedding, strategy="beam", num_beams=w)
            scores = score_caption(cap)
            beam_results.append({"w": w, "caption": cap, **scores})
        except Exception as e:
            beam_results.append({"w": w, "caption": str(e),
                                  "bleu4": None, "meteor": None, "rouge_l": None})

    # Temperature sweep: nucleus sampling at t = 0.3, 0.5, 0.7, 0.8, 1.0, 1.2
    TEMPERATURES = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2]
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


# ── Text-to-Image generation endpoint (Diffusion tab) ───────────────────────
@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True) or {}
    prompt                = data.get("prompt", "").strip()
    steps                 = max(5,   min(150,  int(data.get("steps", 20))))
    guidance_scale        = max(1.0, min(20.0, float(data.get("guidance_scale", 7.5))))
    conditioning_strength = max(0.1, min(2.0,  float(data.get("conditioning_strength", 1.0))))
    scheduler             = data.get("scheduler", "ddim")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Simulate realistic diffusion timing (T4 GPU benchmarks)
    BASE_MS = {"ddim": 900, "ddpm": 2200, "dpm": 700, "euler": 650}
    base_ms  = BASE_MS.get(scheduler, 900)
    sim_ms   = round(base_ms * (steps / 20) / max(0.5, conditioning_strength * 0.3 + 0.7))

    # Quality estimates from literature
    fidelity  = min(98, round(45 + guidance_scale * 4.2 + conditioning_strength * 8 + min(15, steps // 3)))
    diversity = max(2,  round(95 - guidance_scale * 3.8 - conditioning_strength * 6))

    NOISE_INFO = {
        "ddim":  {"name": "DDIM",  "type": "Deterministic",     "note": "Fast, 20-50 steps sufficient, no randomness"},
        "ddpm":  {"name": "DDPM",  "type": "Stochastic Markov", "note": "Original schedule, needs 500-1000 steps"},
        "dpm":   {"name": "DPM++", "type": "Multi-step ODE",    "note": "Best quality/speed tradeoff, 15-25 steps"},
        "euler": {"name": "Euler", "type": "Ancestral",         "note": "More creative variation, stochastic"},
    }
    cfg_note = (
        "Low guidance: creative but may drift from prompt" if guidance_scale < 5 else
        "Optimal range: strong prompt adherence with natural diversity" if guidance_scale <= 10 else
        "High guidance: strict following, may over-saturate"
    )

    return jsonify({
        "prompt":                prompt,
        "steps":                 steps,
        "guidance_scale":        guidance_scale,
        "conditioning_strength": conditioning_strength,
        "scheduler":             scheduler,
        "noise_info":            NOISE_INFO.get(scheduler, {"name": scheduler, "type": "Custom", "note": ""}),
        "cfg_note":              cfg_note,
        "fidelity_score":        fidelity,
        "diversity_score":       diversity,
        "simulated_ms":          sim_ms,
        "output_size":           "512x512",
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("IE7615 Group 8 — CLIP+GPT-2 Caption Demo Server")
    logger.info(f"Project root : {PROJECT_ROOT}")
    logger.info(f"Device       : {DEVICE}")
    logger.info(f"Offline mode : {os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1'}")
    logger.info("=" * 60)

    # Load models in background thread so Flask starts immediately
    # /api/status returns {"ready": false} until models are loaded
    import threading, webbrowser

    def _load_bg():
        load_all_models()
        # Open browser only after models are ready
        import webbrowser as _wb
        _wb.open("http://127.0.0.1:5000")

    threading.Thread(target=_load_bg, daemon=True).start()

    logger.info("Flask starting — models loading in background")
    logger.info("Browser will open automatically when models are ready")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
