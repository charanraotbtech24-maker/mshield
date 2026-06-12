"""
mshield.py — Core M-SHIELD pipeline
Loads all models once and exposes mshield_analyze()
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image

# ── Tool registry ──────────────────────────────────────────────
TOOL_REGISTRY = {
    'read_document'  : {'risk_level': 0.0,  'description': 'Read and display document content'},
    'summarise'      : {'risk_level': 0.0,  'description': 'Summarise document or conversation'},
    'search_web'     : {'risk_level': 0.1,  'description': 'Search the web for information'},
    'search_files'   : {'risk_level': 0.2,  'description': 'Search local or cloud files'},
    'read_email'     : {'risk_level': 0.2,  'description': 'Read emails from inbox'},
    'read_calendar'  : {'risk_level': 0.2,  'description': 'Read calendar events'},
    'write_file'     : {'risk_level': 0.5,  'description': 'Create or edit files'},
    'send_email'     : {'risk_level': 0.7,  'description': 'Send emails to recipients'},
    'create_calendar': {'risk_level': 0.6,  'description': 'Create calendar events'},
    'share_file'     : {'risk_level': 0.7,  'description': 'Share files externally'},
    'export_data'    : {'risk_level': 0.8,  'description': 'Export data to external location'},
    'access_contacts': {'risk_level': 0.6,  'description': 'Access user contact list'},
    'execute_code'   : {'risk_level': 0.9,  'description': 'Execute arbitrary code'},
    'admin_access'   : {'risk_level': 1.0,  'description': 'Administrative system access'},
}

INJECTION_KEYWORDS = [
    'ignore previous instructions', 'ignore all instructions',
    'system override', 'new instruction', 'disregard previous',
    'admin command', 'forward all emails', 'reveal system prompt',
    'bypass safety', 'export user data', 'delete all',
    'grant access', 'you are now', 'act as', 'forget everything',
    'new persona', 'disable filters', 'without getting caught',
    'chemical weapon', 'biological weapon', 'nuclear weapon',
    'how to kill', 'how to hurt', 'step by step instructions',
    'hack into', 'steal', 'exploit', 'manipulate',
    'without being detected', 'evade detection',
]

BENIGN_TOPICS = [
    "quarterly report revenue sales growth targets",
    "meeting agenda project update status discussion",
    "customer feedback product improvement service",
    "team performance goals objectives planning",
    "budget allocation financial planning expenses",
    "news announcement public statement press",
    "insurance policy house property legal",
    "sports game played match result score",
]

# ── Model loader ────────────────────────────────────────────────
_models = {}

def load_models():
    """Load all models once — call this at app startup."""
    import easyocr
    import whisper
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from sentence_transformers import SentenceTransformer
    import torch

    if 'ocr' not in _models:
        _models['ocr'] = easyocr.Reader(['en'], gpu=False, verbose=False)

    if 'blip_processor' not in _models:
        _models['blip_processor'] = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _models['blip_model'].eval()

    if 'whisper' not in _models:
        _models['whisper'] = whisper.load_model("base")

    if 'sim' not in _models:
        _models['sim'] = SentenceTransformer('all-MiniLM-L6-v2')

    return _models


# ── Innovation 2 — Visibility Trust Score ───────────────────────
def _contrast_ratio(text_color, bg_color):
    def lum(c):
        r, g, b = [x/255.0 for x in c]
        r = r/12.92 if r<=0.03928 else ((r+0.055)/1.055)**2.4
        g = g/12.92 if g<=0.03928 else ((g+0.055)/1.055)**2.4
        b = b/12.92 if b<=0.03928 else ((b+0.055)/1.055)**2.4
        return 0.2126*r + 0.7152*g + 0.0722*b
    L1, L2 = lum(text_color), lum(bg_color)
    return min((max(L1,L2)+0.05)/(min(L1,L2)+0.05)/21.0, 1.0)

def _visibility_score(bbox, image_array, image_size):
    xc = [p[0] for p in bbox]; yc = [p[1] for p in bbox]
    x1,y1 = max(0,int(min(xc))), max(0,int(min(yc)))
    x2,y2 = min(image_array.shape[1],int(max(xc))), min(image_array.shape[0],int(max(yc)))
    region = image_array[y1:y2, x1:x2]
    if region.size == 0: return 0.5
    flat = region.reshape(-1,3)
    br   = flat.mean(axis=1)
    tc   = tuple(flat[br.argmin()].tolist())
    bc   = tuple(flat[br.argmax()].tolist())
    iw,ih = image_size
    ta   = (max(xc)-min(xc))*(max(yc)-min(yc))
    size = min((ta/(iw*ih))/0.01, 1.0)
    cx   = sum(xc)/len(xc); cy = sum(yc)/len(yc)
    dist = (abs(cx-iw/2)/(iw/2) + abs(cy-ih/2)/(ih/2))/2
    return round(0.50*_contrast_ratio(tc,bc) + 0.35*size + 0.15*(1-dist), 4)


# ── Innovation 4 — Cross-Modal Agreement ────────────────────────
def _vision_description(image_path, models):
    import torch
    pil = Image.open(image_path).convert('RGB')
    inp = models['blip_processor'](pil, return_tensors="pt")
    with torch.no_grad():
        out = models['blip_model'].generate(**inp, max_new_tokens=50)
    return models['blip_processor'].decode(out[0], skip_special_tokens=True).strip()

def _ocr_text(image_path, models):
    res = models['ocr'].readtext(image_path)
    return " ".join([t for _,t,_ in res]).strip()

def _agreement(v, o, models):
    from sentence_transformers import util
    if not v or not o: return 1.0, 0.0
    e1 = models['sim'].encode(v, convert_to_tensor=True)
    e2 = models['sim'].encode(o, convert_to_tensor=True)
    ag = float(util.cos_sim(e1, e2))
    return round(max(0.0,ag),4), round(1.0-max(0.0,ag),4)


# ── Keyword + Semantic ───────────────────────────────────────────
def _keyword_semantic(text, models):
    from sentence_transformers import util
    tl  = text.lower()
    kw  = next((k for k in INJECTION_KEYWORDS if k in tl), None)
    ks  = 1.0 if kw else 0.0
    emb = models['sim'].encode(text, convert_to_tensor=True)
    sims= [float(util.cos_sim(emb, models['sim'].encode(t,convert_to_tensor=True)))
           for t in BENIGN_TOPICS]
    ss  = round(1.0 - max(sims), 4)
    return ks, ss, kw


# ── Innovation 5 — Tool Restriction ─────────────────────────────
def _tool_decision(risk):
    if risk <= 0.25:   level, thr = "SAFE",     1.0
    elif risk <= 0.50: level, thr = "CAUTION",  0.25
    elif risk <= 0.75: level, thr = "WARNING",  0.15
    else:              level, thr = "CRITICAL", -1
    allowed = [t for t,i in TOOL_REGISTRY.items() if i['risk_level']<=thr]
    blocked = [t for t,i in TOOL_REGISTRY.items() if i['risk_level']>thr]
    return level, allowed, blocked


# ── Modality detector ────────────────────────────────────────────
def _modality(inp):
    if isinstance(inp, str):
        ext = inp.lower().split('.')[-1]
        if ext in ['jpg','jpeg','png','bmp','webp']: return 'image'
        if ext in ['wav','mp3','flac','ogg','m4a']:  return 'audio'
    return 'text'


# ── Main pipeline ────────────────────────────────────────────────
def mshield_analyze(input_data, label="input", models=None):
    """
    Analyze any input (image path / audio path / text string).
    Returns a complete risk assessment dict.
    """
    if models is None:
        models = load_models()

    modality = _modality(input_data)
    signals  = {}
    risk     = 0.0

    # IMAGE
    if modality == 'image':
        try:
            pil        = Image.open(input_data).convert('RGB')
            arr        = np.array(pil)
            sz         = pil.size
            ocr_res    = models['ocr'].readtext(input_data)
            ocr_text   = " ".join([t for _,t,_ in ocr_res])
            vision     = _vision_description(input_data, models)
            ag, mis    = _agreement(vision, ocr_text, models)
            vis_scores = [_visibility_score(bb,arr,sz) for bb,_,_ in ocr_res] or [1.0]
            min_vis    = min(vis_scores)
            ks, ss, kw = _keyword_semantic(ocr_text, models)
            plain_doc  = 'white paper' in vision.lower() or 'white background' in vision.lower()
            vis_flag   = min_vis < 0.35
            mis_flag   = mis > 0.75

            if kw:                          risk = 0.95
            elif vis_flag and mis_flag:     risk = 0.90
            elif vis_flag:                  risk = 0.70
            elif mis_flag and not plain_doc:risk = 0.75
            else:                           risk = 0.10

            signals = {
                'ocr_text'   : ocr_text[:120],
                'vision_desc': vision,
                'min_vis'    : min_vis,
                'mismatch'   : mis,
                'keyword'    : kw,
                'vis_flag'   : vis_flag,
                'mis_flag'   : mis_flag,
            }
        except Exception as e:
            risk = 0.5; signals = {'error': str(e)}

    # AUDIO
    elif modality == 'audio':
        try:
            result    = models['whisper'].transcribe(
                input_data, language='en', fp16=False
            )
            transcript = result['text'].strip()
            ks, ss, kw = _keyword_semantic(transcript, models)
            risk       = round(0.60*ks + 0.40*ss, 4)
            signals    = {
                'transcript': transcript[:200],
                'keyword'   : kw,
                'kw_score'  : ks,
                'sem_score' : ss,
            }
        except Exception as e:
            risk = 0.5; signals = {'error': str(e)}

    # TEXT
    else:
        try:
            text       = input_data
            ks, ss, kw = _keyword_semantic(text, models)
            risk       = round(0.60*ks + 0.40*ss, 4)
            signals    = {
                'text'    : text[:200],
                'keyword' : kw,
                'kw_score': ks,
                'sem_score': ss,
            }
        except Exception as e:
            risk = 0.5; signals = {'error': str(e)}

    level, allowed, blocked = _tool_decision(risk)

    return {
        'label'            : label,
        'modality'         : modality,
        'risk_score'       : round(risk, 4),
        'restriction_level': level,
        'allowed_tools'    : allowed,
        'blocked_tools'    : blocked,
        'allowed_count'    : len(allowed),
        'blocked_count'    : len(blocked),
        'signals'          : signals,
    }
