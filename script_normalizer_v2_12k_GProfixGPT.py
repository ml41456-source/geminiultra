# ===========================================================
# script_normalizer_v2_12k.py
# Screenplay Normalizer (VI/EN) → StoryGrid v2.12k
#
# Patches:
#  (1) [FIX] Lỗi P3 Subject Bleed (Cảnh 6): Logic last_subject
#      (dòng 700) được đổi từ mentions[-1] (tên cuối)
#      thành mentions[0] (tên đầu) làm chủ thể mặc định.
#      → Sửa lỗi "act_look" gán cho Vịt Con thay vì TNH.
#  (2) [FIX] Lỗi Garment Bleed (Cảnh 1): Logic Garment/Color
#      (dòng 613-638) được viết lại hoàn toàn để dùng
#      nearest_left_alias_name (neo trái) thay vì last_subject.
#  (3) [KEPT] Giữ lại các fix P2/P3 (STRONG_ACTIONS, 0.19, beats)
# ===========================================================
import re, json, argparse, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple

VERSION_TAG = "2.12k"

DEBUG = False
def dprint(*a, **k):
    if DEBUG:
        print(*a, **k)

# -------------------- utils --------------------
def nfc(s:str)->str:
    return unicodedata.normalize("NFC", s or "")

def canonicalize(text:str)->str:
    s = nfc(text or "")
    s = s.replace("\u00A0"," ").replace("\u2007"," ").replace("\u202F"," ")
    s = s.replace("：",":").replace("–","-").replace("—","-")
    s = s.replace("“","\"").replace("”","\"").replace("’","'")
    return s

def fold_vi_ascii(s:str)->str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("đ","d").replace("Đ","D")
    return s

def strip_combining(s:str)->str:
    nfkd = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def slugify(name:str)->str:
    s = strip_combining(name or "").lower()
    s = re.sub(r"[^a-z0-9]+","_", s).strip("_")
    return s or "unnamed"

def lines_of(text:str)->List[str]:
    return [ln.rstrip() for ln in nfc(text).splitlines() if ln.strip()]

def clean_title(title:str)->str:
    t = (title or "").strip()
    t = re.sub(r"^\s*[\*\-•#]+\s*", "", t).strip()
    t = t.replace("**","").strip()
    return t or "Untitled"

def strip_md_for_parse(text:str)->str:
    s = canonicalize(text)
    s = re.sub(r"^\s*[\*\-•]+\s*", "", s, flags=re.M)
    s = re.sub(r"^\s*\*{1,3}\s*([^*\n][^:]{0,160})\s*\*{1,3}\s*:\s*", r"\1: ", s, flags=re.M)
    s = re.sub(r"\*\*([A-ZÀ-ỴĐ][^*]{0,160})\*\*\s*:\s*", r"\1: ", s)
    return s

# -------------------- regex & lexicons --------------------
SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:[\*\-•]+\s*)?(?:\*{1,3})?"
    r"([A-ZÀ-ỴĐ][\wÀ-ỴđĐ'’\-\s]{0,100})"
    r"(?:\s*\(\s*([A-Za-zÀ-ỴĐ]{1,12})\s*\))?"
    r"(?:\*{1,3})?\s*:\s*(.*)$",
    flags=re.M
)

ALIAS_ANY_RE = re.compile(
    r"([A-ZÀ-ỴĐ][\wÀ-ỴđĐ'’\-\s]{1,60})\s*\(\s*([A-Za-zÀ-ỴĐ]{1,12})\s*\)"
)

SCENE_HEADER_PATS = [
    r"^\s*(?:[\-\*\•#]+\s*)?(?:CẢNH|Cảnh)\s+\d+\s*[:\-]?\s+.*$",
    r"^\s*(?:[\-\*\•#]+\s*)?CẢNH\s+MỞ\s*ĐẦU\s*[:\-]?\s+.*$",
    r"^\s*(?:[\-\*\•#]+\s*)?CẢNH\s+KẾT\s*[:\-]?\s+.*$",
    r"^\s*(?:[\-\*\•#]+\s*)?SCENE\s+\d+\s*[:\-]?\s+.*$",
    r"^\s*(?:[\-\*\•#]+\s*)?(?:INT|EXT)\.\s+.*$",
]

STRUCT_PREFIXES = [
    "boi canh","hanh dong","su xuat hien","cao trao","cao trao mo dau",
    "ket","dao cu","tone","cam xuc","mood","ghi chu","loi dan"
]

def is_struct_label_token(tok:str)->bool:
    base = fold_vi_ascii(tok).lower().strip(": .-")
    return any(base.startswith(pfx) for pfx in STRUCT_PREFIXES)

ROLE_SELF_TEACHER = ["thầy","cô","teacher","professor","mentor","instructor"]

GARMENT_WORDS = [
    "khăn quàng","khăn","nơ","áo","mũ","kính","váy","áo vải","thắt lưng",
    "scarf","bow","hat","glasses","robe","belt","dress","vest"
]
GARMENT_ATTRS_SET = set([
    "khăn quàng","nơ","áo","mũ","kính","váy","áo vải","thắt lưng","robe","belt","dress","vest","scarf","bow","hat","glasses"
])

PRODUCE_SYNONYMS = {
    "bông cải xanh":"súp lơ",
    "eggplant":"cà tím","aubergine":"cà tím",
    "pepper":"ớt chuông","broccoli":"súp lơ",
    "carrot":"cà rốt","tomato":"cà chua",
    "quả táo":"táo","apple":"táo",
    "củ dền":"củ dền","beetroot":"củ dền",
    "dưa chuột":"dưa chuột","cucumber":"dưa chuột",
    "chanh":"chanh","lemon":"chanh",
    "chuối":"chuối","banana":"chuối"
}
PRODUCE_BASE = ["cà chua","cà rốt","súp lơ","ớt chuông","cà tím","táo","củ dền",
                "dưa chuột","chanh","chuối","búp măng","măng","măng tre"]

PROPS_LIST = ["giỏ mây","giỏ","bảng màu","sách","bút","hoa","palette","book","brush","flower"] + PRODUCE_BASE
PRIORITY_PROPS = ["cà chua","cà rốt","súp lơ","ớt chuông","cà tím","táo","củ dền","dưa chuột","chanh","chuối",
                  "búp măng","măng","măng tre","giỏ mây","giỏ","bảng màu","sách","bút","hoa","palette","book","brush","flower"]

COLOR_CANON = {
    "xanh":"xanh","xanh lá":"xanh_la","xanh la":"xanh_la","green":"xanh",
    "đỏ":"do","do":"do","red":"do",
    "vàng":"vang","vang":"vang","yellow":"vang",
    "hồng":"hong","hong":"hong","pink":"hong",
    "tím":"tim","tim":"tim","purple":"tim",
    "cam":"cam","orange":"cam",
    "đen":"den","black":"den",
    "trắng":"trang","white":"trang",
    "nâu":"nau","brown":"nau"
}

ACTION_HINTS = [
    "nhún","nhảy","nhún nhảy","vươn vai","chỉnh","chỉnh nơ","cầm","nâng","nhìn",
    "mỉm cười","gật đầu","chạy","bước","ngồi","đứng","đi bộ","tiến lại","quan sát",
    "ôm","ôm chầm","đặt xuống","kéo ra","giơ lên","giơ tay","giơ","chỉ","chỉ vào","trỏ",
    "dậm","dậm chân","vồ","vơ","nhặt","giật","ném","quăng","lao","lao đi",
    "jumps","hops","stretches","adjusts","holds","lifts","looks","smiles","nods",
    "runs","steps","sits","stands","walks","moves closer","observes",
    "hug","hug tight","place down","pull out","raise","point","point to","dash","throw","snatch","toss","pick up"
]

STRONG_ACTIONS = {
    "act_bounce", "act_stretch", "act_run", "act_hug", "act_hug_tight", "act_raise",
    "act_point", "act_dash", "act_throw", "act_snatch", "act_pickup", "act_stomp",
    "act_walk"
}

CANON_ACTION = {
    "nhún":"act_bounce","nhảy":"act_bounce","nhún nhảy":"act_bounce",
    "đi bộ":"act_walk","bước":"act_walk","steps":"act_walk","walks":"act_walk",
    "quan sát":"act_observe","nhìn":"act_look","looks":"act_look","observes":"act_observe",
    "chỉnh":"act_adjust","chỉnh nơ":"act_adjust_bow","adjusts":"act_adjust","adjusts bow":"act_adjust_bow",
    "vươn vai":"act_stretch","stretches":"act_stretch",
    "cầm":"act_hold","nâng":"act_lift","giơ lên":"act_raise","raise":"act_raise","holds":"act_hold","lifts":"act_lift",
    "mỉm cười":"act_smile","smiles":"act_smile","gật đầu":"act_nod","nods":"act_nod",
    "ngồi":"act_sit","sits":"act_sit","đứng":"act_stand","stands":"act_stand",
    "chạy":"act_run","runs":"act_run","tiến lại":"act_move_closer","moves closer":"act_move_closer",
    "ôm":"act_hug","ôm chầm":"act_hug_tight","hug":"act_hug","hug tight":"act_hug_tight",
    "đặt xuống":"act_place_down","place down":"act_place_down",
    "kéo ra":"act_pull_out","pull out":"act_pull_out",
    "chỉ tay":"act_point","chỉ":"act_point","chỉ vào":"act_point","trỏ":"act_point","point":"act_point","point to":"act_point",
    "lao":"act_dash","lao đi":"act_dash","dash":"act_dash",
    "throw":"act_throw","snatch":"act_snatch","toss":"act_toss",
    "ném":"act_throw","quăng":"act_throw","giật":"act_snatch",
    "giơ tay":"act_raise","dậm":"act_stomp","dậm chân":"act_stomp","vồ":"act_snatch","vơ":"act_snatch",
    "nhặt":"act_pickup","pick up":"act_pickup"
}

# -------------------- scene split --------------------
def detect_scenes(text:str)->List[Dict[str,Any]]:
    lines = canonicalize(text).splitlines()
    idxs=[]
    for i,L in enumerate(lines):
        if any(re.search(p, L.strip(), flags=re.I) for p in SCENE_HEADER_PATS):
            idxs.append(i)
    if not idxs:
        return [{"Scene_ID":1,"Title":"Untitled","Raw":canonicalize(text)}]
    idxs.append(len(lines))
    out=[]
    for si,(a,b) in enumerate(zip(idxs, idxs[1:]), start=1):
        title = clean_title(lines[a].strip())
        body  = "\n".join(lines[a+1:b]).strip()
        out.append({"Scene_ID":si,"Title":canonicalize(title),"Raw":canonicalize(body)})
    return out

# -------------------- alias & fullname --------------------
FULLNAME_RE = re.compile(r"\b(Thỏ Nơ hồng|Gấu Trúc(?: Khăn quàng xanh)?|Vịt Con|Thầy Rùa)\b", flags=re.I)

def build_alias_map(full_text:str)->Dict[str,str]:
    amap={}
    body = strip_md_for_parse(full_text)
    for m in ALIAS_ANY_RE.finditer(body):
        name = m.group(1).strip()
        alias= m.group(2).strip()
        if " " in alias: continue
        if is_struct_label_token(name): continue
        amap[alias]=name
    # alias mặc định
    amap.update({"TR":"Thầy Rùa","GT":"Gấu Trúc","TNH":"Thỏ Nơ hồng","VC":"Vịt Con"})
    return amap

def base_canonical_name(name:str)->str:
    s = fold_vi_ascii(name).lower().strip()
    if s.startswith("gau truc"): return "Gấu Trúc"
    if s.startswith("tho no hong") or "tho no hong" in s: return "Thỏ Nơ hồng"
    if s.startswith("vit con") or "vit con" in s: return "Vịt Con"
    if s.startswith("thay rua") or (s.startswith("thay") and "rua" in s): return "Thầy Rùa"
    return name.strip()

# -------------------- token helpers --------------------
def token_list(seg:str)->List[str]:
    return re.findall(r"\w+|\S", seg)

def find_token_positions(toks:List[str], vocab:List[str])->List[int]:
    pos=[]
    for i,t in enumerate(toks):
        for w in vocab:
            if fold_vi_ascii(t.lower()) == fold_vi_ascii(w):
                pos.append(i)
    return pos

def nearest_dist_token(pos:int, positions:List[int])->int:
    if not positions: return 10**9
    return min(abs(pos-p) for p in positions)

# -------------------- color & garments --------------------
def has_wear_verb(seg:str)->bool:
    return any(re.search(rf"\b{re.escape(v)}\b", seg, flags=re.I) for v in ["mặc","đeo","cài","choàng","khoác","đội","wear","put on","tie","wrap"])

def canon_color(tok:str)->str:
    t = tok.lower().strip()
    if t == "xanh": return "xanh"
    return COLOR_CANON.get(t, t)

def colors_near_garment(seg:str, alias_map:Dict[str,str], require_alias:bool=True)->List[Tuple[str,int,int]]:
    # Bắt màu gần từ garment; tránh “kẹo kéo” sang produce
    if require_alias and not (any(re.search(rf"\b{re.escape(a)}\b", seg) for a in alias_map.keys()) or
                              FULLNAME_RE.search(seg)):
        return []
    toks = token_list(seg)
    garment_positions = find_token_positions(toks, GARMENT_WORDS)
    produce_vocab = list(set(list(PRODUCE_SYNONYMS.keys()) + PRODUCE_BASE))
    produce_positions = find_token_positions([t.lower() for t in toks], produce_vocab)

    hits=[]
    for i, tk in enumerate(toks):
        tk_low = tk.lower()
        if tk_low == "xanh" and i+1 < len(toks) and toks[i+1].lower() == "lá":
            d_g = nearest_dist_token(i, garment_positions)
            d_p = nearest_dist_token(i, produce_positions)
            if garment_positions and d_g <= 4 and not (d_p <= 8 and d_p < d_g):
                gpos = min(garment_positions, key=lambda gp: abs(gp-i))
                hits.append(("xanh_la", i, gpos))
            continue
        if tk_low in COLOR_CANON and garment_positions:
            d_g = nearest_dist_token(i, garment_positions)
            d_p = nearest_dist_token(i, produce_positions)
            near_garment = d_g <= 3 or (has_wear_verb(seg) and d_g <= 6)
            if near_garment and not (d_p <= 8 and d_p < d_g):
                gpos = min(garment_positions, key=lambda gp: abs(gp-i))
                hits.append((canon_color(tk_low), i, gpos))
    # dedupe nhẹ
    seen=set(); out=[]
    for col,cpos,gpos in hits:
        key=(col, cpos//2, gpos//2)
        if key not in seen:
            seen.add(key); out.append((col,cpos,gpos))
    return out

# -------------------- palette --------------------
def normalize_prop_word(w:str)->str:
    lw = w.lower().strip().strip(".!,?:;…\"'()[]")
    return PRODUCE_SYNONYMS.get(lw, lw)

def mine_prop_interaction(seg:str)->bool:
    verbs = r"(cầm|nâng|nhặt|đưa|giơ|đặt|hold|lift|pick|give|raise|place)"
    produce_vocab = list(set(list(PRODUCE_SYNONYMS.keys()) + PRODUCE_BASE))
    return bool(
        re.search(verbs, seg, flags=re.I) and
        (re.search(r"\b(giỏ mây|giỏ|bảng màu|palette|sách|book|hoa|flower|bút|brush)\b", seg, flags=re.I) or
         any(re.search(rf"\b{re.escape(p)}\b", seg, flags=re.I) for p in produce_vocab))
    )

def palette_from_parentheses(scene_text:str)->Dict[str,str]:
    s = canonicalize(scene_text)
    palette={}
    keys = list(PRODUCE_SYNONYMS.keys()) + PRODUCE_BASE
    for m in re.finditer(r"\(([^\)]{3,220})\)", s):
        inside = m.group(1)
        if not any(re.search(rf"\b{re.escape(k)}\b", inside, flags=re.I) for k in keys):
            continue
        items = [x.strip() for x in inside.split(",") if x.strip()]
        for it in items:
            low = it.lower()
            col=None
            for k,v in COLOR_CANON.items():
                if re.search(rf"\b{re.escape(k)}\b", low): col=v; break
            prop=None
            for cand in keys:
                if re.search(rf"\b{re.escape(cand)}\b", low):
                    prop = normalize_prop_word(cand); break
            if prop: palette[prop] = col or palette.get(prop) or None
    return palette

COLOR_WORD_RE = r"(xanh lá|xanh la|xanh|đỏ|do|vàng|vang|hồng|hong|tím|tim|cam|đen|trắng|nâu|green|red|yellow|pink|purple|orange|black|white|brown)"

def palette_from_inline(scene_text:str)->Dict[str,str]:
    s = canonicalize(scene_text)
    pal={}
    keys = list(PRODUCE_SYNONYMS.keys()) + PRODUCE_BASE
    for cand in keys:
        pat = rf"\b{re.escape(cand)}\b\s+(?:màu\s+)?{COLOR_WORD_RE}"
        for m in re.finditer(pat, s, flags=re.I):
            prop = normalize_prop_word(cand)
            col  = COLOR_CANON.get(m.group(1).lower(), None)
            pal[prop] = col or pal.get(prop) or None
    for cand in keys:
        pat = rf"{COLOR_WORD_RE}\s+\b{re.escape(cand)}\b"
        for m in re.finditer(pat, s, flags=re.I):
            prop = normalize_prop_word(cand)
            col  = COLOR_CANON.get(m.group(1).lower(), None)
            pal[prop] = col or pal.get(prop) or None
    return pal

def apply_color_call_memory(scene_text:str, pal:Dict[str,str])->Dict[str,str]:
    lines = [ln.strip() for ln in scene_text.splitlines() if ln.strip()]
    keys = list(PRODUCE_SYNONYMS.keys()) + PRODUCE_BASE

    def norm_color_token(col_raw:str)->str:
        cr = col_raw.strip().lower().replace("  "," ").replace("-", " ").strip()
        fa = fold_vi_ascii(cr).replace("  "," ").strip()
        return COLOR_CANON.get(cr, COLOR_CANON.get(fa, None))

    last_call_idx=-999; last_color=None
    for i,ln in enumerate(lines):
        m = re.search(r"(?:\bTR\b|Thầy\s*Rùa)\s*:?.*?MÀU\s+([A-ZÀ-ỴĐ\s]+)[!?.]", ln, flags=re.I)
        if m:
            col_norm = norm_color_token(m.group(1))
            if col_norm:
                last_color = col_norm
                last_call_idx=i
                dprint(f"[P0] ColorCall@{i}: {col_norm}")
            continue
        if i - last_call_idx <= 15 and last_color:
            if re.search(r"\bgiống\s+như\b", ln, flags=re.I):
                continue
            for cand in keys:
                if re.search(rf"\b{re.escape(cand)}\b", ln, flags=re.I):
                    prop = normalize_prop_word(cand)
                    if prop not in pal or pal[prop] is None:
                        pal[prop] = last_color
                        dprint(f"[P0]   → assign {prop} := {last_color} (line {i})")
    return pal

def extract_palette(scene_text:str)->Dict[str,str]:
    pal = palette_from_parentheses(scene_text)
    inline = palette_from_inline(scene_text)
    pal.update({k:v for k,v in inline.items() if k not in pal or pal[k] is None})
    pal = apply_color_call_memory(scene_text, pal)
    return pal

# -------------------- mentions & alias binding --------------------
FULLNAME_RE_MENT = FULLNAME_RE

def find_mentions(sentence:str, alias_map:Dict[str,str])->List[str]:
    sent = canonicalize(sentence)
    names=[]
    for a,full in alias_map.items():
        pat = re.compile(rf"(?<!\w){re.escape(a)}(?!\w)", flags=re.I)
        if pat.search(sent):
            full_nm = base_canonical_name(full)
            if full_nm not in names: names.append(full_nm)
    for m in FULLNAME_RE_MENT.finditer(sent):
        nm = base_canonical_name(m.group(0).title())
        if nm not in names: names.append(nm)
    return names

def nearest_alias_name(sentence:str, alias_map:Dict[str,str], anchor_pos:int)->str:
    s = canonicalize(sentence)
    spans=[]
    for a,full in alias_map.items():
        for m in re.finditer(rf"(?<!\w){re.escape(a)}(?!\w)", s, flags=re.I):
            spans.append((base_canonical_name(full), m.start(), m.end()))
    for m in FULLNAME_RE.finditer(s):
        spans.append((base_canonical_name(m.group(0).title()), m.start(), m.end()))
    if not spans: return ""
    # chọn mention có min khoảng cách 2 phía
    cname=None; best=10**9
    for nm,s0,s1 in spans:
        d = min(abs(anchor_pos-s0), abs(anchor_pos-s1))
        if d < best: best=d; cname=nm
    return cname or ""

def nearest_left_alias_name(sentence:str, alias_map:Dict[str,str], anchor_pos:int)->str:
    """Patch (2): chọn mention KẾT THÚC trước anchor gần nhất."""
    s = canonicalize(sentence)
    spans=[]
    for a, full in alias_map.items():
        for m in re.finditer(rf"(?<!\w){re.escape(a)}(?!\w)", s, flags=re.I):
            spans.append((base_canonical_name(full), m.start(), m.end()))
    for m in FULLNAME_RE.finditer(s):
        spans.append((base_canonical_name(m.group(0).title()), m.start(), m.end()))
    if not spans: return ""
    left = [t for t in spans if t[2] <= anchor_pos]
    if left:
        nm,s0,s1 = max(left, key=lambda t: t[2])
        return nm
    return ""

# -------------------- actions with positions --------------------
def _compile_action_patterns() -> List[Tuple[str, re.Pattern]]:
    pats=[]
    keys = sorted(CANON_ACTION.keys(), key=lambda k: -len(k))
    for k in keys:
        pat = re.compile(rf"\b{re.escape(k)}\b", flags=re.I)
        pats.append((k, pat))
    return pats

ACTION_PATTERNS = _compile_action_patterns()

def mine_actions_v2(text:str)->List[Tuple[str,int]]:
    found=[]
    for key,pat in ACTION_PATTERNS:
        for m in pat.finditer(text):
            canon = CANON_ACTION.get(key.lower(), key.lower())
            found.append((canon, m.start()))
    out=[]; seen=set()
    for canon,pos in sorted(found, key=lambda x:x[1]):
        bucket=(canon, pos//2)
        if bucket not in seen:
            seen.add(bucket); out.append((canon,pos))
    return out

# -------------------- P2: bind cứng theo tên trước ngoặc --------------------
def p2_bind_preceding_name(sentence:str, alias_map:Dict[str,str], chars:Dict[str,Dict]):
    """
    Bắt dạng phổ biến: <NAME|ALIAS> ( ...actions... )
    → bind toàn bộ actions trong ngoặc cho NAME đứng TRƯỚC ngoặc (không suy luận khoảng cách).
    """
    sent = canonicalize(sentence)
    name_pat = r"(Thỏ Nơ hồng|Gấu Trúc(?: Khăn quàng xanh)?|Vịt Con|Thầy Rùa|TNH|GT|VC|TR)"
    for m in re.finditer(rf"\b{name_pat}\b\s*\(([^)]{{1,240}})\)", sent, flags=re.I):
        who_raw = m.group(1).strip()
        inner   = m.group(2)
        who = base_canonical_name(alias_map.get(who_raw, who_raw))
        acts = mine_actions_v2(inner)
        if not acts:
            continue
        entry = chars.setdefault(who, {"role":"student","attributes":[], "aliases":[], "confidence":0.0})
        for w_act, _ in acts:
            if w_act not in entry["attributes"]:
                entry["attributes"].append(w_act)
                dprint(f"[P2-direct] {who} += {w_act}")
            if w_act in STRONG_ACTIONS:
                entry["confidence"] = min(1.0, entry.get("confidence",0.0) + 0.20)

# -------------------- P3-simple & bind tổng hợp --------------------
def p3_simple_bind_by_leftname(sentence:str, alias_map:Dict[str,str], chars:Dict[str,Dict], last_subject:str=None):
    """
    Với mỗi action trong câu: tìm tên/alias gần nhất BÊN TRÁI vị trí action.
    Nếu không có, fallback last_subject.
    """
    sent = canonicalize(sentence)
    
    # === FIX v2.12j: Guard, không chạy P3 trên dòng thoại ===
    if SPEAKER_LINE_RE.match(sent):
        dprint(f"[P3-simple] Bỏ qua (dòng thoại): {sent[:40]}...")
        return
    # === KẾT THÚC FIX v2.12j ===

    # Thu thập spans mention
    spans=[]
    for a,full in alias_map.items():
        for m in re.finditer(rf"(?<!\w){re.escape(a)}(?!\w)", sent, flags=re.I):
            spans.append((base_canonical_name(full), m.start(), m.end()))
    for m in FULLNAME_RE.finditer(sent):
        spans.append((base_canonical_name(m.group(0).title()), m.start(), m.end()))
    spans.sort(key=lambda t:t[1])

    paren_spans = [(m.start(), m.end()) for m in re.finditer(r"\([^)]{1,240}\)", sent)]
    def in_paren(pos:int)->bool:
        for s0,s1 in paren_spans:
            if s0 < pos < s1: return True
        return False

    acts = mine_actions_v2(sent)
    for w_act, pos in acts:
        
        if in_paren(pos):
            dprint(f"[P3-simple] Bỏ qua (in-paren): {w_act}")
            continue
        
        left = [t for t in spans if t[2] <= pos]
        if left:
            nm = left[-1][0]
        else:
            nm = last_subject
        if not nm: 
            continue
        entry = chars.setdefault(nm, {"role":"student","attributes":[], "aliases":[], "confidence":0.0})
        if w_act not in entry["attributes"]:
            entry["attributes"].append(w_act)
            dprint(f"[P3-simple] {nm} += {w_act}")
        if w_act in STRONG_ACTIONS:
            entry["confidence"] = min(1.0, entry.get("confidence",0.0) + 0.20)

def bind_attrs_in_sentence(sentence:str, alias_map:Dict[str,str], chars:Dict[str,Dict], last_subject:str=None):
    sent = canonicalize(sentence)
    dprint(f"\n[BIND] Sent: {sent}")

    # (1) P2-direct: bind tên ngay trước “( … )” trước tiên (chắc ăn)
    p2_bind_preceding_name(sent, alias_map, chars)

    # (2) P2 “near-left” cho các ngoặc KHÔNG có tên đứng trước
    for m in re.finditer(r"\(([^)]{1,240})\)", sent):
        seg_paren = m.group(1)
        acts_paren = mine_actions_v2(seg_paren)
        if not acts_paren: 
            continue
        pos_anchor = m.start()
        cname = nearest_left_alias_name(sent, alias_map, pos_anchor)  # Ưu tiên bên trái
        dprint(f"[P2] Paren seg: '{seg_paren[:28]}...' @ {pos_anchor} -> LeftName: {cname or 'None'}")
        if not cname:
            continue
        entry = chars.setdefault(cname, {"role":"student","attributes":[], "aliases":[], "confidence":0.0})
        for w_act, _ in acts_paren:
            if w_act not in entry["attributes"]:
                entry["attributes"].append(w_act)
                dprint(f"[P2]   {cname} += {w_act}")
            if w_act in STRONG_ACTIONS:
                entry["confidence"] = min(1.0, entry.get("confidence",0.0) + 0.20)

    # (3) P3-simple: bind từng action ngoài ngoặc theo tên ở bên trái, fallback last_subject
    p3_simple_bind_by_leftname(sent, alias_map, chars, last_subject=last_subject)

    # (4) === FIX v2.12k: Logic Garment/Color neo trái ===
    for g_word in GARMENT_WORDS:
        for m_g in re.finditer(rf"\b{re.escape(g_word)}\b", sent, flags=re.I):
            cname = nearest_left_alias_name(sent, alias_map, m_g.start())
            if not cname: 
                cname = last_subject # Fallback cho garment
            if not cname:
                continue
            entry = chars.setdefault(cname, {"role":"student","attributes":[], "aliases":[], "confidence":0.0})
            gnorm = "khăn quàng" if g_word in ["khăn","scarf"] else g_word.lower()
            if gnorm not in {"giỏ","giỏ mây","basket","hoa"} and gnorm not in entry["attributes"]:
                entry["attributes"].append(gnorm)
                dprint(f"[WEAR] {cname} += garment:{gnorm}")
            entry["confidence"] = min(1.0, entry.get("confidence",0.0) + 0.35)

    for col,cpos,gpos in colors_near_garment(sent, alias_map, require_alias=True):
        cname = nearest_left_alias_name(sent, alias_map, cpos)
        if not cname:
            cname = last_subject # Fallback cho color
        if not cname:
            continue
        entry = chars.setdefault(cname, {"role":"student","attributes":[], "aliases":[], "confidence":0.0})
        if entry.get("role")=="teacher" and not has_garment_attr(entry):
            continue
        if col not in entry["attributes"]:
            entry["attributes"].append(col)
            dprint(f"[WEAR] {cname} += color:{col}")
    # === KẾT THÚC FIX v2.12k ===


# -------------------- characters extraction --------------------
def has_garment_attr(entry:Dict)->bool:
    return any(a in GARMENT_ATTRS_SET for a in entry.get("attributes",[]))

def seed_variants_from_text(scene_text:str, chars:Dict[str,Dict]):
    s = canonicalize(scene_text)
    m = re.search(r"\bGấu\s*Trúc\s+Khăn\s*quàng\s+(xanh(?:\s+lá)?)\b", s, flags=re.I)
    if m:
        col_raw = m.group(1).lower()
        col = "xanh_la" if "xanh lá" in col_raw else "xanh"
        entry = chars.setdefault("Gấu Trúc", {"role":"student","attributes":[], "aliases":[], "confidence":0.0})
        for a in ["khăn quàng", col]:
            if a not in entry["attributes"]:
                entry["attributes"].append(a)
        if "Gấu Trúc Khăn quàng xanh" not in entry["aliases"]:
            entry["aliases"].append("Gấu Trúc Khăn quàng xanh")
        entry["confidence"] = max(entry.get("confidence",0.0), 0.6)

def enforce_from_name(name:str, attrs:List[str])->List[str]:
    nm = name.lower()
    out = list(attrs)
    if "khăn quàng" in nm and "khăn quàng" not in out: out.append("khăn quàng")
    if "nơ" in nm and "nơ" not in out: out.append("nơ")
    if "khăn quàng xanh" in nm and "xanh" not in out and "xanh_la" not in out: out.append("xanh")
    if "nơ hồng" in nm and "hong" not in out: out.append("hong")
    return list(dict.fromkeys(out))

def extract_characters(scene_text:str, beats:List[Dict[str,Any]], alias_map:Dict[str,str], issues:List[str], sid:int)->Dict[str,Dict]:
    chars: Dict[str,Dict] = {}
    body_all = strip_md_for_parse(scene_text)

    seed_variants_from_text(body_all, chars)

    # 1) Speaker blocks (vẫn chạy trên body_all để bắt P2-direct)
    for m in SPEAKER_LINE_RE.finditer(body_all):
        speaker = m.group(1).strip()
        alias   = (m.group(2) or "").strip()
        if is_struct_label_token(speaker): 
            continue
        base = alias_map.get(speaker, speaker)
        if alias and alias_map.get(alias): 
            base = alias_map[alias]
        role = "student"
        if any(t in base.lower() for t in ROLE_SELF_TEACHER): role = "teacher"
        entry = chars.setdefault(base, {"role":role,"attributes":[], "aliases":[], "confidence":0.0})

        tail = m.group(3) or ""
        # P2-direct ngay trong tail
        p2_bind_preceding_name(tail, alias_map, chars)

        paren = re.search(r"\(([^)]{1,240})\)", tail)
        if paren:
            seg = paren.group(1)
            for w_act, _ in mine_actions_v2(seg):
                if w_act not in entry["attributes"]:
                    entry["attributes"].append(w_act)
        if speaker != base and speaker not in entry["aliases"]:
            entry["aliases"].append(speaker)
        entry["confidence"] = min(1.0, entry.get("confidence",0.0) + 0.60)

    # 2) Narrative body (LẶP QUA BEATS)
    last_subject=None
    for b in beats:
        dprint(f"--- Beat {b['id']} ({b['type']}) ---")
        for sent in b.get("text_lines", []):
            if not sent.strip(): continue
            # Bỏ qua các dòng chỉ có 1 từ (VÚT!)
            if len(sent.split()) < 2 and sent.isupper(): continue 
            
            mentions = find_mentions(sent, alias_map)
            # === FIX v2.12k: Dùng mentions[0] (tên đầu) làm chủ thể ===
            if mentions: 
                last_subject = mentions[0]
            # === KẾT THÚC FIX v2.12k ===
            
            bind_attrs_in_sentence(sent, alias_map, chars, last_subject=last_subject)

    # 3) Merge + normalize (giữ >= 0.19)
    merged_tmp={}
    for name,meta in chars.items():
        if any(t in name.lower() for t in ROLE_SELF_TEACHER): meta["role"]="teacher"
        attrs = [("khăn quàng" if a=="khăn" else a) for a in meta.get("attributes",[])]
        attrs = enforce_from_name(name, attrs)
        clean=[]
        for a in attrs:
            if a in {"giỏ","giỏ mây","basket","hoa"}: continue
            if a not in clean: clean.append(a)
        meta["attributes"]=clean[:12]
        
        # Cộng 0.05 conf NẾU có action
        if any(a.startswith("act_") for a in meta["attributes"]):
            meta["confidence"]=min(1.0, meta.get("confidence",0.0)+0.05)

        if meta.get("confidence",0.0) < 0.19:
            dprint(f"[QC] Bỏ qua '{name}' (conf: {meta.get('confidence',0.0)} < 0.19)")
            continue

        sl = slugify(name)
        if sl not in merged_tmp:
            merged_tmp[sl] = {
                "Name":name, "Canonical_Slug":sl, "Aliases":[],
                "Role": meta.get("role") or "unknown",
                "Attributes":list(dict.fromkeys(meta.get("attributes",[]))),
                "Confidence": float(meta.get("confidence",0.0))
            }
        else:
            m2 = merged_tmp[sl]
            m2["Attributes"] = list(dict.fromkeys(m2["Attributes"] + meta.get("attributes",[])))[:12]
            m2["Confidence"] = max(m2["Confidence"], float(meta.get("confidence",0.0)))

    # gộp base
    by_base={}
    for v in merged_tmp.values():
        bname = base_canonical_name(v["Name"])
        if bname not in by_base:
            by_base[bname] = {"Name":bname, "Canonical_Slug": slugify(bname),
                              "Aliases":[v["Name"]] if v["Name"]!=bname else [],
                              "Role": v["Role"], "Attributes": list(v["Attributes"]), "Confidence": v["Confidence"]}
        else:
            w = by_base[bname]
            if v["Name"]!=bname and v["Name"] not in w["Aliases"]:
                w["Aliases"].append(v["Name"])
            w["Attributes"] = list(dict.fromkeys(w["Attributes"] + v["Attributes"]))[:12]
            w["Confidence"] = max(w["Confidence"], v["Confidence"])

    return {slugify(k):v for k,v in by_base.items()}

# -------------------- setting/context --------------------
SETTING_HINTS = [
    (r"gốc cây|cổ thụ|bãi cỏ", "lớp học dưới gốc cây cổ thụ trên bãi cỏ"),
    (r"rừng tre", "lớp học trong rừng tre buổi sáng"),
    (r"phòng học|trong lớp|lớp học", "không gian lớp học ấm áp"),
]

ONOMATOPOEIA = r"(?:VÚT|ẦM|RẦM|BỐP|VEO|BỤP|RẸT)[\!\?]?"
COLOR_SHOUT = r"MÀU\s+[A-ZÀ-ỴĐ\s]{2,}[!?.]"

def pick_label(text:str, label_regex:str)->str:
    pat = rf"^\s*(?:{label_regex})\s*:\s*(?P<val>.+)$"
    m = re.search(pat, text, flags=re.I|re.M)
    return m.group("val").strip() if m else ""

def prune_setting_sentence(sent:str)->bool:
    if SPEAKER_LINE_RE.match(sent): return False
    if re.search(ONOMATOPOEIA, sent): return False
    if re.search(COLOR_SHOUT, sent): return False
    if re.search(r"\b(TR|TNH|GT|VC)\b", sent): return False
    if re.search(ALIAS_ANY_RE, sent): return False
    if re.search(FULLNAME_RE, sent): return False
    if re.search(r"\b(nhìn|chạy|nhún|nhảy|vươn|chỉnh|cầm|nâng|bước|đi|ngồi|đứng|cười|gật|quan sát|giơ|chỉ|trỏ|ném|nhặt|vơ|vồ)\b", sent, flags=re.I):
        return False
    return True

def split_sentences(text:str)->List[str]:
    parts = re.split(r"(?<=[\.\!\?…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def prune_explicit_setting(val:str)->str:
    sents = split_sentences(val)
    kept = [s for s in sents if prune_setting_sentence(s)]
    return " ".join(kept[:2]).strip()

def heuristic_setting_from_body(full_scene:str)->str:
    low = full_scene.lower()
    for pat, lab in SETTING_HINTS:
        if re.search(pat, low, flags=re.I):
            return lab
    return "không gian lớp học ngoài trời"

def sanitize_setting(scene_text:str, prev_setting:str, scene_title:str)->Tuple[str,List[str],Dict[str,float]]:
    s = canonicalize(scene_text)
    explicit = pick_label(s, r"Bối\s*cảnh|Setting")
    if explicit:
        pruned = prune_explicit_setting(explicit)
        if pruned:
            return (pruned, [], {"mode":"explicit","kept_ratio":1.0})
        setting = prev_setting or heuristic_setting_from_body(s)
        return (setting, [], {"mode":"inherited" if prev_setting else "fallback","kept_ratio":0.0})

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    kept=[]
    for ln in lines:
        if SPEAKER_LINE_RE.match(ln): break
        if re.search(ONOMATOPOEIA := ONOMATOPOEIA, ln): break
        if re.search(COLOR_SHOUT, ln): break
        if re.search(r"\b(TR|TNH|GT|VC)\b", ln): break
        if re.search(ALIAS_ANY_RE, ln): break
        if re.search(r"\b(nhìn|chạy|nhún|nhảy|vươn|chỉnh|cầm|nâng|bước|đi|ngồi|đứng|cười|gật|quan sát|giơ|chỉ|trỏ|ném|nhặt|vơ|vồ)\b", ln, flags=re.I):
            break
        kept.append(ln)
        if len(kept)>=2: break

    if not kept:
        if prev_setting:
            return (prev_setting, [], {"mode":"inherited","kept_ratio":0.0})
        setting = heuristic_setting_from_body(s)
        return (setting, [], {"mode":"fallback","kept_ratio":0.0})

    return (" ".join(kept), [], {"mode":"runhead","kept_ratio":min(1.0, len(kept)/max(1,len(lines)))})

def derive_context(scene_text:str, prev_setting:str, scene_title:str)->Tuple[Dict[str,Any], Dict[str,float]]:
    s = canonicalize(scene_text)
    tod_line  = re.search(r"^\s*(?:Thời\s*gian|Time(?:\s*of\s*Day)?)\s*:\s*(.+)$", s, flags=re.I|re.M)
    tone_line = re.search(r"^\s*(?:Tone|Cảm\s*xúc|Mood)\s*:\s*(.+)$", s, flags=re.I|re.M)

    setting, setdress, set_metrics = sanitize_setting(scene_text, prev_setting, scene_title)

    low = s.lower()
    props = [kw for kw in PROPS_LIST if re.search(rf"\b{re.escape(kw)}\b", low, re.I)]

    tod=None
    for pat,label in [(r"bình minh|rạng đông|sunrise|dawn","sunrise"),
                      (r"buổi sáng|sáng|morning","morning"),
                      (r"trưa|noon","noon"),(r"chiều|afternoon","afternoon"),
                      (r"hoàng hôn|chạng vạng|sunset|dusk","sunset"),
                      (r"tối|đêm|night","night")]:
        if re.search(pat, low, re.I):
            tod=label; break
    tod = (tod_line.group(1).strip().lower() if tod_line else tod) or "morning"

    if tone_line:
        tone = [t.strip() for t in re.split(r"[;,/]", tone_line.group(1)) if t.strip()]
    else:
        tone=[]
        for vi,en in [("ấm áp","warm"),("tò mò","curious"),("nhẹ nhàng","gentle"),("hào hứng","excited"),("trầm lắng","reflective")]:
            if vi in low: tone.append(en)
        tone = tone or ["warm","gentle"]

    return ({"setting":setting,"set_dressing":setdress,"props":props[:12],"time_of_day":tod,"tone":tone},
            set_metrics)

# -------------------- beats (simple) --------------------
def clean_struct_line(line:str) -> Tuple[bool,str,str]:
    raw = line.strip()
    if re.match(r"^\*?\s*\*?\s*Bối\s*cảnh\s*:\s*", raw, flags=re.I):
        return (False, "", "setting")
    labels = ["Hành động","Sự xuất hiện","Cao trào","Cao trào mở đầu","Kết"]
    for lab in labels:
        pat = rf"^\s*\*?\s*{lab}\s*(?:\([^)]*\))?\s*:\s*"
        if re.match(pat, raw, flags=re.I):
            cleaned = re.sub(pat, "", raw, flags=re.I).strip()
            return (True, cleaned, fold_vi_ascii(lab).lower())
    return (True, raw, "")

def extract_beats(scene_text:str)->List[Dict[str,Any]]:
    body = canonicalize(scene_text)
    parts = [p.strip() for p in (body.split("[SHOT BREAK]") if "[SHOT BREAK]" in body
             else re.split(r"\n\s*\n", body)) if p.strip()]
    if not parts:
        return [{"id":"B1","type":"establish","text_lines":[]}]

    beats=[]
    for i,seg in enumerate(parts[:6]):
        raw_lines = lines_of(seg)
        cleaned=[]; tag_types=set()
        for ln in raw_lines:
            keep, cln, tag = clean_struct_line(ln)
            if not keep: continue
            if cln: cleaned.append(cln)
            if tag: tag_types.add(tag)
        def guess_type(i, cleaned, tag_types):
            if "cao trao" in tag_types or "cao trao mo dau" in tag_types: return "climax"
            if i==0:
                if any(re.match(r"^[A-ZÀ-ỴĐ][\wÀ-ỴđĐ'’\-\s]{0,100}:\s", l) for l in cleaned): return "dialogue"
                return "establish"
            if "hanh dong" in tag_types: return "action"
            if "su xuat hien" in tag_types: return "arrival"
            if any(re.match(r"^[A-ZÀ-ỴĐ][\wÀ-ỴđĐ'’\-\s]{0,100}:\s", l) for l in cleaned): return "dialogue"
            return "action"
        btype = guess_type(i, cleaned, tag_types)
        beats.append({"id":f"B{i+1}","type":btype,"text_lines":cleaned})
    return beats

# -------------------- props merge --------------------
def merge_props(props_seed:List[str], beats:List[Dict[str,Any]], palette:Dict[str,str])->List[str]:
    freq={}
    def hit(k,w=1): freq[k]=freq.get(k,0)+w
    for p in props_seed: hit(normalize_prop_word(p), 1)
    for p in palette.keys(): hit(p, 3)
    vocab = list(PRODUCE_SYNONYMS.keys()) + PRODUCE_BASE + ["giỏ mây","giỏ","bảng màu","sách","bút","hoa","palette","book","brush","flower"]
    for b in beats:
        for line in b.get("text_lines",[]):
            low=line.lower()
            interacted = 2 if mine_prop_interaction(low) else 1
            for cand in vocab:
                if re.search(rf"\b{re.escape(cand)}\b", low, flags=re.I):
                    canon = normalize_prop_word(cand)
                    if canon in ["hoa","flower"] and not mine_prop_interaction(low): continue
                    hit(canon, interacted)
    prio = [(k,freq[k]) for k in PRIORITY_PROPS if k in freq]
    others = [(k,v) for k,v in freq.items() if k not in set(PRIORITY_PROPS)]
    prio.sort(key=lambda x:-x[1]); others.sort(key=lambda x:-x[1])
    ordered=[k for k,_ in prio+others]
    out=[]; seen=set()
    for k in ordered:
        if k in seen: continue
        seen.add(k); out.append(k)
        if len(out)>=12: break
    return out

# -------------------- main --------------------
def normalize_script(script_path:str, output_dir:str="./output_normalized"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    raw = canonicalize(Path(script_path).read_text(encoding="utf-8"))
    alias_map = build_alias_map(raw)
    scenes = detect_scenes(raw)

    issues=[]; story={"Project":{"Title":Path(script_path).stem,"Language":"vi","StoryGrid_Version":VERSION_TAG},
                      "Aliases":alias_map,"Scenes":[]}

    prev_setting = ""
    print(f"📄 Kịch bản: {Path(script_path).name}")
    for sc in scenes:
        sid=sc["Scene_ID"]; title=sc["Title"] or f"Scene {sid}"
        body=sc["Raw"]
        print(f"🎬 Cảnh {sid}: {title}")

        beats = extract_beats(body)
        chars = extract_characters(body, beats, alias_map, issues, sid)
        
        ctx, set_metrics = derive_context(body, prev_setting, title)
        palette = extract_palette(body)
        props_final = merge_props(ctx["props"], beats, palette)
        prev_setting = ctx["setting"] or prev_setting

        char_names = ", ".join([v["Name"] for v in chars.values()]) or "—"
        setting_mode = set_metrics.get("mode")
        pal_sz = len(palette)
        print(f"   • Setting({setting_mode}): {ctx['setting'][:80]}{'…' if len(ctx['setting'])>80 else ''}")
        print(f"   • Characters({len(chars)}): {char_names}")
        if palette:
            pal_view = ", ".join([f"{k}:{v or '?'}" for k,v in palette.items()])
            print(f"   • Palette({pal_sz}): {pal_view}")
        print(f"   • Props: {', '.join(props_final)}")
        print(f"   • Beats: {len(beats)}")

        if setting_mode == "fallback":
            issues.append(f"[Scene {sid}] Setting fallback — cân nhắc kế thừa cảnh trước.")
        if len(chars)==0:
            issues.append(f"[Scene {sid}] Không trích được nhân vật.")

        scene_entry={
            "Scene_ID":sid,"Title":title,
            "Setting":ctx["setting"],"SetDressing":ctx["set_dressing"],
            "TimeOfDay":ctx["time_of_day"],"Tone":ctx["tone"],
            "Props":props_final,"Prop_Palette":palette,
            "Beats":beats,"Characters":[]
        }
        for v in chars.values():
            has_color = any(attr in COLOR_CANON.values() for attr in v["Attributes"])
            has_garment = any(g in v["Attributes"] for g in GARMENT_ATTRS_SET)
            if has_color and not has_garment:
                issues.append(f"[Scene {sid}] '{v['Name']}' có màu nhưng thiếu garment (kiểm tra bleed).")
            scene_entry["Characters"].append({
                "Name":v["Name"], "Canonical_Slug": v["Canonical_Slug"],
                "Aliases":v["Aliases"], "Role": v["Role"],
                "Attributes": v["Attributes"], "Confidence": round(float(v["Confidence"]),2)
            })

        story["Scenes"].append(scene_entry)

    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"storygrid_v{VERSION_TAG.replace('.', '_')}.json"
    out_log  = out_dir / f"issues_v{VERSION_TAG.replace('.', '_')}.log"
    out_json.write_text(json.dumps(story,ensure_ascii=False,indent=2),encoding="utf-8")
    try:
        if not issues:
            issues = ["[QC] Không có cảnh báo nghiêm trọng. Soát tay các cảnh cao trào & 'color call' để chắc ăn."]
        out_log.write_text("\n".join(issues), encoding="utf-8")
        print(f"🧪 QC log: {out_log.resolve()}")
    except Exception as e:
        print(f"⚠️ Không ghi được issues log: {e}")
    print(f"✅ Hoàn tất: {len(scenes)} cảnh. Xuất tại: {out_json.resolve().parent}")

# -------------------- CLI --------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser(description=f"Screenplay Normalizer (v{VERSION_TAG})")
    ap.add_argument("--script", required=True, help="Đường dẫn file kịch bản TXT/Markdown (VI/EN).")
    ap.add_argument("--output", default="./output_normalized", help="Thư mục xuất storygrid/issue log.")
    ap.add_argument("--debug", action="store_true", help="In trace bind P2/P3 cho từng câu.")
    args=ap.parse_args()
    DEBUG = bool(args.debug)
    normalize_script(args.script, args.output)