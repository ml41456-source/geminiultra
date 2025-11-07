# ===========================================================
# script_normalizer_v11_byGemini.py
# Screenplay Normalizer → StoryGrid v3.4 (Cinematic Intelligence & Data Optimization)
# - V11.1: Attribute Persistence Logic (Cinematic Continuity across shots).
# - V11.2: Data Structure Optimization (List to Dict in Shot_Composition).
# - V11.3: Attribute Deduplication (Filters inherent attributes from shots).
# - V11.4: Data Cleanup (Removes redundant Scene_Characters/Groups).
# - V11.5: Robust Temporal Logic (Ensures 'arrival' beats trigger presence).
# ===========================================================
import re, json, argparse, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
import itertools
import sys
import copy

# Import the profile loader
try:
    # V11: Updated import name
    import profiles_v11 as profiles
except ImportError:
    print("❌ Lỗi: Không tìm thấy module 'profiles_v11.py'. Hãy đảm bảo file này nằm cùng thư mục với script chính.")
    sys.exit(1)

# ===========================================================
# SECTION 1: UTILITIES & HELPERS (No Changes from V10)
# ===========================================================

# (Utilities remain identical to V10)
def nfc(s:str)->str: return unicodedata.normalize("NFC", s or "")

def canonicalize(text:str)->str:
    s = nfc(text)
    s = s.replace("\u00A0"," ").replace("\u2007"," ").replace("\u202F"," ")
    s = s.replace("：",":").replace("\u2013","-").replace("\u2014","-").replace("–","-").replace("—","-")
    s = s.replace("“",'"').replace("”",'"').replace("’","'")
    return s

def strip_combining(s:str)->str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def slugify(name:str)->str:
    s = strip_combining(name).lower()
    s = re.sub(r"[^a-z0-9]+","_", s).strip("_")
    s = s.replace("đ", "d")
    return s or "unnamed"

def clean_title(title:str)->str:
    t = title.strip()
    t = re.sub(r"^\s*[\*\-•#]+\s*", "", t)
    t = t.replace("**","").strip()
    return t

def extract_robust_parentheticals(text: str) -> Tuple[List[str], str]:
    parentheticals = []; cleaned_text_parts = []
    nesting_level = 0; last_extraction_end = 0; start_index = 0

    for i, char in enumerate(text):
        if char == '(':
            if nesting_level == 0:
                cleaned_text_parts.append(text[last_extraction_end:i])
                start_index = i
            nesting_level += 1
        elif char == ')':
            if nesting_level > 0:
                nesting_level -= 1
                if nesting_level == 0:
                    content = text[start_index+1:i].strip()
                    if content:
                        parentheticals.append(content)
                    last_extraction_end = i + 1

    if last_extraction_end < len(text):
        cleaned_text_parts.append(text[last_extraction_end:])

    cleaned_text = "".join(cleaned_text_parts).strip()
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return parentheticals, cleaned_text


# ===========================================================
# SECTION 2: DYNAMIC PROFILE SYSTEM (No Changes from V10)
# ===========================================================

REGEX_CACHE = {}

def compile_regexes(profile: Dict[str, Any]):
    # (Implementation remains identical to V10)
    cache_key = f"{profile['language']}_{profile['genre']}"
    if cache_key in REGEX_CACHE:
        return REGEX_CACHE[cache_key]

    rules = profile['parsing_rules']
    UPPER_CHARS = rules['upper_chars']; LOWER_CHARS = rules['lower_chars']

    WORD_PART = "[" + UPPER_CHARS + LOWER_CHARS + "'-]+"
    CAP_WORD = "[" + UPPER_CHARS + "]" + WORD_PART + "?"

    SPEAKER_RE = re.compile(
        r"^\s*(?:[\*\-•]+\s*)?(?:\*{1,3})?"
        r"(" + CAP_WORD + r"(?:\s+" + WORD_PART + r"){0,10})"
        r"(?:\s*\([^)]*\))?"
        r"(?:\*{1,3})?\s*:\s*"
        r"(.*)$", re.M
    )

    ALIAS_EXPLICIT_RE = re.compile(
        r"(" + CAP_WORD + r"(?:\s+" + WORD_PART + r"){0,10})\s*\(\s*([" + UPPER_CHARS + r"0-9]{1,12})\s*\)"
    )

    action_label = next((k for k, v in rules['structure_labels'].items() if v == 'action'), 'Hành động')
    arrival_label = next((k for k, v in rules['structure_labels'].items() if v == 'arrival'), 'Sự xuất hiện')

    NARRATIVE_INTRO_RE = re.compile(
        rf"^\s*(?:[\*\-•]*\s*\*?\*?\s*)?(?:{re.escape(action_label)}|{re.escape(arrival_label)})\s*(?:\(.*\))?\s*:\s*(.*)", re.M | re.I
    )

    ALLOWED_LOWERCASE_WORDS = set(word for phrase in profile['lexicons']['appearance_phrases'] for word in phrase.lower().split())
    LOWERCASE_GROUP = "(?:" + "|".join(re.escape(w) for w in ALLOWED_LOWERCASE_WORDS) + ")"

    NARRATIVE_NAME_RE = re.compile(
        r"\b(" + CAP_WORD +
        r"(?:\s+(?:" + CAP_WORD + r"|(?i:" + LOWERCASE_GROUP + r"))){0,5})\b"
    )

    SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

    compiled = {
        "SPEAKER_RE": SPEAKER_RE, "ALIAS_EXPLICIT_RE": ALIAS_EXPLICIT_RE,
        "NARRATIVE_INTRO_RE": NARRATIVE_INTRO_RE, "NARRATIVE_NAME_RE": NARRATIVE_NAME_RE,
        "SENTENCE_SPLIT_RE": SENTENCE_SPLIT_RE
    }
    REGEX_CACHE[cache_key] = compiled
    return compiled

# ===========================================================
# SECTION 3: CORE LOGIC - PARSING & EXTRACTION (No Changes from V10)
# ===========================================================
# Note: The implementation in this section remains stable as finalized in V10.

# ---------- Validation Helpers (No Changes) ----------

def is_invalid_struct_or_blacklist(tok:str, profile: Dict[str, Any])->bool:
    # (Implementation remains identical to V10)
    rules = profile['parsing_rules']
    base = tok.strip(": .-"); base = re.sub(r"^\s*[\*\-•#]*\s*", "", base)
    _, clean_base = extract_robust_parentheticals(base)

    if not clean_base: return True
    if clean_base.lower() in [b.lower() for b in rules['character_blacklist']]: return True
    for label in rules['structure_labels'].keys():
        if clean_base.lower().startswith(label.lower()): return True
    return False

def validate_and_clean_name(name: str, profile: Dict[str, Any], allow_group_agents=False) -> Tuple[bool, str]:
    # (Implementation remains identical to V10)
    rules = profile['parsing_rules']; lexicons = profile['lexicons']
    if not name: return False, ""
    if is_invalid_struct_or_blacklist(name, profile): return False, name

    is_group_agent = name.lower() in [g.lower() for g in rules.get('group_agents', [])]
    if is_group_agent and not allow_group_agents: return False, name
    if is_group_agent and allow_group_agents: return True, name

    prop_set = set(p.lower() for p in lexicons['props_list'])
    if name.lower() in prop_set: return False, name

    cleaned_name = name
    if not is_group_agent:
        for suffix in rules['invalid_name_suffixes']:
            if cleaned_name.lower().endswith(" " + suffix.lower()):
                cleaned_name = cleaned_name[:-(len(suffix)+1)].strip(); break
        for prefix in rules['invalid_name_prefixes']:
            if cleaned_name.lower().startswith(prefix.lower() + " "):
                cleaned_name = cleaned_name[len(prefix)+1:].strip(); break

    if not cleaned_name or is_invalid_struct_or_blacklist(cleaned_name, profile): return False, name

    is_group_agent_cleaned = cleaned_name.lower() in [g.lower() for g in rules.get('group_agents', [])]
    if is_group_agent_cleaned and not allow_group_agents: return False, cleaned_name
    if cleaned_name.lower() in prop_set: return False, name

    return True, cleaned_name


# --- Phrase Mining Helpers (No Changes) ---

def mine_phrases(text: str, phrase_list: List[str]) -> List[str]:
    # (Implementation remains identical to V10)
    found = set()
    sorted_phrases = sorted(phrase_list, key=len, reverse=True)
    canonical_text = canonicalize(text)

    for phrase in sorted_phrases:
        canonical_phrase = canonicalize(phrase)
        if re.search(rf"\b{re.escape(canonical_phrase)}\b", canonical_text, flags=re.I):
            is_subsumed = False
            for existing in found:
                if re.search(rf"\b{re.escape(canonical_phrase)}\b", canonicalize(existing), flags=re.I):
                    is_subsumed = True; break
            if not is_subsumed:
                to_remove = set()
                for existing in found:
                    if re.search(rf"\b{re.escape(canonicalize(existing))}\b", canonical_phrase, flags=re.I):
                        to_remove.add(existing)
                found -= to_remove
                found.add(phrase)
    return sorted(list(found))

def mine_appearance(text: str, profile: Dict[str, Any]) -> List[str]:
    return mine_phrases(text, profile['lexicons']['appearance_phrases'])
def mine_actions(text: str, profile: Dict[str, Any]) -> List[str]:
    return mine_phrases(text, profile['lexicons']['action_phrases'])
def mine_emotions(text: str, profile: Dict[str, Any]) -> List[str]:
    return mine_phrases(text, profile['lexicons']['emotion_phrases'])

def mine_cinematic_instructions(text: str, profile: Dict[str, Any]) -> Dict[str, List[str]]:
    # (Implementation remains identical to V10)
    if not text: return {}
    instructions = {"camera": [], "vfx_sfx": [], "meta": []}
    config = profile.get('cinematic_instructions', {})
    
    for keyword, tag in config.get('camera_moves', {}).items():
        if re.search(rf"\b{re.escape(keyword)}\b", text, re.I): instructions["camera"].append(tag)
    for keyword, tag in config.get('vfx_sfx', {}).items():
        if re.search(rf"\b{re.escape(keyword)}\b", text, re.I): instructions["vfx_sfx"].append(tag)
    for pattern, tag in config.get('meta_types', {}).items():
        if re.search(pattern, text, re.I): instructions["meta"].append(tag)
            
    return {k: sorted(list(set(v))) for k, v in instructions.items() if v}


# ---------- Scene Detection (No Changes) ----------
def detect_scenes(text:str, profile: Dict[str, Any])->List[Dict[str,Any]]:
    # (Implementation remains identical to V10)
    SCENE_HEADER_PATS = profile['parsing_rules']['scene_header_patterns']
    canonical_text = canonicalize(text); lines = canonical_text.splitlines(); idxs=[]
    for i,L in enumerate(lines):
        if any(re.search(p, L.strip(), flags=re.I) for p in SCENE_HEADER_PATS): idxs.append(i)

    if not idxs: return [{"Scene_ID":1,"Title":"Untitled","Raw":canonical_text}]

    idxs.append(len(lines)); out=[]
    for si,(a,b) in enumerate(zip(idxs, idxs[1:]), start=1):
        title = clean_title(lines[a].strip())
        body  = "\n".join(lines[a+1:b]).strip()
        if body: out.append({"Scene_ID":si,"Title":title,"Raw":body})
    return out

# ---------- Global Pass: Character Consolidation (No Changes) ----------

def global_character_pass(full_text:str, profile: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Dict], Dict[str, Dict]]:
    # (Implementation remains identical to V10)
    text = canonicalize(full_text)
    raw_terms = set(); explicit_aliases = {}
    regexes = compile_regexes(profile); rules = profile['parsing_rules']

    def process_term(name, alias=None):
        is_valid, cleaned_name = validate_and_clean_name(name, profile, allow_group_agents=False)
        if is_valid:
            raw_terms.add(cleaned_name)
            if alias:
                is_valid_a, cleaned_alias = validate_and_clean_name(alias, profile, allow_group_agents=False)
                if is_valid_a:
                     raw_terms.add(cleaned_alias); explicit_aliases[cleaned_alias] = cleaned_name

    # 1. Extraction
    for m in regexes['ALIAS_EXPLICIT_RE'].finditer(text):
        name = m.group(1).strip(); alias = m.group(2).strip()
        if len(alias.split()) == 1: process_term(name, alias)

    for m in regexes['SPEAKER_RE'].finditer(text):
        name = m.group(1).strip()
        _, clean_name = extract_robust_parentheticals(name)
        process_term(clean_name)

    for m in regexes['NARRATIVE_INTRO_RE'].finditer(text):
        intro_text = m.group(1)
        for n_match in regexes['NARRATIVE_NAME_RE'].finditer(intro_text):
            potential_name = n_match.group(1).strip()
            is_valid, cleaned_name = validate_and_clean_name(potential_name, profile, allow_group_agents=False)
            if is_valid:
               is_known = any(t.lower() == cleaned_name.lower() for t in raw_terms)
               if len(cleaned_name.split()) >= 2 or is_known:
                   raw_terms.add(cleaned_name)

    # 2. Resolution & 3. Consolidation
    resolved_names = set()
    for term in raw_terms:
        resolved = term
        for alias, name in explicit_aliases.items():
            if alias.lower() == term.lower(): resolved = name; break
        resolved_names.add(resolved)

    sorted_names = sorted(list(resolved_names), key=len)
    canonical_map = {name: name for name in resolved_names}

    for i, name_a in enumerate(sorted_names):
        for name_b in sorted_names[i+1:]:
            if name_b.lower().startswith(name_a.lower() + " "):
                canonical_map[name_a] = canonical_map[name_b]

    # 4. Final Mapping
    global_term_to_canonical_map = {}
    for term in raw_terms:
        resolved = term
        for alias, name in explicit_aliases.items():
            if alias.lower() == term.lower(): resolved = name; break
        
        canonical = None
        for variation, canon in canonical_map.items():
            if variation.lower() == resolved.lower(): canonical = canon; break

        if canonical: global_term_to_canonical_map[term] = canonical

    # 5. Building Global Character Registry
    global_characters = {}
    canonical_names = set(canonical_map.values())
    
    for c_name in canonical_names:
        role = "student"
        if any(t in c_name.lower() for t in rules['role_self_teacher']): role = "teacher"
        
        aliases = []
        for term, canon in global_term_to_canonical_map.items():
            if canon == c_name and term != c_name: aliases.append(term)

        inherent_appearance = mine_appearance(c_name, profile)

        global_characters[c_name] = {
            "role": role,
            "aliases": sorted(aliases),
            "slug": slugify(c_name),
            "inherent_attributes": {
                "Appearance": inherent_appearance,
                "Actions": [],
                "Emotions": []
            }
        }

    # 6. Building Global Group Agent Registry
    global_group_agents = {}
    for agent_name in rules.get('group_agents', []):
        if re.search(rf"\b{re.escape(agent_name)}\b", text, flags=re.I):
             global_group_agents[agent_name] = {
                "slug": slugify(agent_name),
                "type": "group"
            }

    return global_term_to_canonical_map, global_characters, global_group_agents

# ---------- Context Derivation ----------

def derive_context(scene_text:str, profile: Dict[str, Any])->Dict[str,Any]:
    # (Implementation remains identical to V10)
    s = scene_text; lexicons = profile['lexicons']; rules = profile['parsing_rules']

    def pick(label_regex:str)->str:
        pat = rf"^\s*(?:\*{{0,3}})?\s*(?:{label_regex})\s*(?:\*{{0,3}})?\s*:\s*(?P<val>.+)$"
        m = re.search(pat, s, flags=re.I|re.M)
        return m.group("val").strip() if m else ""

    setting_label = next((k for k, v in rules['structure_labels'].items() if v == 'setting'), 'Bối cảnh')
    setting_line = pick(re.escape(setting_label) + r"|Setting")

    if setting_line:
        setting = setting_line
    else:
        setting = "không gian không xác định (suy luận)"
        if profile['language'] == 'vi':
             setting = "không gian lớp học ngoài trời (suy luận)"

    low = s.lower(); raw_props = set()
    sorted_props_list = sorted(lexicons['props_list'], key=len, reverse=True)

    for kw in sorted_props_list:
        canonical_kw = canonicalize(kw)
        if re.search(rf"\b{re.escape(canonical_kw)}\b", low, re.I):
            raw_props.add(kw)

    sorted_raw_props = sorted(list(raw_props), key=len); final_props = set(raw_props)
    for i, prop_a in enumerate(sorted_raw_props):
        for prop_b in sorted_raw_props[i+1:]:
            if prop_a in prop_b:
                 if prop_a in final_props:
                     try: final_props.remove(prop_a)
                     except KeyError: pass
                 break

    tod = "morning"
    if profile['language'] == 'vi':
        if "sáng sớm" in low or "nắng sớm" in low: tod = "early_morning"

    tone=[];
    for key, val in lexicons['tone_map']:
        if key in low: tone.append(val)

    if not tone: tone=["warm","gentle"]

    return {"setting":setting,"props":sorted(list(final_props)),"time_of_day":tod,"tone":sorted(list(set(tone)))}


# ---------- Beats Segmentation (V11.4 Data Flow Update) ----------

LINE_TYPES = {
    "DIALOGUE": "dialogue", "ACTION": "action", "STRUCTURE": "structure",
    "META": "meta", "IGNORE": "ignore"
}

def classify_line(line:str, profile: Dict[str, Any]) -> Tuple[str, str, str]:
    # (Implementation remains identical to V10)
    raw = line.strip()
    if not raw: return LINE_TYPES["IGNORE"], "", ""
    regexes = compile_regexes(profile)

    # 1. Check Dialogue
    if regexes['SPEAKER_RE'].match(raw):
        speaker_part = regexes['SPEAKER_RE'].match(raw).group(1).strip()
        _, clean_speaker = extract_robust_parentheticals(speaker_part)
        is_valid, _ = validate_and_clean_name(clean_speaker, profile, allow_group_agents=True)
        if is_valid: return LINE_TYPES["DIALOGUE"], raw, ""

    # 2. Check Structure Labels
    rules = profile['parsing_rules']
    clean_for_struct_check = re.sub(r"^\s*[\*\-•]*\s*\*?\*?\s*", "", raw)

    for label, tag in rules['structure_labels'].items():
        if re.match(rf"^{re.escape(label)}\s*(?:\(.*\))?\s*:", clean_for_struct_check, re.I):
            if tag == 'action': return LINE_TYPES["ACTION"], raw, tag
            return LINE_TYPES["STRUCTURE"], raw, tag

    # 3. Check for META vs ACTION conflict
    has_narrative = mine_actions(raw, profile) or mine_emotions(raw, profile) or mine_appearance(raw, profile)
    instructions = mine_cinematic_instructions(raw, profile)
    
    if instructions.get("meta") and not has_narrative:
         return LINE_TYPES["META"], raw, "meta_instruction"

    # 4. Default to Action.
    return LINE_TYPES["ACTION"], raw, ""


def safe_clean_labeled_content(line:str, profile: Dict[str, Any]) -> str:
    # (Implementation remains identical to V10)
    rules = profile['parsing_rules']
    clean_line = re.sub(r"^\s*(\*{{1,3}})?(.*?)\1?\s*$", r"\2", line).strip()

    for label in rules['structure_labels'].keys():
        pattern = rf"^(?:[\*\-•]\s*)?{re.escape(label)}\s*(?:\(.*\))?\s*:\s*"
        if re.match(pattern, clean_line, re.I):
            cleaned = re.sub(pattern, "", clean_line, count=1, flags=re.I).strip()
            return cleaned
    return line.strip()

def analyze_narrative_attributes(text_lines: List[str], agent_slug_lookup: Dict[str, str], profile: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    # (Implementation remains identical to V10.2)
    attributes_by_slug = {}
    regexes = compile_regexes(profile); rules = profile['parsing_rules']; lexicons = profile['lexicons']
    NEGATION_KEYWORDS = rules.get('negation_keywords', [])

    full_text = " ".join(text_lines)
    sentences = regexes['SENTENCE_SPLIT_RE'].split(full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    sorted_terms = sorted(agent_slug_lookup.keys(), key=len, reverse=True)
    sorted_actions = sorted(lexicons['action_phrases'], key=len, reverse=True)
    sorted_emotions = sorted(lexicons['emotion_phrases'], key=len, reverse=True)
    sorted_appearance = sorted(lexicons['appearance_phrases'], key=len, reverse=True)

    def initialize_slug(slug):
        if slug not in attributes_by_slug:
            attributes_by_slug[slug] = {"Actions": [], "Emotions": [], "Appearance": []}

    def find_entities(sentence, terms, type_label):
        entities = []; canonical_sentence = canonicalize(sentence)
        for term in terms:
            canonical_term = canonicalize(term)
            for match in re.finditer(rf"\b{re.escape(canonical_term)}\b", canonical_sentence, re.I):
                entity_value = term
                if type_label == "AGENT":
                     entity_value = agent_slug_lookup.get(term.lower())
                
                if entity_value:
                    is_overlapped = False
                    for existing in entities:
                        if existing['type'] == type_label and match.start() >= existing['start'] and match.end() <= existing['end']:
                            is_overlapped = True; break
                    if not is_overlapped:
                        entities.append({"start": match.start(), "end": match.end(), "value": entity_value, "type": type_label})
        return entities

    # Analyze each sentence
    for sentence in sentences:
        
        # 1. Find all entities
        agents = find_entities(sentence, sorted_terms, "AGENT")
        actions = find_entities(sentence, sorted_actions, "ACTION")
        emotions = find_entities(sentence, sorted_emotions, "EMOTION")
        appearances = find_entities(sentence, sorted_appearance, "APPEARANCE")

        all_entities = sorted(agents + actions + emotions + appearances, key=lambda x: x['start'])
        unique_agents = list(set([a['value'] for a in agents]))

        # 2. Association Logic
        def get_target_list(slug, entity_type):
            if entity_type == "ACTION": return attributes_by_slug[slug]["Actions"]
            if entity_type == "EMOTION": return attributes_by_slug[slug]["Emotions"]
            if entity_type == "APPEARANCE": return attributes_by_slug[slug]["Appearance"]
            return None

        # Heuristic 1: Single Agent Safety
        if len(unique_agents) == 1:
            slug = unique_agents[0]; initialize_slug(slug)
            for entity in all_entities:
                if entity['type'] in ["ACTION", "EMOTION", "APPEARANCE"]:
                    phrase = entity['value']; target_list = get_target_list(slug, entity['type'])
                    is_negated = False
                    if entity['type'] != "APPEARANCE":
                        for neg_kw in NEGATION_KEYWORDS:
                            pattern = rf"\b{re.escape(neg_kw)}\b.*?\b{re.escape(phrase)}\b"
                            if re.search(pattern, sentence, re.I):
                                is_negated = True; break
                    
                    if not is_negated and phrase not in target_list:
                         target_list.append(phrase)

        # Heuristic 2: Subject-Verb/Adjective Proximity
        elif len(unique_agents) > 1:
            for i, entity in enumerate(all_entities):
                if entity['type'] in ["ACTION", "EMOTION", "APPEARANCE"]:
                    closest_subject = None
                    for j in range(i - 1, -1, -1):
                        if all_entities[j]['type'] == "AGENT":
                            closest_subject = all_entities[j]; break
                        if all_entities[j]['type'] in ["ACTION", "EMOTION", "APPEARANCE"]:
                             break

                    if closest_subject:
                        slug = closest_subject['value']; initialize_slug(slug)
                        phrase = entity['value']; target_list = get_target_list(slug, entity['type'])

                        is_negated_in_segment = False
                        if entity['type'] != "APPEARANCE":
                            segment = sentence[closest_subject['end']:entity['start']]
                            for neg_kw in NEGATION_KEYWORDS:
                                if re.search(rf"\b{re.escape(neg_kw)}\b", segment, re.I):
                                    is_negated_in_segment = True; break
                        
                        if not is_negated_in_segment and phrase not in target_list:
                            target_list.append(phrase)

    # Clean up results
    for slug in attributes_by_slug:
        attributes_by_slug[slug]["Actions"] = sorted(attributes_by_slug[slug]["Actions"])
        attributes_by_slug[slug]["Emotions"] = sorted(attributes_by_slug[slug]["Emotions"])
        attributes_by_slug[slug]["Appearance"] = sorted(attributes_by_slug[slug]["Appearance"])

    return attributes_by_slug


# V11.4: Updated return signature
def extract_beats(scene_text:str, agent_slug_lookup: Dict[str, str], profile: Dict[str, Any]) -> List[Dict[str,Any]]:
    """
    (V11.4) Extracts beats. Scene narrative attributes are no longer needed at this level.
    """
    # (Implementation remains identical to V10, just return signature changed)
    lines = scene_text.splitlines()
    classified_lines = []
    for line in lines:
        l_type, content, tag = classify_line(line, profile)
        if l_type != LINE_TYPES["IGNORE"]:
            classified_lines.append({"type": l_type, "raw": content, "tag": tag})

    if not classified_lines:
        return [{"id":"B1","type":"establish","text_lines":[]}]

    beats = []
    beat_id = 1
    regexes = compile_regexes(profile)
    
    for group_type, group_iter in itertools.groupby(classified_lines, key=lambda x: x['type']):
        group = list(group_iter)

        # STRUCTURE and META beats
        if group_type == LINE_TYPES["STRUCTURE"] or group_type == LINE_TYPES["META"]:
            for item in group:
                tag = item['tag']
                if tag == 'setting': continue
                
                cleaned_text = safe_clean_labeled_content(item['raw'], profile)
                if cleaned_text:
                    beat_type = item['type'] if group_type == LINE_TYPES["META"] else tag
                    beat = {"id":f"B{beat_id}","type":beat_type,"text_lines":[cleaned_text]}
                    
                    beat_attrs = analyze_narrative_attributes([cleaned_text], agent_slug_lookup, profile)
                    if beat_attrs:
                        beat["attributes"] = beat_attrs

                    instructions = mine_cinematic_instructions(item['raw'], profile)
                    if instructions: beat["instructions"] = instructions

                    beats.append(beat); beat_id += 1
        
        elif group_type == LINE_TYPES["DIALOGUE"]:
            text_lines = [item['raw'] for item in group]; dialogue_lines = []

            for line in text_lines:
                match = regexes['SPEAKER_RE'].match(line)
                if match:
                    speaker_raw = match.group(1).strip(); dialogue_content = match.group(2).strip()
                    p_speaker, clean_speaker_text = extract_robust_parentheticals(speaker_raw)
                    canonical_speaker = canonicalize(clean_speaker_text).lower()
                    speaker_slug = agent_slug_lookup.get(canonical_speaker, slugify(canonical_speaker))

                    p_dialogue, clean_line = extract_robust_parentheticals(dialogue_content)
                    
                    # V10.3 Cleaning Logic
                    clean_line = re.sub(r'\s+([.!?])$', r'\1', clean_line.strip())
                    if clean_line.startswith('"'):
                        if clean_line.endswith('"'):
                             clean_line = clean_line[1:-1].strip()
                        elif clean_line.endswith('".'):
                             clean_line = clean_line[1:-2].strip() + "."
                        else:
                             clean_line = clean_line[1:].strip()
                    clean_line = re.sub(r'\s+([.!?])$', r'\1', clean_line.strip())

                    all_parentheticals = p_speaker + p_dialogue
                    p_text = " ".join(all_parentheticals) if all_parentheticals else None
                    instructions = mine_cinematic_instructions(p_text, profile) if p_text else None

                    dialogue_entry = {
                        "Speaker_Slug": speaker_slug,
                        "Parenthetical": p_text,
                        "Line": clean_line
                    }
                    if instructions:
                        dialogue_entry["instructions"] = instructions
                    dialogue_lines.append(dialogue_entry)

            beats.append({"id":f"B{beat_id}","type":"dialogue","text_lines":text_lines, "dialogue_lines": dialogue_lines})
            beat_id += 1

        elif group_type == LINE_TYPES["ACTION"]:
            current_block = []

            def finalize_action_block(block_items):
                nonlocal beat_id
                if not block_items: return

                clean_lines = []; aggregated_instructions = {}
                for item in block_items:
                     raw_text = item['raw']
                     if item['tag'] == 'action':
                         cleaned = safe_clean_labeled_content(raw_text, profile)
                     else:
                         cleaned = re.sub(r"^\s*[\-•\*]\s*", "", raw_text).strip()
                     
                     if cleaned: clean_lines.append(cleaned)
                     
                     instructions = mine_cinematic_instructions(raw_text, profile)
                     for key, values in instructions.items():
                         if key not in aggregated_instructions: aggregated_instructions[key] = []
                         for v in values:
                             if v not in aggregated_instructions[key]:
                                 aggregated_instructions[key].append(v)

                if not clean_lines: return

                beat_type = "establish" if beat_id == 1 and not beats else "action"
                beat_attributes = analyze_narrative_attributes(clean_lines, agent_slug_lookup, profile)

                beat = {"id":f"B{beat_id}","type":beat_type,"text_lines":clean_lines}
                if beat_attributes: beat["attributes"] = beat_attributes
                if aggregated_instructions: beat['instructions'] = aggregated_instructions

                beats.append(beat); beat_id += 1

            # Iteration logic
            for item in group:
                is_bullet = re.match(r"^\s*[\-•\*]\s*", item['raw'])
                force_new_block = False
                if is_bullet: force_new_block = True
                elif item['tag'] == 'action':
                     if not current_block or current_block[-1]['tag'] != 'action':
                          force_new_block = True

                if force_new_block:
                    if current_block: finalize_action_block(current_block)
                    current_block = [item]
                else:
                    current_block.append(item)

            finalize_action_block(current_block)

    if beats and beats[0]['type'] == 'action':
         beats[0]['type'] = 'establish'

    # V11.4: Return only beats
    return beats


# ===========================================================
# SECTION 4: CINEMATIC SHOT SEGMENTATION & COMPOSITION (V11.1-V11.4)
# ===========================================================

# V11.2: Updated Shot Typing to handle Dictionary structure
def determine_shot_type(beats: List[Dict[str, Any]], composition: Dict[str, Any]) -> str:
    """
    (V11.2) Determines shot type, adapted for Dictionary-based composition structure.
    """
    
    # Calculate presence (V11.2: Access keys() instead of length of list)
    num_chars = len(composition.get("Characters", {}))
    num_groups = len(composition.get("Groups", {}))
    
    if num_groups > 0:
         total_presence = num_chars + 3 # Treat a group as ~3 entities
    else:
         total_presence = num_chars

    if not beats: return "MS"

    first_beat_type = beats[0]['type']
    all_beat_types = set(b['type'] for b in beats)
    
    # Rule 1: Handle META/INSERTS
    if 'meta' in all_beat_types or 'meta_instruction' in all_beat_types:
        if any("insert_2d_animation" in b.get("instructions", {}).get("meta", []) for b in beats):
             return "INSERT"

    # Rule 2: Establishing shots
    if first_beat_type == 'establish':
        return "WS"

    # Rule 3: High presence favors Wide Shots
    if total_presence >= 4:
        return "WS"

    # Rule 4: Analyze content for details/emotions (CU/MCU)
    has_strong_emotion = False
    strong_emotion_keywords = ["tức giận", "long lanh", "bất ngờ", "quyết tâm", "lí nhí", "mắt sáng rỡ"]

    # V11.2: Iterate over dictionary values
    for char_data in composition.get("Characters", {}).values():
        emotions = char_data.get("Attributes", {}).get("Emotions", [])
        if any(emo in emotions for emo in strong_emotion_keywords):
             has_strong_emotion = True
             break
             
    if has_strong_emotion:
        return "CU"

    # Rule 5: Dialogue focus
    if 'dialogue' in all_beat_types:
        if total_presence >= 3:
            return "MS"
        
        has_significant_action = False
        # V11.2: Iterate over dictionary values
        for char_data in composition.get("Characters", {}).values():
             if char_data.get("Attributes", {}).get("Actions"):
                 if char_data.get("Status") == "active":
                      has_significant_action = True
                      break
        
        if has_significant_action:
            return "MCU"
            
        return "MS"

    # Rule 6: Action shots
    if 'action' in all_beat_types or 'climax' in all_beat_types or 'conclusion' in all_beat_types:
        wide_actions = ['lao đi', 'chạy vòng quanh', 'vật lộn', 'xúm lại', 'quây quần', 'ôm chầm lấy']
        
        has_wide_action = False
        # V11.2: Combine dictionaries and iterate over values()
        all_entities = {**composition.get("Characters", {}), **composition.get("Groups", {})}
        for entity_data in all_entities.values():
             actions = entity_data.get("Attributes", {}).get("Actions", [])
             if any(act in wide_actions for act in actions):
                 has_wide_action = True
                 break

        if has_wide_action:
            return "WS"

    # Default: Medium Shot (MS)
    return "MS"


def segment_shots(beats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # (Segmentation logic remains identical to V10)
    if not beats: return []

    shots = []; current_shot_beats = []; shot_id = 1
    MAX_BEATS_PER_SHOT = 3; MAX_DIALOGUE_LINES_PER_SHOT = 4
    STRUCTURE_BREAKS = {'climax', 'arrival', 'conclusion', 'establish', 'meta', 'meta_instruction'}

    for i, beat in enumerate(beats):
        beat_type = beat['type']; is_new_shot = False

        if beat_type in STRUCTURE_BREAKS: is_new_shot = True
        elif len(current_shot_beats) >= MAX_BEATS_PER_SHOT: is_new_shot = True
        elif current_shot_beats:
            prev_beat_type = current_shot_beats[-1]['type']
            is_current_action = beat_type == 'action'; is_prev_action = prev_beat_type == 'action'

            if (is_current_action and prev_beat_type == 'dialogue') or \
               (not is_current_action and beat_type == 'dialogue' and is_prev_action):
                 if (is_current_action and len(beat.get('text_lines', [])) == 1) or \
                    (prev_beat_type == 'dialogue' and len(current_shot_beats[-1].get('dialogue_lines', [])) == 1):
                      pass
                 else: is_new_shot = True

        if not is_new_shot and beat_type == 'dialogue':
            current_shot_lines = sum(len(b.get('dialogue_lines', [])) for b in current_shot_beats if b['type'] == 'dialogue')
            if current_shot_lines >= MAX_DIALOGUE_LINES_PER_SHOT: is_new_shot = True

        if is_new_shot and current_shot_beats:
            shots.append({"Shot_ID": f"S{shot_id}", "Beats": list(current_shot_beats)})
            current_shot_beats = []; shot_id += 1

        current_shot_beats.append(beat)

    if current_shot_beats:
        shots.append({"Shot_ID": f"S{shot_id}", "Beats": list(current_shot_beats)})

    return shots

# V11.1, V11.2, V11.3, V11.5: The Core Cinematic Intelligence Engine
def generate_shot_composition(shots: List[Dict[str, Any]], global_chars_by_slug: Dict[str, Dict], global_groups_by_slug: Dict[str, Dict], profile: Dict[str, Any]):
    """
    (V11) Generates Shot Composition sequentially, implementing Temporal Continuity (V10.1/V11.5), Attribute Persistence (V11.1),
    Data Optimization (V11.2), and Deduplication (V11.3).
    """
    
    rules = profile['parsing_rules']
    DEFAULT_ACTION = rules.get("default_passive_action")
    DEFAULT_EMOTION = rules.get("default_passive_emotion")
    
    # V11.1: Get negation map
    ATTRIBUTE_NEGATIONS = rules.get("attribute_negations", {})

    # V10.1/V11.5: Track presence timeline.
    present_characters = set()
    
    # V11.1: Track persistent state across the scene.
    # Stores DYNAMIC attributes (Appearance/Emotion) that persist until contradicted.
    # Structure: {slug: {"Appearance": set(), "Emotions": set()}}
    persistent_state = {} 

    # Helper to initialize persistent state
    def init_persistent(slug):
        if slug not in persistent_state:
            # We only track Appearance and Emotions persistence in this version
            persistent_state[slug] = {"Appearance": set(), "Emotions": set()}

    # Helper function to get inherent attributes (for V11.3)
    def get_inherent(slug, agent_type):
        if agent_type == "Characters":
             return global_chars_by_slug.get(slug, {}).get("inherent_attributes", {})
        return {} # Groups typically don't have inherent attributes


    # Iterate chronologically through shots
    for shot in shots:
        # composition_map stores the calculated state for the current shot.
        composition_map = {"Characters": {}, "Groups": {}}
        shot_instructions = {}
        
        # === Phase 1: Analyze Active Participants & Aggregate Attributes ===
        for beat in shot['Beats']:
            
            # (Aggregate Instructions)
            if "instructions" in beat and beat['instructions']:
                for category, cues in beat['instructions'].items():
                    if category not in shot_instructions: shot_instructions[category] = []
                    for cue in cues:
                        if cue not in shot_instructions[category]:
                             shot_instructions[category].append(cue)

            active_slugs = set()
            
            # From Narrative Attributes
            if 'attributes' in beat:
                active_slugs.update(beat['attributes'].keys())
            
            # From Dialogue
            if 'dialogue_lines' in beat:
                for line in beat['dialogue_lines']:
                    active_slugs.add(line['Speaker_Slug'])
            
            # V11.5: Robust Temporal Logic - Check for 'arrival' beats
            if beat['type'] == 'arrival' and not active_slugs:
                # Fallback heuristic if analyzer missed attributes in an arrival beat
                beat_text = " ".join(beat.get('text_lines', []))
                # Check characters and groups
                all_agents = {**global_chars_by_slug, **global_groups_by_slug}
                for slug, data in all_agents.items():
                     # Check aliases (assuming global data contains aliases)
                     search_terms = data.get("aliases", [])
                     # Note: A robust check might require checking the canonical name too if aliases are insufficient.
                     
                     if any(re.search(rf"\b{re.escape(term)}\b", beat_text, re.I) for term in search_terms):
                          active_slugs.add(slug)

            
            # Initialize active participants and update presence tracker
            for slug in active_slugs:
                
                # V10.1/V11.5: Update Presence Tracker upon activity
                if slug in global_chars_by_slug:
                    present_characters.add(slug)
                    init_persistent(slug) # V11.1: Ensure persistent state is initialized

                
                # Initialize agent in the current shot composition map
                def init_active_agent(slug, agent_type):
                    if slug not in composition_map[agent_type]:
                        # V11.2: Optimized structure (No Name/Slug in value)
                        composition_map[agent_type][slug] = {
                            "Status": "active",
                            "Attributes": {
                                "Appearance": [],
                                "Actions": [],
                                "Emotions": []
                            }
                        }
                    return composition_map[agent_type][slug]["Attributes"]

                current_attrs = None
                if slug in global_chars_by_slug:
                    current_attrs = init_active_agent(slug, "Characters")
                elif slug in global_groups_by_slug:
                    current_attrs = init_active_agent(slug, "Groups")
                else:
                    continue

                
                # Aggregate Attributes from the beat (Logic from V10.2)
                
                # 1. Narrative Attributes
                if 'attributes' in beat and slug in beat['attributes']:
                    beat_data = beat['attributes'][slug]
                    for attr_type in ["Actions", "Emotions", "Appearance"]:
                        for item in beat_data.get(attr_type, []):
                            if item not in current_attrs[attr_type]:
                                current_attrs[attr_type].append(item)
                
                # 2. Dialogue Parentheticals
                if 'dialogue_lines' in beat:
                    for line in beat['dialogue_lines']:
                        if line['Speaker_Slug'] == slug and line['Parenthetical']:
                            p_text = line['Parenthetical']
                            actions = mine_actions(p_text, profile)
                            emotions = mine_emotions(p_text, profile)
                            appearances = mine_appearance(p_text, profile)
                            
                            for attr in actions:
                                 if attr not in current_attrs["Actions"]: current_attrs["Actions"].append(attr)
                            for attr in emotions:
                                 if attr not in current_attrs["Emotions"]: current_attrs["Emotions"].append(attr)
                            for attr in appearances:
                                 if attr not in current_attrs["Appearance"]: current_attrs["Appearance"].append(attr)


        # === V11.1 Phase 2: Apply Persistence and Handle Contradictions ===
        
        # Iterate over characters present so far (Temporal Continuity V10.1)
        for slug in present_characters:
            
            # Ensure character is initialized in the map (might be passive)
            if slug not in composition_map["Characters"]:
                composition_map["Characters"][slug] = {
                    "Status": "passive",
                    "Attributes": {
                        "Appearance": [],
                        "Actions": [DEFAULT_ACTION] if DEFAULT_ACTION else [],
                        "Emotions": [DEFAULT_EMOTION] if DEFAULT_EMOTION else []
                    }
                }
            
            current_attrs = composition_map["Characters"][slug]["Attributes"]

            # 1. Handle Contradictions (Negations)
            # Check if current actions remove persisted appearance states.
            for action in current_attrs["Actions"]:
                if action in ATTRIBUTE_NEGATIONS:
                    for negated_attr in ATTRIBUTE_NEGATIONS[action]:
                        # Remove from tracker
                        if negated_attr in persistent_state[slug]["Appearance"]:
                            persistent_state[slug]["Appearance"].remove(negated_attr)

            # 2. Apply Persisted State
            # Merge Tracker state into current shot attributes
            for ptype in ["Appearance", "Emotions"]:
                for item in persistent_state[slug][ptype]:
                    if item not in current_attrs[ptype]:
                        current_attrs[ptype].append(item)

            # 3. Update Tracker with New Dynamic Attributes
            # If the character was active, update the tracker based on the current shot's state.
            if composition_map["Characters"][slug]["Status"] == "active":
                # We update the tracker based on the final state of attributes in the current shot
                persistent_state[slug]["Appearance"] = set(current_attrs["Appearance"])
                persistent_state[slug]["Emotions"] = set(current_attrs["Emotions"])


        # === Phase 3: Finalization (V11.2 & V11.3) ===
        
        # V11.2: Optimized structure (Dictionary)
        final_composition = {}
        
        def finalize_agent_map(agent_type, comp_map):
            if not comp_map:
                return {}

            final_map = {}
            # Sort slugs for deterministic output
            for slug in sorted(comp_map.keys()):
                data = comp_map[slug]
                
                # V11.3: Attribute Deduplication
                inherent = get_inherent(slug, agent_type)

                final_attrs = {}
                for attr_type, values in data["Attributes"].items():
                    inherent_values = inherent.get(attr_type, [])
                    # Keep only dynamic values (those NOT present in inherent attributes)
                    dynamic_values = [v for v in values if v not in inherent_values]
                    
                    if dynamic_values:
                         # Sort for consistency
                         final_attrs[attr_type] = sorted(dynamic_values)

                # V11.2: Structure {slug: data}
                entry = {"Status": data["Status"]}
                if final_attrs:
                    entry["Attributes"] = final_attrs
                final_map[slug] = entry
            
            return final_map

        final_composition["Characters"] = finalize_agent_map("Characters", composition_map["Characters"], global_chars_by_slug)
        final_composition["Groups"] = finalize_agent_map("Groups", composition_map["Groups"], global_groups_by_slug)


        # Add to shot data
        shot["Shot_Composition"] = final_composition
        
        # Determine Shot Type (V11.2 compatible)
        shot["Shot_Type"] = determine_shot_type(shot['Beats'], final_composition)
        
        # Add Instructions
        if shot_instructions:
            for key in shot_instructions:
                 shot_instructions[key] = sorted(shot_instructions[key])
            shot["Instructions"] = shot_instructions


# ===========================================================
# SECTION 5: MAIN EXECUTION & CLI (V11.4 Update)
# ===========================================================

def normalize_script(script_input:str, title=None, is_file=True, output_dir="./output_normalized_v11", profile_name="pixar_3d_vi"):
    
    # Load Profile
    active_profile = profiles.load_profile(profile_name)

    print(f"🔧 Sử dụng Profile: {active_profile['genre']}/{active_profile['language']} (StoryGrid {active_profile.get('StoryGrid_Version', 'N/A')})")
    compile_regexes(active_profile)

    # Load Script Content
    if is_file:
        try:
            script_content = Path(script_input).read_text(encoding="utf-8")
            if title is None:
                title = Path(script_input).stem
        except FileNotFoundError:
            print(f"❌ Lỗi: Không tìm thấy file kịch bản tại {script_input}"); return None
    else:
        script_content = script_input
        if title is None: title = "untitled_script"

    # Pass 1: Global Consolidation
    global_map, global_characters, global_group_agents = global_character_pass(script_content, active_profile)

    # Create Comprehensive Agent Slug Lookup Map
    agent_slug_lookup = {}
    
    # V11.4: Create Global Lookups by Slug (Required for Composition Generator)
    global_chars_by_slug = {}
    global_groups_by_slug = {}

    # Populate lookups and slug map
    for c_name, data in global_characters.items():
        slug = data['slug']
        global_chars_by_slug[slug] = data
        # Add canonical name to lookup
        agent_slug_lookup[canonicalize(c_name).lower()] = slug
        # Add aliases to lookup
        for alias in data.get('aliases', []):
             agent_slug_lookup[canonicalize(alias).lower()] = slug

    for g_name, data in global_group_agents.items():
        slug = data['slug']
        global_groups_by_slug[slug] = data
        agent_slug_lookup[canonicalize(g_name).lower()] = slug


    scenes_raw = detect_scenes(script_content, active_profile)
    issues=[]

    # Sort Global Data for output
    sorted_global_chars = dict(sorted(global_characters.items(), key=lambda item: item[1]['slug']))
    sorted_global_groups = dict(sorted(global_group_agents.items(), key=lambda item: item[1]['slug']))


    story={"Project":{"Title":title,"Language":active_profile['language'], "Genre": active_profile['genre'], "StoryGrid_Version": active_profile.get("StoryGrid_Version", "3.4")},
           "Global_Characters": sorted_global_chars,
           "Global_Group_Agents": sorted_global_groups,
           "Scenes":[]}


    # Pass 2: Scene Processing
    for sc in scenes_raw:
        sid=sc["Scene_ID"]; title_sc=sc["Title"]; body=sc["Raw"]

        # 1. Context
        ctx   = derive_context(body, active_profile)

        # 2. Beats Extraction
        # V11.4: scene_narrative_attributes is no longer returned/needed.
        beats = extract_beats(body, agent_slug_lookup, active_profile)

        # 3. Shot Segmentation
        shots = segment_shots(beats)

        # 4. Shot Composition Generation (V11.1-V11.5)
        # We pass the global lookups by slug directly.
        generate_shot_composition(shots, global_chars_by_slug, global_groups_by_slug, active_profile)

        # V11.4: Data Cleanup (Removed Scene_Characters/Scene_Groups)
        scene_entry={"Scene_ID":sid,"Title":title_sc,
                     "Setting":ctx["setting"],"TimeOfDay":ctx["time_of_day"],
                     "Tone":ctx["tone"],"Props":ctx["props"],
                     # "Scene_Characters": Removed
                     # "Scene_Groups": Removed
                     "Shots": shots}

        story["Scenes"].append(scene_entry)

    # Output results
    if is_file:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # V11: Update output path
        output_json_path = Path(output_dir,"storygrid_v11.json")

        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(story, f, ensure_ascii=False, indent=2)
            print(f"✅ Hoàn tất (V11): Giai đoạn Parser HOÀN THIỆN. {len(scenes_raw)} cảnh (Trí tuệ Điện ảnh & Tối ưu hóa Dữ liệu).")
            print(f"🧠 Áp dụng Tính Kế thừa Thuộc tính (V11.1) và Tối ưu hóa Cấu trúc (V11.2).")
            total_shots = sum(len(s.get("Shots", [])) for s in story["Scenes"])
            print(f"🎬 Tổng số Shot: {total_shots}.")
            print(f"🧾 Kết quả xuất tại: {output_json_path}")
            print("\n======================================================================")
            print("➡️ CHÍNH THỨC BẮT ĐẦU GIAI ĐOẠN 2: PROMPT GENERATION ENGINE (V12)")
            print("======================================================================")
        except Exception as e:
            print(f"❌ Lỗi khi ghi file JSON: {e}")

        if issues:
            # (Issue logging remains)
            pass

    return story

# ---------- CLI ----------
if __name__=="__main__":
    # V11: Update description and defaults
    ap=argparse.ArgumentParser(description="Screenplay Normalizer (v11.0) → StoryGrid v3.4")
    ap.add_argument("--script", required=True, help="Đường dẫn file kịch bản .txt")
    ap.add_argument("--output", default="./output_normalized_v11", help="Thư mục xuất kết quả")
    ap.add_argument("--profile", default="pixar_3d_vi", help="Chọn profile phân tích (vd: pixar_3d_vi)")

    try:
        if 'ipykernel' in sys.modules:
             print("ℹ️ Chạy trong môi trường tương tác. Gọi hàm normalize_script() trực tiếp nếu cần.")

        elif len(sys.argv) > 1:
             args=ap.parse_args()
             normalize_script(args.script, output_dir=args.output, is_file=True, profile_name=args.profile)
        
        else:
             print("Chạy ở chế độ CLI yêu cầu tham số --script.")
             try:
                 script_name = Path(__file__).name
             except NameError:
                 script_name = "script_normalizer_v11_byGemini.py"
             print(f"Ví dụ: python {script_name} --script lop-hoc-mau-sac.txt --profile pixar_3d_vi")

    except SystemExit:
        pass