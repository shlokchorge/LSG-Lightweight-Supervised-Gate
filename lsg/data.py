"""
Data loading and labeling for LSG.
Domains:
  - "taskoriented": Bitext customer-support dataset (slot/intent-rich)
  - "personachat":  AlekseyKorshuk/persona-chat (chit-chat)

STORE=1: contains a named entity, slot value, stated preference, or specific intent.
IGNORE=0: filler, backchannel, generic question with no extractable fact.
"""
import re
import pandas as pd
from datasets import load_dataset

_STORE_RE = re.compile(
    r"\b(my name is|i am|i'm|i live|i work|i like|i love|i hate|i prefer|"
    r"i have|i want|i need|i'm looking|order|cancel|refund|invoice|account|"
    r"delivery|shipping|address|phone|email|password|subscription|payment|"
    r"charge|bill|hotel|restaurant|train|taxi|book|reservation|arrive|depart|"
    r"check.?in|check.?out|people|nights?|rooms?|price|cost|cheap|expensive)\b",
    re.IGNORECASE,
)
_IGNORE_RE = re.compile(
    r"^(ok|okay|yes|no|sure|thanks|thank you|great|alright|"
    r"hello|hi|bye|goodbye|got it|sounds good|perfect|nice|cool|wow|"
    r"can you|could you|what is|what are|do you|is there|really|oh|ah|"
    r"that('s| is) (great|nice|cool|interesting|awesome|good|fun))[,\s!?.]*$",
    re.IGNORECASE,
)
_NER_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b")
_SLOT_RE = re.compile(r"\{\{[^}]+\}\}")

# Persona sentences are first-person factual statements → always STORE
_PERSONA_RE = re.compile(
    r"^i (am|'m|have|like|love|hate|prefer|work|live|own|play|enjoy|go|do|"
    r"used to|want|need|grew up|was born|studied|graduated|drive|eat|drink|"
    r"speak|know|believe|think|feel)\b",
    re.IGNORECASE,
)


def _label(text: str) -> int:
    text = text.strip()
    if _IGNORE_RE.match(text):
        return 0
    if _PERSONA_RE.match(text):
        return 1
    if _SLOT_RE.search(text) or _STORE_RE.search(text):
        return 1
    if _NER_RE.search(text):
        return 1
    # Short utterances with no signal → IGNORE
    if len(text.split()) <= 5:
        return 0
    return 0


def _label_taskoriented(instruction: str, response: str) -> tuple[int, int]:
    instr = instruction.strip()
    instr_label = 1 if (_SLOT_RE.search(instr) or _STORE_RE.search(instr)) else 0
    # Agent responses restate/confirm — never assert new storable facts
    return instr_label, 0


def load_taskoriented(max_samples: int = 2000) -> pd.DataFrame:
    """Bitext customer-support: instruction (user) + response (agent)."""
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    rows = []
    for item in ds:
        il, rl = _label_taskoriented(item["instruction"], item["response"])
        for text, lbl in ((item["instruction"], il), (item["response"], rl)):
            if text and text.strip():
                rows.append({"text": text.strip(), "domain": "taskoriented", "label": lbl})
        if len(rows) >= max_samples:
            break
    return pd.DataFrame(rows[:max_samples])


def load_personachat(max_samples: int = 2000) -> pd.DataFrame:
    """
    PersonaChat: persona sentences (always STORE) + dialogue history utterances.
    Returns a single DataFrame with 'text', 'domain', 'label'.
    """
    ds = load_dataset("AlekseyKorshuk/persona-chat", split="train")
    rows = []
    seen = set()

    for item in ds:
        # Persona sentences are explicit first-person facts → STORE
        for p in item.get("personality", []):
            p = p.strip()
            if p and p not in seen:
                seen.add(p)
                rows.append({"text": p, "domain": "personachat", "label": 1})

        # Dialogue utterances — label by heuristic
        for utt in item["utterances"]:
            for text in utt.get("history", []) + utt.get("candidates", [])[-1:]:
                if isinstance(text, str) and text.strip() and text not in seen:
                    seen.add(text)
                    rows.append({
                        "text": text.strip(),
                        "domain": "personachat",
                        "label": _label(text),
                    })
        if len(rows) >= max_samples:
            break

    return pd.DataFrame(rows[:max_samples])
