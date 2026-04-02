import re
import html

def clean_transcription(text):
    parasites = ["sous-titres", "sous titres", "amara.org", "sous-titre", "sous titre", "communauté"]
    if not text:
        return ""
    lowered = text.lower()
    if any(p in lowered for p in parasites):
        return ""
    return text.strip()


def remove_emojis(text):
    return re.sub(r"[^\w\s.,!?;:'\"()\-/%€$\/]", "", text)


def sanitize_for_tts(text, max_len=3500):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > max_len:
        cut = text[:max_len]
        m = re.search(r'([.!?])(?=[^.!?]*$)', cut[::-1])
        text = cut
    return text


def convert_morse_block(block):
    pron = []
    for ch in block:
        if ch == ".":
            pron.append("point")
        elif ch == "-":
            pron.append("tiret")
    return " ".join(pron)


def prepare_tts(text):
    if not text:
        return ""
    t = html.unescape(text)
    t = t.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.strip()
    t = re.sub(r'\s*/\s*', ' / ', t)

    def repl(m):
        return convert_morse_block(m.group(0))

    processed = re.sub(r'(?<![\w./-])([.\-]{1,20})(?![\w./-])', repl, t)
    processed = processed.replace(' / ', ' slash ')
    return processed


