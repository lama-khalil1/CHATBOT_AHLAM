from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ---------- تحميل config ----------
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

TOP_K = int(config.get("top_k", 7))

# ---------- تحميل chunks ----------
with open(config["chunks_file"], "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ---------- تحميل FAISS index ----------
index = faiss.read_index(config["faiss_index_file"])

# ---------- تحميل embedder (من مجلد محلي) ----------
embedder = SentenceTransformer("models/paraphrase-multilingual-MiniLM-L12-v2")

# ---------- OpenAI Client ----------
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ✅ تشخيص مهم: لازم يتطابق عدد الـ vectors مع عدد الـ chunks
print("chunks count:", len(chunks))
print("faiss ntotal:", getattr(index, "ntotal", "unknown"))
print("faiss dim:", getattr(index, "d", "unknown"))
print("embedder dim:", embedder.get_sentence_embedding_dimension())


# ---------- FastAPI ----------
app = FastAPI(title="RAG QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"status": "ok", "ui": "/ui/index.html", "docs": "/docs"}


def _normalize_chunk(item):
    if isinstance(item, dict):
        return {
            "text": item.get("text", ""),
            "source": item.get("source", ""),
            "page": item.get("page", ""),
        }
    return {"text": str(item), "source": "", "page": ""}

@app.post("/ask")
def ask(q: Question):
    query_vec = np.array(embedder.encode([q.question]), dtype="float32")

    top_k = min(TOP_K, len(chunks))
    if top_k <= 0:
        return {"question": q.question, "final_answer": "", "answer_text": "ما فيه chunks.", "sources": []}

    D, I = index.search(query_vec, top_k)

    # فلترة indices الغلط (-1 أو خارج مدى chunks)
    hits = []
    bad = 0
    for idx in I[0]:
        idx = int(idx)
        if idx < 0 or idx >= len(chunks):
            bad += 1
            continue
        hits.append(_normalize_chunk(chunks[idx]))

    if not hits:
        return {
            "question": q.question,
            "final_answer": "",
            "answer_text": f"ما لقيت نتائج صالحة. (indices خارج المدى: {bad})",
            "sources": [],
        }

    # نص جاهز للعرض (chunks)
    answer_text = "\n\n---\n\n".join(
        [
            f"({h.get('source','')} ص{h.get('page','')})\n{h.get('text','')}".strip()
            if (h.get("source") or h.get("page"))
            else h.get("text", "").strip()
            for h in hits
            if h.get("text", "").strip()
        ]
    )

    # ---------- صياغة جواب من OpenAI (نفس النوتبوك) ----------
    contexts = [h.get("text", "").strip() for h in hits if h.get("text", "").strip()]
    context_text = "\n\n".join(contexts)

    prompt = f"""
أنتِ مختصة في التربية المبكرة ورياض الأطفال.

المهمة:
إعادة صياغة وتجميع النصوص التالية في جواب واحد متكامل يجيب بشكل مباشر وواضح على السؤال التالي:
"{q.question}"

القواعد:
- استخدمي فقط المعلومات الواردة في النصوص.
- لا تضيفي أي معلومة من خارج النص.
- لا تذكري أن هذه النصوص أو مصادر.
- لا تكرري الأفكار المتشابهة، بل ادمجيها بسلاسة.
- إن وُجد تعارض، قدّمي الصياغة الأكثر عمومية دون ترجيح.
- اكتبي بأسلوب تربوي رسمي واضح.

طريقة العرض:
- قسّمي الجواب إلى فقرات مترابطة.
- استخدمي عناوين فرعية عند الحاجة.
- قدّمي إجابة وافية ومفصلة دون إطالة غير ضرورية.
- لا تذكري مقدمة، عرض، خاتمة.

النصوص:
{context_text}

الجواب النهائي:
"""

    final_answer = ""
    try:
        resp = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            max_output_tokens=3000,  
        )
        
        final_answer = resp.output_text
    except Exception as e:
        final_answer = f"LLM error: {str(e)}"

    return {
        "question": q.question,
        "final_answer": final_answer,
        "answer_text": answer_text,
        "sources": hits,
    }

