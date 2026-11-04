# app.py - Chainlit + BM25 + Gemini
import os, json, re, math, requests
from collections import Counter
from pathlib import Path
import chainlit as cl

STOP={"الي","علي","اذا","هذا","هذه","ذلك","تلك","التي","الذي","الذين","كان","يكون","بين","غير","عند","كما","بعض","قال","قوله","وجل","ايضا","كذا","كل","عز"}

def norm(t):
    t=re.sub(r'[\u0617-\u061A\u064B-\u0652]','',t)
    t=re.sub(r'[ـ]','',t)
    t=re.sub(r'[أإآ]','ا',t)
    t=re.sub(r'ى','ي',t)
    t=re.sub(r'ة','ه',t)
    t=re.sub(r'ؤ','و',t)
    t=re.sub(r'ئ','ي',t)
    t=re.sub(r'\s+',' ',t)
    return t.strip()

def tokens(t):
    toks=re.findall(r'[\u0621-\u064A\u0660-\u06690-9a-zA-Z]+',norm(t))
    return [w for w in toks if w not in STOP and len(w)>2]

def load_data():
    # dukung multi-kitab nanti: data/*.json
    files = list(Path("data").glob("*.json"))
    if not files:
        if Path("13550.json").exists():
            files = [Path("13550.json")]
        else:
            return [], [], [], Counter(), 0, 1
    DATA=[]
    for fp in files:
        DATA.extend(json.loads(fp.read_text(encoding="utf-8")))
    docs=[Counter(tokens(d.get('teks_bersih',''))) for d in DATA]
    dl=[sum(c.values()) or 1 for c in docs]
    df=Counter()
    for c in docs:
        for w in c: df[w]+=1
    N=len(DATA)
    avg=sum(dl)/N if N else 1
    return DATA, docs, dl, df, N, avg

DATA, DOCS, DL, DF, N, AVG = load_data()
print(f"Loaded {N} chunk")

def search(q,k=3):
    qt=tokens(q)
    scores=[0.0]*N
    k1=1.2; b=0.75
    for qi in qt:
        if qi not in DF: continue
        idf=math.log((N-DF[qi]+0.5)/(DF[qi]+0.5)+1)
        for i in range(N):
            tf=DOCS[i].get(qi,0)
            if not tf: continue
            denom=tf+k1*(1-b+b*DL[i]/AVG)
            scores[i]+=idf*(tf*(k1+1)/denom)
    rank=sorted(range(N),key=lambda i:scores[i],reverse=True)[:k]
    return [(DATA[i],scores[i],qt) for i in rank if scores[i]>0]

def gemini(prompt, key, model="gemini-2.0-flash"):
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    r=requests.post(url,json={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.3}},timeout=60)
    j=r.json()
    if "candidates" in j:
        return j['candidates'][0]['content']['parts'][0]['text']
    raise Exception(str(j)[:1000])

@cl.set_starters
async def starters():
    return [
        cl.Starter(label="Apa itu Istihsan?", message="Apa itu istihsan menurut Thufi?"),
        cl.Starter(label="Perintah = Wajib?", message="Apakah perintah menunjukkan wajib menurut Thufi? berikan dalilnya"),
        cl.Starter(label="Apa itu Qiyas?", message="Bagaimana Thufi menjelaskan qiyas?"),
    ]

@cl.on_chat_start
async def on_start():
    await cl.Message(content=f"""Ahlan! Saya membaca **{N} potongan** kitab **الإشارات الإلهية - الطوفي**.

Silakan tanya pakai Indonesia. Saya carikan kutipan Arab aslinya + sumbernya di panel kanan.""").send()

@cl.on_message
async def main(message: cl.Message):
    q = message.content.strip()
    KEY = os.environ.get("GEMINI_API_KEY","")
    MODEL = os.environ.get("GEMINI_MODEL","gemini-2.0-flash")
    if not KEY:
        await cl.Message(content="`GEMINI_API_KEY` belum disetting di Secrets Space.").send()
        return
    try:
        async with cl.Step(name="Menerjemahkan ke Arab...", type="tool") as step:
            kw = gemini(f"Ubah pertanyaan ini jadi 3-7 kata kunci ARAB untuk pencarian kitab ushul fiqih. Jawab HANYA kata kunci Arab.\nPertanyaan: {q}", KEY, MODEL)
            step.output = kw

        async with cl.Step(name=f"📖 Mencari: {kw}", type="retrieval") as step:
            hasil = search(kw+" "+q, k=3)
            if not hasil:
                await cl.Message(content="Tidak ditemukan dalam database kitab ini. Coba kata kunci Arab lain.").send()
                return
            step.output = f"Ketemu {len(hasil)} sumber, score {hasil[0][1]:.2f}"

        ctx="\n\n".join([f"[{d['kitab']}]\n{d['teks_asli']}" for d,_,_ in hasil])
        prompt=f"""Kamu asisten penuntut ilmu. Jawab HANYA dari KONTEKS di bawah.
Pakai Bahasa Indonesia sopan, sertakan kutipan Arab asli, sebut sumber.
Jika tidak ada di konteks, katakan tidak ditemukan, jangan mengarang.
Ini hanya bantu pencarian, bukan fatwa.

KONTEKS:
{ctx}

PERTANYAAN:
{q}

JAWABAN:"""
        ans = gemini(prompt, KEY, MODEL)

        elements=[]
        for i,(d,s,_) in enumerate(hasil,1):
            elements.append(cl.Text(name=f"Sumber {i} score {s:.1f}", content=f"Kitab: {d.get('kitab')}\nBab: {d.get('bab')}\n\n{d.get('teks_asli')}", display="side"))

        await cl.Message(content=ans, elements=elements).send()
    except Exception as e:
        await cl.Message(content=f"Error: {e}\n\nJika 404, ganti GEMINI_MODEL jadi gemini-1.5-flash di Settings.").send()
