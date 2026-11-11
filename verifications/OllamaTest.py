"""
testar_ollama.py - testa a ligação a um servidor Ollama remoto
    python testar_ollama.py
"""
import json
import time
import urllib.request
import urllib.error

IP = "172.20.10.15"
PORTA = 11343
MODELO = "qwen3.8:27b"
TIMEOUT = 300          # o primeiro pedido carrega o modelo para a GPU e pode demorar

BASE = f"http://{IP}:{PORTA}"


def get(caminho):
    with urllib.request.urlopen(BASE + caminho, timeout=15) as r:
        return json.loads(r.read())


def post(caminho, dados):
    req = urllib.request.Request(BASE + caminho, data=json.dumps(dados).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


# 1. O servidor responde?
try:
    print("Versão do Ollama:", get("/api/version").get("version"))
except urllib.error.URLError as e:
    raise SystemExit(f"Sem ligação a {BASE}: {e.reason}")

# 2. Que modelos existem?
modelos = [m["name"] for m in get("/api/tags").get("models", [])]
print("Modelos disponíveis:", modelos)
if MODELO not in modelos:
    raise SystemExit(f"O modelo '{MODELO}' não está na lista. Usa um dos nomes acima.")

# 3. Pedido de teste
print(f"\nA pedir resposta ao {MODELO} (o primeiro pedido carrega o modelo)...")
t0 = time.time()
r = post("/api/chat", {
    "model": MODELO,
    "messages": [{"role": "user", "content": "Responde numa frase: qual é a capital de Portugal?"}],
    "stream": False,
    "options": {"temperature": 0},
})
dt = time.time() - t0
print("Resposta:", r["message"]["content"].strip())

n = r.get("eval_count", 0)
t_ger = r.get("eval_duration", 0) / 1e9
print(f"\nTempo total: {dt:.1f} s | tokens gerados: {n} | "
      f"velocidade: {n / t_ger:.1f} tokens/s" if t_ger else f"\nTempo total: {dt:.1f} s")

# 4. O modelo está todo na GPU?
for m in get("/api/ps").get("models", []):
    if m["name"] == MODELO:
        total, vram = m.get("size", 0), m.get("size_vram", 0)
        print(f"Memória do modelo: {total / 1e9:.1f} GB | na GPU: {vram / 1e9:.1f} GB "
              f"({100 * vram / total:.0f}%)" if total else "")