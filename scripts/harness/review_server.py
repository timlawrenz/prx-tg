#!/usr/bin/env python3
"""Blind-review web UI (Phase 1) — dependency-free, localhost only.

Serves pairs from a pool.json as neutral URLs (/img/<pair_id>/<l|r>.png) so
filenames never leak which side is which. Votes POST as JSONL in the exact
format review_aggregate.py consumes: {"pair_id","choice","rater"}.

USAGE:
    .venv/bin/python3 scripts/harness/review_server.py \
        --pool research/avenues/review_10k/realism/pool.json \
        --rater tim --out research/avenues/votes/dip-10k/tim.jsonl \
        --port 8765
    then open http://localhost:8765
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Blind realism review — prx-tg</title>
<style>
:root{--bg:#0f1115;--panel:#181b22;--border:#2a2f3a;--text:#e8eaf0;--dim:#9aa3b2;--accent:#C9A85C}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:16px/1.5 system-ui,sans-serif;
     display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:24px 16px}
h1{font-size:20px;font-weight:600;letter-spacing:.02em}
.sub{color:var(--dim);font-size:13px;margin-top:4px}
#progress{width:min(720px,90vw);height:4px;background:var(--border);border-radius:2px;margin:20px 0 8px}
#bar{height:100%;width:0;background:var(--accent);border-radius:2px;transition:width .2s}
#count{color:var(--dim);font-size:13px;margin-bottom:16px}
#stage{display:flex;gap:12px;align-items:stretch;justify-content:center;width:min(1100px,94vw)}
.side{flex:1;min-width:0;background:var(--panel);border:1px solid var(--border);border-radius:10px;
      overflow:hidden;cursor:pointer;position:relative;transition:border-color .15s}
.side:hover{border-color:var(--accent)}
.side img{width:100%;height:auto;display:block}
.side .tag{position:absolute;top:8px;left:8px;background:rgba(15,17,21,.82);color:var(--dim);
           font-size:11px;padding:2px 8px;border-radius:4px;letter-spacing:.04em}
.side.chosen{border-color:var(--accent);outline:2px solid var(--accent)}
#controls{display:flex;gap:10px;margin-top:18px}
button{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:8px;
       padding:10px 18px;font-size:14px;cursor:pointer;transition:border-color .15s}
button:hover:not(:disabled){border-color:var(--accent)}
button:disabled{opacity:.4;cursor:default}
kbd{background:var(--border);border-radius:4px;padding:1px 6px;font-size:12px;color:var(--dim)}
#done{display:none;text-align:center;padding:60px 0;color:var(--text)}
#done .big{font-size:22px;margin-bottom:8px}
#done button{margin-top:16px}
</style></head><body>
<h1>Blind realism review</h1>
<div class="sub">Which side looks more like a real photograph? Click an image or use keys.</div>
<div id="progress"><div id="bar"></div></div>
<div id="count"></div>
<div id="stage"></div>
<div id="controls">
  <button id="prev" disabled>← prev</button>
  <button id="tie" disabled>Tie (T)</button>
  <button id="restart">restart</button>
</div>
<div id="done">
  <div class="big">All pairs reviewed — thanks!</div>
  <div class="sub" id="summary"></div>
  <button id="reviewAgain">review again anyway</button>
</div>
<script>
let pairs=[], idx=0, votes={};
async function boot(){
  const r=await fetch('/api/pairs');
  pairs=await r.json();
  try{const v=(await (await fetch('/api/votes')).json()).votes; votes=Object.fromEntries(v.map(x=>[x.pair_id,x.choice]));}catch(e){}
  render();
}
function render(){
  const n=pairs.length, done=Object.keys(votes).length;
  document.getElementById('bar').style.width=(n?done/n*100:0)+'%';
  document.getElementById('count').textContent=`${done} / ${n} reviewed`;
  if(idx>=n){document.getElementById('stage').innerHTML='';document.getElementById('done').style.display='block';
    document.getElementById('summary').textContent=`${done}/${n} voted.`;
    document.getElementById('prev').disabled=true;document.getElementById('tie').disabled=true;return;}
  document.getElementById('done').style.display='none';
  const p=pairs[idx];
  const prevVote=votes[p.pair_id];
  document.getElementById('stage').innerHTML=
    `<div class="side" id="L" data-side="L"><span class="tag">A</span><img src="${p.left}"></div>`+
    `<div class="side" id="R" data-side="R"><span class="tag">B</span><img src="${p.right}"></div>`;
  ['L','R'].forEach(s=>{const el=document.getElementById(s);
    if(prevVote===s)el.classList.add('chosen');
    el.onclick=()=>vote(s);});
  document.getElementById('prev').disabled=idx===0;
  document.getElementById('tie').disabled=false;
}
async function vote(choice){
  const p=pairs[idx];
  votes[p.pair_id]=choice;
  await fetch('/vote',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({pair_id:p.pair_id,choice})});
  idx++; render();
}
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft'||e.key.toLowerCase()==='a')vote('L');
  else if(e.key==='ArrowRight'||e.key.toLowerCase()==='d')vote('R');
  else if(e.key.toLowerCase()==='t'||e.key==='ArrowDown')vote('T');
  else if(e.key==='Backspace'){if(idx>0)idx--;render();}
});
document.getElementById('prev').onclick=()=>{if(idx>0)idx--;render();};
document.getElementById('tie').onclick=()=>vote('T');
document.getElementById('restart').onclick=()=>{idx=0;render();};
document.getElementById('reviewAgain').onclick=()=>{idx=0;render();};
boot();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    pool = None
    pairs = None
    out_path = None
    rater = None

    def log_message(self, *a):
        pass  # noise off

    def _send(self, code, body, ctype="application/json"):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/pairs":
            # Blind by construction: only pair_id + neutral image URLs, no ground truth.
            data = [{"pair_id": p["pair_id"],
                     "left": f"/img/{p['pair_id']}/l.png",
                     "right": f"/img/{p['pair_id']}/r.png"}
                    for p in self.pairs]
            self._send(200, json.dumps(data).encode())
        elif self.path == "/api/votes":
            body = []
            if self.out_path.exists():
                body = [json.loads(l) for l in self.out_path.read_text().splitlines() if l.strip()]
            self._send(200, json.dumps({"votes": body}).encode())
        elif self.path.startswith("/img/"):
            parts = self.path.split("/")  # ['', 'img', <pair_id>, 'l.png']
            if len(parts) != 4:
                self._send(404, b"not found"); return
            _, _, pid, fname = parts
            side = fname[0]  # 'l' or 'r'
            match = next((p for p in self.pairs if p["pair_id"] == pid), None)
            if not match:
                self._send(404, b"no pair"); return
            key = "left" if side == "l" else "right"
            path = Path(match[key])
            if not path.exists():
                self._send(404, b"no image"); return
            with open(path, "rb") as f:
                self._send(200, f.read(), "image/png")
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if self.path != "/vote":
            self._send(404, b"not found"); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, b"bad json"); return
        pid, choice = body.get("pair_id"), body.get("choice")
        valid = {p["pair_id"] for p in self.pairs}
        if pid not in valid or choice not in ("L", "R", "T", "S"):
            self._send(400, b"bad vote"); return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent per pair_id: drop prior vote for this pair, append new one.
        lines = []
        if self.out_path.exists():
            lines = [l for l in self.out_path.read_text().splitlines() if l.strip()
                     and json.loads(l)["pair_id"] != pid]
        lines.append(json.dumps({"pair_id": pid, "choice": choice, "rater": self.rater}))
        self.out_path.write_text("\n".join(lines) + "\n")
        self._send(200, json.dumps({"ok": True}).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--rater", default="tim")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text())
    Handler.pool = pool
    Handler.pairs = pool["pairs"]
    Handler.out_path = args.out.resolve()
    Handler.rater = args.rater

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Blind review UI: http://0.0.0.0:{args.port}  (LAN: http://192.168.86.49:{args.port})")
    print(f"  pairs: {len(pool['pairs'])}   rater: {args.rater}")
    print(f"  votes -> {Handler.out_path}")
    print("  keys: ←/a = left, →/d = right, t = tie, backspace = back")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()