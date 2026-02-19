// frontend/src/App.tsx
import React, { useEffect, useRef, useState } from "react";

// function clamp(v: number, a: number, b: number) { return Math.max(a, Math.min(b, v)); }

export default function App() {
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [imgUrl, setImgUrl] = useState<string>("");
  const [thr, setThr] = useState<number>(128); // 0..255
  const [probUrl, setProbUrl] = useState<string>("");

  const imgRef = useRef<HTMLImageElement | null>(null);
  const scribbleRef = useRef<HTMLCanvasElement | null>(null);
  const outRef = useRef<HTMLCanvasElement | null>(null);

  const [drawing, setDrawing] = useState(false);
  const last = useRef<{x:number,y:number} | null>(null);

  // 画像ロード時にCanvasを同サイズに
  useEffect(() => {
    const img = imgRef.current;
    const sc = scribbleRef.current;
    const out = outRef.current;
    if (!img || !sc || !out) return;

    const onLoad = () => {
      sc.width = img.naturalWidth;
      sc.height = img.naturalHeight;
      out.width = img.naturalWidth;
      out.height = img.naturalHeight;

      // スクリブル初期化
      const ctx = sc.getContext("2d")!;
      ctx.clearRect(0,0,sc.width,sc.height);
      // 出力初期化
      const octx = out.getContext("2d")!;
      octx.clearRect(0,0,out.width,out.height);
    };

    img.addEventListener("load", onLoad);
    return () => img.removeEventListener("load", onLoad);
  }, [imgUrl]);

  const onUpload = (f: File) => {
    setImgFile(f);
    const u = URL.createObjectURL(f);
    setImgUrl(u);
    setProbUrl("");
  };

  const drawLine = (x:number, y:number) => {
    const sc = scribbleRef.current!;
    const ctx = sc.getContext("2d")!;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgb(255,0,0)"; // 赤で描く（バック側が検出）
    ctx.lineWidth = 12; // 太めでもOK（サーバ側で線部分のみ抽出）
    ctx.beginPath();
    const p = last.current;
    if (p) ctx.moveTo(p.x, p.y);
    else ctx.moveTo(x, y);
    ctx.lineTo(x, y);
    ctx.stroke();
    last.current = {x,y};
  };

  const getPos = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const sc = scribbleRef.current!;
    const rect = sc.getBoundingClientRect();
    // 表示サイズと実解像度が違うのでスケール変換
    const sx = sc.width / rect.width;
    const sy = sc.height / rect.height;
    return {
      x: (e.clientX - rect.left) * sx,
      y: (e.clientY - rect.top) * sy,
    };
  };

  const clearScribble = () => {
    const sc = scribbleRef.current!;
    sc.getContext("2d")!.clearRect(0,0,sc.width,sc.height);
    setProbUrl("");
    const out = outRef.current!;
    out.getContext("2d")!.clearRect(0,0,out.width,out.height);
  };

  const predict = async () => {
    if (!imgFile || !scribbleRef.current) return;
    const sc = scribbleRef.current;

    const blob: Blob = await new Promise((resolve) =>
      sc.toBlob((b) => resolve(b!), "image/png")
    );

    const fd = new FormData();
    fd.append("image", imgFile);
    fd.append("scribble", new File([blob], "scribble.png", { type: "image/png" }));
    fd.append("params", JSON.stringify({
      iters: 600,
      fg_perc: 70,
      thr: 0.5,
      max_side: 1024
    }));

    const res = await fetch("http://127.0.0.1:8000/api/predict", { method: "POST", body: fd });
    if (!res.ok) {
      alert("predict failed");
      return;
    }
    const outBlob = await res.blob(); // prob png
    const url = URL.createObjectURL(outBlob);
    setProbUrl(url);
  };

  // prob png を out canvas に二値化表示（thr変更で即時反映）
  useEffect(() => {
    if (!probUrl || !outRef.current) return;

    const img = new Image();
    img.onload = () => {
      const out = outRef.current!;
      const ctx = out.getContext("2d")!;
      ctx.clearRect(0,0,out.width,out.height);
      // probを読み込み
      ctx.drawImage(img, 0, 0);
      const im = ctx.getImageData(0,0,out.width,out.height);
      const d = im.data;
      // prob png はグレースケール(0..255)想定
      for (let i=0; i<d.length; i+=4) {
        const p = d[i]; // R
        const v = (p >= thr) ? 255 : 0;
        d[i] = v; d[i+1] = v; d[i+2] = v; d[i+3] = 255;
      }
      ctx.putImageData(im, 0, 0);
    };
    img.src = probUrl;
  }, [probUrl, thr]);

  return (
    <div style={{padding: 16, fontFamily: "sans-serif"}}>
      <h2>Local Line Extraction (Scribble-guided)</h2>

      <div style={{display:"flex", gap:16, alignItems:"center", marginBottom:12}}>
        <input type="file" accept="image/*" onChange={(e)=> {
          const f = e.target.files?.[0]; if (f) onUpload(f);
        }} />
        <button onClick={clearScribble} disabled={!imgUrl}>Clear</button>
        <button onClick={predict} disabled={!imgUrl}>Predict</button>
        <div style={{display:"flex", alignItems:"center", gap:8}}>
          <span>Threshold</span>
          <input type="range" min={0} max={255} value={thr}
                 onChange={(e)=>setThr(parseInt(e.target.value,10))}/>
          <span>{thr}</span>
        </div>
      </div>

      {imgUrl && (
        <div style={{position:"relative", display:"inline-block"}}>
          {/* 元画像 */}
          <img ref={imgRef} src={imgUrl} alt="" style={{maxWidth: 900, height:"auto", display:"block"}} />

          {/* スクリブル */}
          <canvas
            ref={scribbleRef}
            style={{position:"absolute", left:0, top:0, width:"100%", height:"100%"}}
            onPointerDown={(e)=>{ setDrawing(true); last.current = null; const p=getPos(e); drawLine(p.x,p.y); }}
            onPointerMove={(e)=>{ if(!drawing) return; const p=getPos(e); drawLine(p.x,p.y); }}
            onPointerUp={()=>{ setDrawing(false); last.current=null; }}
            onPointerLeave={()=>{ setDrawing(false); last.current=null; }}
          />

          {/* 出力（二値化結果） */}
          <canvas
            ref={outRef}
            style={{
              position:"absolute", left:0, top:0, width:"100%", height:"100%",
              mixBlendMode: "multiply", opacity: 0.8, pointerEvents:"none"
            }}
          />
        </div>
      )}
    </div>
  );
}
