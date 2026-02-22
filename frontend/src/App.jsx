// frontend/src/App.jsx
import React, { useEffect, useRef, useState } from "react";

export default function App() {
  const [imgFile, setImgFile] = useState(null);
  const [imgUrl, setImgUrl] = useState("");
  const [thr, setThr] = useState(128);
  const [lineWidth, setLineWidth] = useState(30);
  const [probUrl, setProbUrl] = useState("");
  const [maxSize, setMaxSize] = useState(5000);
  const [scribbleColor, setScribbleColor] = useState("rgb(0,255,0)");
  const [penType, setPenType] = useState("scribble");
  const [useClahe, setUseClahe] = useState(false);
  const [claheClip, setClaheClip] = useState(2.0);
  const [claheGrid, setClaheGrid] = useState(8);
  const [lr, setLr] = useState(1e-3);
  const [iters, setIters] = useState(700);
  const [device, setDevice] = useState("cuda")


  const imgRef = useRef(null);
  const scribbleRef = useRef(null);
  const outRef = useRef(null);

  const [drawing, setDrawing] = useState(false);
  const last = useRef(null); // {x:number, y:number}
  const [cursorPos, setCursorPos] = useState(null); // {x:number, y:number} | null

  // --- Undo (Ctrl+Z) 履歴管理 ---
  const [history, setHistory] = useState([]);

  const saveState = () => {
    const sc = scribbleRef.current;
    if (!sc) return;
    const ctx = sc.getContext("2d");
    const data = ctx.getImageData(0, 0, sc.width, sc.height);
    setHistory((prev) => {
      const newHistory = [...prev, data];
      if (newHistory.length > 30) newHistory.shift(); // 履歴は直近30回まで
      return newHistory;
    });
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      // ユーザーが Input や Textarea にフォーカスしている場合はネイティブの Undo に任せる
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        setHistory((prev) => {
          if (prev.length === 0) return prev; // 何も保存されていなければ何もしない
          const newHistory = [...prev];
          const lastState = newHistory.pop();
          const sc = scribbleRef.current;
          if (sc) {
            sc.getContext("2d").putImageData(lastState, 0, 0);
          }
          return newHistory;
        });
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);


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
      const ctx = sc.getContext("2d");
      ctx.clearRect(0, 0, sc.width, sc.height);

      // 出力初期化
      const octx = out.getContext("2d");
      octx.clearRect(0, 0, out.width, out.height);

      setHistory([]); // 画像読み込み時は履歴リセット
    };

    img.addEventListener("load", onLoad);
    return () => img.removeEventListener("load", onLoad);
  }, [imgUrl]);

  const onUpload = (f) => {
    setImgFile(f);
    const u = URL.createObjectURL(f);
    setImgUrl(u);
    setProbUrl("");
  };

  const drawLine = (x, y) => {
    const sc = scribbleRef.current;
    if (!sc) return;
    const ctx = sc.getContext("2d");

    // キャンバスの表示スケールを計算して線の太さを補正
    const rect = sc.getBoundingClientRect();
    const displayScale = rect.width / sc.width;

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.globalCompositeOperation = penType === "scribble" ? "source-over" : "destination-out";
    ctx.strokeStyle = penType === "scribble" ? scribbleColor : "rgba(255,255,255,1)";
    ctx.lineWidth = lineWidth / displayScale; // 画面上のピクセルサイズに合わせる
    ctx.beginPath();

    const p = last.current;
    if (p) ctx.moveTo(p.x, p.y);
    else ctx.moveTo(x, y);

    ctx.lineTo(x, y);
    ctx.stroke();

    last.current = { x, y };
  };

  const getPos = (e) => {
    const sc = scribbleRef.current;
    if (!sc) return { x: 0, y: 0 };

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
    saveState(); // 消去前にも履歴を保存
    const sc = scribbleRef.current;
    const out = outRef.current;
    if (sc) sc.getContext("2d").clearRect(0, 0, sc.width, sc.height);
    if (out) out.getContext("2d").clearRect(0, 0, out.width, out.height);
    setProbUrl("");
  };

  const refineScribble = async () => {
    const sc = scribbleRef.current;
    if (!sc || !imgFile) return;
    const blob = await new Promise((resolve) =>
      sc.toBlob((b) => resolve(b), "image/png")
    );
    if (!blob) {
      alert("スクリブル画像の取得に失敗しました");
      return;
    }

    const formData = new FormData();
    formData.append("image", imgFile);
    formData.append("scribble", new File([blob], "scribble.png", { type: "image/png" }));
    formData.append("use_clahe", useClahe);
    formData.append("clahe_clip", claheClip);
    formData.append("clahe_grid", claheGrid);
    formData.append("max_size", maxSize);

    const res = await fetch("http://127.0.0.1:8000/api/refine_scribble", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      alert("スクリブル画像の送信に失敗しました");
      return;
    }

    const outBlob = await res.blob();
    const url = URL.createObjectURL(outBlob);
    setProbUrl(url);
  };



  const predict = async () => {
    const sc = scribbleRef.current;
    const line = outRef.current;
    if (!imgFile || !sc || !line) return;

    const blob_scr = await new Promise((resolve) =>
      sc.toBlob((b) => resolve(b), "image/png")
    );
    const blob_line = await new Promise((resolve) =>
      line.toBlob((b) => resolve(b), "image/png")
    );
    if (!blob_scr || !blob_line) {
      alert("Failed to capture scribble");
      return;
    }

    const formData = new FormData();
    formData.append("image", imgFile);
    formData.append("scribble", new File([blob_scr], "scribble.png", { type: "image/png" }));
    formData.append("refined_scribble", new File([blob_line], "refined_scribble.png", { type: "image/png" }));
    formData.append("lr", lr);
    formData.append("iters", iters);
    formData.append("device", device)
    formData.append("max_size", maxSize);

    // CRAのproxyを使うなら "/api/predict" のように相対パス推奨
    const res = await fetch("http://127.0.0.1:8000/api/predict_line", {
      method: "POST",
      body: formData,
    });

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
    const out = outRef.current;
    if (!probUrl || !out) return;

    const img = new Image();
    img.onload = () => {
      const ctx = out.getContext("2d");
      ctx.clearRect(0, 0, out.width, out.height);

      // probを読み込み
      ctx.drawImage(img, 0, 0);

      const im = ctx.getImageData(0, 0, out.width, out.height);
      const d = im.data;

      // prob png はグレースケール(0..255)想定
      for (let i = 0; i < d.length; i += 4) {
        const p = d[i]; // R
        const v = p >= thr ? 255 : 0;
        d[i] = v;
        d[i + 1] = v;
        d[i + 2] = v;
        d[i + 3] = 255;
      }

      ctx.putImageData(im, 0, 0);
    };
    img.src = probUrl;
  }, [probUrl, thr]);






  return (
    <div style={{ padding: 16, fontFamily: "sans-serif" }}>
      <h2>Scribble-Supervised Line Extractor</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 12 }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            const f = e.target.files && e.target.files[0];
            if (f) onUpload(f);
          }}
        />
        <button onClick={clearScribble} disabled={!imgUrl}>Clear</button>

        {/* <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>Threshold</span>
          <input
            type="range"
            min={0}
            max={255}
            value={thr}
            onChange={(e) => setThr(parseInt(e.target.value, 10))}
          />
          <span>{thr}</span>
        </div> */}

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>lineWidth</span>
          <input
            type="range"
            min={1}
            max={300}
            value={lineWidth}
            onChange={(e) => setLineWidth(parseInt(e.target.value, 10))}
          />
          <span>{lineWidth}</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>maxSize</span>
          <input
            type="number"
            min={256}
            step={256}
            value={maxSize}
            onChange={(e) => setMaxSize(parseInt(e.target.value || "0", 10) || 0)}
            style={{ width: 90 }}
          />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>Tool:</span>
          <label>
            <input type="radio" value="scribble" checked={penType === "scribble"} onChange={(e) => setPenType(e.target.value)} /> Pen
          </label>
          <label>
            <input type="radio" value="erase" checked={penType === "erase"} onChange={(e) => setPenType(e.target.value)} /> Eraser
          </label>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8, opacity: penType === "erase" ? 0.5 : 1, pointerEvents: penType === "erase" ? "none" : "auto" }}>
          <span>Color:</span>
          <label>
            <input type="radio" value="rgb(0,255,0)" checked={scribbleColor === "rgb(0,255,0)"} onChange={(e) => setScribbleColor(e.target.value)} /> Line
          </label>
          <label>
            <input type="radio" value="rgb(255,0,0)" checked={scribbleColor === "rgb(255,0,0)"} onChange={(e) => setScribbleColor(e.target.value)} /> Background
          </label>
        </div>
      </div>

      <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 12 }}>
        <button onClick={refineScribble} disabled={!imgUrl}>Refine Scribble</button>
        <button onClick={predict} disabled={!imgUrl}>Predict Line</button>
        <span>Iterations</span>
          <input
            type="range"
            min={1}
            max={2000}
            value={iters}
            onChange={(e) => setIters(parseInt(e.target.value, 10))}
          />
        <span>{iters}</span>
      </div>

      {imgUrl && (
        <div style={{ display: "flex", gap: 16 }}>
          {/* 左側: 入力画像とスクリブル */}
          <div style={{ position: "relative", display: "inline-block" }}>
            <img
              ref={imgRef}
              src={imgUrl}
              alt=""
              style={{ maxWidth: "45vw", height: "auto", display: "block" }}
            />

            {/* スクリブル */}
            <canvas
              ref={scribbleRef}
              style={{ position: "absolute", left: 0, top: 0, width: "100%", height: "100%", opacity: 0.7, cursor: "none" }}
              onPointerDown={(e) => {
                saveState(); // 描画開始前に履歴を保存
                setDrawing(true);
                last.current = null;
                const p = getPos(e);
                drawLine(p.x, p.y);
              }}
              onPointerMove={(e) => {
                const p = getPos(e);

                // カーソルサイズは画面上のピクセル（lineWidth）にそのまま合わせる
                setCursorPos({ x: e.clientX, y: e.clientY, size: lineWidth });
                if (!drawing) return;
                drawLine(p.x, p.y);
              }}
              onPointerUp={() => {
                setDrawing(false);
                last.current = null;
              }}
              onPointerLeave={() => {
                setDrawing(false);
                last.current = null;
                setCursorPos(null);
              }}
            />

            {/* カスタムカーソル */}
            {cursorPos && (
              <div
                style={{
                  position: "fixed",
                  left: cursorPos.x,
                  top: cursorPos.y,
                  width: cursorPos.size,
                  height: cursorPos.size,
                  borderRadius: "50%",
                  border: "1px solid black",
                  backgroundColor: penType === "erase" ? "rgba(0,0,0,0.2)" : scribbleColor,
                  opacity: 0.3,
                  transform: "translate(-50%, -50%)",
                  pointerEvents: "none",
                  zIndex: 9999,
                }}
              />
            )}
          </div>

          {/* 右側: 出力結果 */}
          <div style={{ position: "relative", display: "inline-block", backgroundColor: "white" }}>
            <img
              src={imgUrl}
              alt="Output background"
              style={{ maxWidth: "45vw", height: "auto", display: "block", visibility: "hidden" }}
            />
            {/* 出力（二値化結果） */}
            <canvas
              ref={outRef}
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                width: "100%",
                height: "100%",
                mixBlendMode: "multiply", // 白背景に対しての乗算なので線画が黒く乗る
                pointerEvents: "none",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}