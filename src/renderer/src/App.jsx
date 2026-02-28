import React, { useEffect, useRef, useState } from "react";

export default function App() {
  // UI上の状態
  const [imgFile, setImgFile] = useState(null);
  const [method, setMethod] = useState("frangi");
  const [useClahe, setUseClahe] = useState(false);
  const [claheClip, setClaheClip] = useState(2.0);
  const [claheGrid, setClaheGrid] = useState(8);
  const [maxSize, setMaxSize] = useState(2000);
  const [lineWidth, setLineWidth] = useState(30);
  const [penType, setPenType] = useState("scribble");
  const [scribbleColor, setScribbleColor] = useState("rgb(0,255,0)");
  const [lr, setLr] = useState(1e-3);
  const [iters, setIters] = useState(400);
  const [thr, setThr] = useState(128);
  const [frangiPercentile, setFrangiPercentile] = useState(99);

  // 画像のURL
  const [imgUrl, setImgUrl] = useState("");
  const [probUrl, setProbUrl] = useState("");
  const [frangiOutUrl, setFrangiOutUrl] = useState("");
  const [unetOutUrl, setUnetOutUrl] = useState("");

  // DOM要素への参照
  const imgRef = useRef(null);
  const scribbleRef = useRef(null);
  const outRef = useRef(null);

  // 描画状態
  const [drawing, setDrawing] = useState(false);
  const last = useRef(null); // {x:number, y:number}
  const [cursorPos, setCursorPos] = useState(null); // {x:number, y:number} | null
  // Undo用履歴管理
  const [history, setHistory] = useState([]);
  const refineScribbleFn = useRef(null);

  // 描画状態保存
  const saveState = () => {
    const scr = scribbleRef.current;
    // スクリブルがないときは何もしない
    if (!scr) return;
    const ctx = scr.getContext("2d");
    const data = ctx.getImageData(0, 0, scr.width, scr.height);
    setHistory((prev) => {
      const newHistory = [...prev, data];
      if (newHistory.length > 30) newHistory.shift(); // 履歴は直近30回まで
      return newHistory;
    });
  };

  // 画像アップロード時の処理
  const onUpload = (file) => {
    setImgFile(file);
    const url = URL.createObjectURL(file);
    setImgUrl(url);
    setProbUrl("");
    setFrangiOutUrl("");
    setUnetOutUrl("");
  };

  // スクリブル描画
  const drawLine = (x, y) => {
    const scr = scribbleRef.current;
    if (!scr) return;
    const ctx = scr.getContext("2d");

    // スケール補正
    const rect = scr.getBoundingClientRect();
    const displayScale = rect.width / scr.width;

    // 描画設定
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.globalCompositeOperation = penType === "scribble" ? "source-over" : "destination-out";
    ctx.strokeStyle = penType === "scribble" ? scribbleColor : "rgba(255,255,255,1)";
    ctx.lineWidth = lineWidth / displayScale;
    ctx.beginPath();

    const lastPoint = last.current;
    if (lastPoint) ctx.moveTo(lastPoint.x, lastPoint.y);
    else ctx.moveTo(x, y);

    ctx.lineTo(x, y);
    ctx.stroke();

    last.current = { x, y };
  };

  // カーソル位置取得
  const getPos = (e) => {
    const scr = scribbleRef.current;
    if (!scr) return { x: 0, y: 0 };

    // スケール補正
    const rect = scr.getBoundingClientRect();
    const sx = scr.width / rect.width;
    const sy = scr.height / rect.height;

    return {
      x: (e.clientX - rect.left) * sx,
      y: (e.clientY - rect.top) * sy,
    };
  };

  // スクリブル消去
  const clearScribble = () => {
    // 履歴を保存
    saveState();
    const scr = scribbleRef.current;
    if (scr) scr.getContext("2d").clearRect(0, 0, scr.width, scr.height);
    const out = outRef.current;
    if (out) out.getContext("2d").clearRect(0, 0, out.width, out.height);
    setUnetOutUrl(null);
  };

  // Frangi応答を計算+パーセンタイルを適用
  const computeFrangi = async () => {
    if (!imgFile) {
      alert("画像がありません．");
      return;
    }

    const formData = new FormData();
    formData.append("image", imgFile);
    formData.append("use_clahe", useClahe);
    formData.append("clahe_clip", claheClip);
    formData.append("clahe_grid", claheGrid);
    formData.append("max_size", maxSize);
    const res = await fetch("http://127.0.0.1:8000/api/compute_frangi", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const data = await res.json();
      alert(data.error);
      return;
    }

    // 初回のpercentile適用
    await _applyPercentile(frangiPercentile);
  };

  // Frangi応答のパーセンタイルを適用
  const _applyPercentile = async (p) => {
    const formData = new FormData();
    formData.append("percentile", p);
    const res = await fetch("http://127.0.0.1:8000/api/apply_frangi_percentile", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const data = await res.json();
      alert(data.error);
      return;
    }
    const url = URL.createObjectURL(await res.blob());
    setFrangiOutUrl(url);
  };

  // Frangi応答のパーセンタイルを適用（スライダー用）
  const applyPercentile = (p) => {
    if (!frangiOutUrl) {
      alert("Frangi応答がありません．");
      return;
    }
    _applyPercentile(p);
  };


  // スクリブルの線画化（自動のみ）
  const refineScribble = async () => {
    const scr = scribbleRef.current;
    if (!scr) {
      alert("スクリブルがありません．");
      return;
    }
    if (!imgFile) {
      alert("画像がありません．");
      return;
    }

    const blob = await new Promise((resolve) => scr.toBlob((b) => resolve(b), "image/png"));
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
      const data = await res.json();
      alert(data.error);
      return;
    }

    const refinedBlob = await res.blob();
    const url = URL.createObjectURL(refinedBlob);
    setUnetOutUrl(url);
  };
  // 最新の関数を保持
  refineScribbleFn.current = refineScribble;

  // 全体の線画を予測
  const predict = async () => {
    const scr = scribbleRef.current;
    if (!imgFile) {
      alert("画像がありません．");
      return;
    }
    if (!scr) {
      alert("スクリブルがありません．");
      return;
    }

    const blob_scr = await new Promise((resolve) =>
      scr.toBlob((b) => resolve(b), "image/png")
    );

    if (!blob_scr) {
      alert("スクリブル画像の取得に失敗しました");
      return;
    }

    const formData = new FormData();
    formData.append("lr", lr);
    formData.append("iters", iters);
    formData.append("max_size", maxSize);
    const res = await fetch("http://127.0.0.1:8000/api/predict_line", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const data = await res.json();
      alert(data.error);
      return;
    }

    const outBlob = await res.blob();
    const url = URL.createObjectURL(outBlob);
    setUnetOutUrl(url);
  };



  // 画面更新時，キーイベントリスナーを設定
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
          const scr = scribbleRef.current;
          if (scr) {
            scr.getContext("2d").putImageData(lastState, 0, 0);
            if (refineScribbleFn.current !== null) {
              refineScribbleFn.current();
            }
          }
          return newHistory;
        });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // 画像読み込み時の処理
  useEffect(() => {
    const img = imgRef.current;
    const scr = scribbleRef.current;
    const out = outRef.current;
    // 画像がなければ何もしない
    if (!img || !scr || !out) return;

    const onLoad = () => {
      scr.width = img.naturalWidth;
      scr.height = img.naturalHeight;
      out.width = img.naturalWidth;
      out.height = img.naturalHeight;

      // スクリブル初期化
      const ctx = scr.getContext("2d");
      ctx.clearRect(0, 0, scr.width, scr.height);

      // 出力初期化
      const octx = out.getContext("2d");
      octx.clearRect(0, 0, out.width, out.height);

      // 履歴リセット
      setHistory([]);
    };

    // 読み込まれたときにonLoadを実行
    img.addEventListener("load", onLoad);
    return () => img.removeEventListener("load", onLoad);
  }, [imgUrl]);


  // probをout canvasに表示
  useEffect(() => {
    const out = outRef.current;
    if (!out) return;

    // 切替先に出力が無い場合は、前の結果が残らないようクリアする
    if (!probUrl) {
      const ctx = out.getContext("2d");
      ctx.clearRect(0, 0, out.width, out.height);
      return;
    }

    // probUrl が変わったときprobを表示
    const img = new Image();
    img.onload = () => {
      out.width = img.naturalWidth;
      out.height = img.naturalHeight;
      const ctx = out.getContext("2d");
      ctx.clearRect(0, 0, out.width, out.height);
      ctx.drawImage(img, 0, 0);
    };
    img.src = probUrl;

  }, [probUrl]);


  const saveAlphaPng = async () => {
    const src = outRef.current;
    if (!src || !src.width || !src.height) {
      alert("保存する画像がありません");
      return;
    }

    // 画像データを取得
    const w = src.width;
    const h = src.height;
    const sctx = src.getContext("2d");
    const srcData = sctx.getImageData(0, 0, w, h);
    const d = srcData.data;

    // アルファ付きキャンバスを作成
    const dstCanvas = document.createElement("canvas");
    dstCanvas.width = w;
    dstCanvas.height = h;
    const dctx = dstCanvas.getContext("2d");
    const out = dctx.createImageData(w, h);
    const o = out.data;

    // probをalphaに変換
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i];
      const g = d[i + 1];
      const b = d[i + 2];
      // Rec.709 luma approximation in 0..255
      const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const a = Math.max(0, Math.min(255, Math.round(255 - y)));

      o[i] = r;
      o[i + 1] = g;
      o[i + 2] = b;
      o[i + 3] = a;
    }
    dctx.putImageData(out, 0, 0);

    const blob = await new Promise((resolve) => dstCanvas.toBlob(resolve, "image/png"));
    if (!blob) {
      alert("画像の保存に失敗しました．");
      return;
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "line_alpha.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // frangiモードのときはcursorPosをnullにする
  useEffect(() => {
    if (method === "frangi") setCursorPos(null);
  }, [method]);

  // methodが変わったらprobUrlを更新
  useEffect(() => {
    setProbUrl(method === "frangi" ? frangiOutUrl : unetOutUrl);
  }, [method, frangiOutUrl, unetOutUrl]);

  // frangiモードのときは描画を無効にする
  useEffect(() => {
    if (method === "frangi") {
      setDrawing(false);
      last.current = null;
    }
  }, [method]);






  return (
    <div
      style={{
        padding: 16,
        fontFamily: "sans-serif",
        height: "100%",
        overflowY: "auto",
        overflowX: "hidden",
        boxSizing: "border-box",
      }}
    >
      <h2>Scribble-Supervised Line Extractor - 線画抽出器</h2>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => {
          const f = e.target.files && e.target.files[0];
          if (f) onUpload(f);
        }}
      />

      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
        <span>方式:</span>
        <button onClick={() => setMethod("frangi")} style={{ opacity: method === "frangi" ? 1 : 0.6 }}>
          Frangi（高速）
        </button>
        <button onClick={() => setMethod("refine")} style={{ opacity: method === "refine" ? 1 : 0.6 }}>
          U-Net（低速）
        </button>
        <span style={{ marginLeft: 20 }}></span>
        {method === "refine" ? (
          <>
            <button onClick={clearScribble} disabled={!imgUrl}>スクリブル消去</button>

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
              <span>ペンの太さ</span>
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
              <span>最大解像度</span>
              <input
                type="number"
                min={256}
                step={256}
                value={maxSize}
                onChange={(e) => setMaxSize(e.target.value)}
                style={{ width: 90 }}
              />
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span>ツール:</span>
              <label>
                <input type="radio" value="scribble" checked={penType === "scribble"} onChange={(e) => setPenType(e.target.value)} /> ペン
              </label>
              <label>
                <input type="radio" value="erase" checked={penType === "erase"} onChange={(e) => setPenType(e.target.value)} /> 消しゴム
              </label>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 8, opacity: penType === "erase" ? 0.5 : 1, pointerEvents: penType === "erase" ? "none" : "auto" }}>
              <span>線画/背景:</span>
              <label>
                <input type="radio" value="rgb(0,255,0)" checked={scribbleColor === "rgb(0,255,0)"} onChange={(e) => setScribbleColor(e.target.value)} /> 線画
              </label>
              <label>
                <input type="radio" value="rgb(255,0,0)" checked={scribbleColor === "rgb(255,0,0)"} onChange={(e) => setScribbleColor(e.target.value)} /> 背景
              </label>
            </div>
          </>
        ) : (
          <>
          </>
        )}
      </div>




      <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 12 }}>
        {method === "frangi" ? (
          <>
            <button onClick={computeFrangi} disabled={!imgUrl}>線画抽出</button>
            <span style={{ opacity: frangiOutUrl ? 1 : 0.4 }}>パーセンタイル</span>
            <input
              type="range"
              min={80}
              max={100}
              step={0.1}
              value={frangiPercentile}
              disabled={!frangiOutUrl}
              onChange={(e) => {
                setFrangiPercentile(e.target.value);
                applyPercentile(e.target.value);
              }}
              style={{ width: 120 }}
            />
            <input
              type="number"
              min={80}
              max={100}
              step={0.1}
              value={frangiPercentile}
              disabled={!frangiOutUrl}
              onChange={(e) => {
                setFrangiPercentile(e.target.value);
                applyPercentile(e.target.value);
              }}
              style={{ width: 70 }}
            />
          </>
        ) : (
          <>
            <button onClick={predict} disabled={!imgUrl}>全体の線画を生成</button>
            <button onClick={refineScribble} disabled={!imgUrl}>生成をキャンセル</button>
            <span>学習率</span>
            <input
              type="number"
              min={1e-7}
              step={1e-7}
              value={lr}
              onChange={(e) => setLr(e.target.value)}
              style={{ width: 90 }}
            />
            <span>学習ステップ数</span>
            <input
              type="number"
              min={10}
              step={100}
              value={iters}
              onChange={(e) => setIters(e.target.value)}
              style={{ width: 90 }}
            />
          </>
        )}

        <button onClick={saveAlphaPng}>線画を保存</button>
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
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                width: "100%",
                height: "100%",
                opacity: method === "frangi" ? 0 : 0.7,
                cursor: method === "frangi" ? "auto" : "none",
                pointerEvents: method === "frangi" ? "none" : "auto",
              }}
              onPointerDown={(e) => {
                if (method === "frangi") return;
                saveState(); // 描画開始前に履歴を保存
                setDrawing(true);
                last.current = null;
                const p = getPos(e);
                drawLine(p.x, p.y);
              }}
              onPointerMove={(e) => {
                if (method === "frangi") return;
                const p = getPos(e);

                // カーソルサイズは画面上のピクセル（lineWidth）にそのまま合わせる
                if (method !== "frangi") setCursorPos({ x: e.clientX, y: e.clientY, size: lineWidth });
                if (!drawing) return;
                drawLine(p.x, p.y);
              }}
              onPointerUp={() => {
                if (method === "frangi") return;
                setDrawing(false);
                refineScribble(); // 描画完了ごとに線画を更新
                last.current = null;
              }}
              onPointerLeave={() => {
                if (method === "frangi") return;
                setDrawing(false);
                last.current = null;
                setCursorPos(null);
              }}
            />

            {/* カスタムカーソル */}
            {method !== "frangi" && cursorPos && (
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
                mixBlendMode: "multiply", // 白背景に対して乗算
                pointerEvents: "none",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}