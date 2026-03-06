import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

export default function App() {
  // UI上の状態
  const [imgFile, setImgFile] = useState(null)
  const [method, setMethod] = useState('unet')
  const [useClahe] = useState(false)
  const [claheClip] = useState(2.0)
  const [claheGrid] = useState(8)
  const [maxSize, setMaxSize] = useState(2000)
  const [lineWidth, setLineWidth] = useState(30)
  const [penType, setPenType] = useState('scribble')
  const [scribbleColor, setScribbleColor] = useState('rgb(0,255,0)')
  const [lr, setLr] = useState(1e-4)
  const [iters, setIters] = useState(400)
  const [frangiPercentile, setFrangiPercentile] = useState(95)

  // 画像のURL
  const [imgUrl, setImgUrl] = useState('')
  const [frangiOutUrl, setFrangiOutUrl] = useState('')
  const [unetOutUrl, setUnetOutUrl] = useState('')
  const probUrl = method === 'frangi' ? frangiOutUrl : unetOutUrl

  // DOM要素への参照
  const imgRef = useRef(null)
  const scribbleRef = useRef(null)
  const outRef = useRef(null)
  const historyScrollRef = useRef(null)

  // 描画状態
  const [drawing, setDrawing] = useState(false)
  const last = useRef(null) // {x:number, y:number}
  const [cursorPos, setCursorPos] = useState(null) // {x:number, y:number} | null

  // Undo, Redo用履歴管理
  const historyRef = useRef([])
  const redoHistoryRef = useRef([])
  const [histLength, setHistLength] = useState(0) // 履歴の長さ（UI更新用）
  const [redoHistLength, setRedoHistLength] = useState(0) // Redo履歴の長さ（UI更新用）

  // プログレスバー用
  const [progress, setProgress] = useState({ it: 0, iters: 0, loss: 0 })
  const progressIntervalRef = useRef(null)

  const previewRef = useRef(null)
  const refineReqSeqRef = useRef(0)

  // ウィンドウサイズ
  const [windowSize, setWindowSize] = useState(() => ({
    width: window.innerWidth,
    height: window.innerHeight
  }))

  // 画像の表示サイズ
  const [imgSize, setImgSize] = useState({ width: 0, height: 0 })

  // 画像の拡大縮小移動の状態
  const [canvasTransform, setCanvasTransform] = useState({
    clicked: false,
    scale: 1,
    offset: { x: 0, y: 0 }
  })
  // 生成履歴
  const [unetOutImages, setUnetOutImages] = useState([])
  const [frangiOutImages, setFrangiOutImages] = useState([])
  const outImages = method === 'frangi' ? frangiOutImages : unetOutImages

  // ここから描画関連の関数
  // 描画状態保存
  const saveState = () => {
    const scr = scribbleRef.current
    // スクリブルがないときは何もしない
    if (!scr) return
    const ctx = scr.getContext('2d')
    const data = ctx.getImageData(0, 0, scr.width, scr.height)
    historyRef.current.push(data)
    redoHistoryRef.current = [] // 新たに更新したらRedo履歴をクリア
    setHistLength(historyRef.current.length)
    setRedoHistLength(redoHistoryRef.current.length)
    if (historyRef.current.length > 200) historyRef.current.shift() // 履歴は直近200回まで
  }

  // 画像アップロード時の処理（URL関連の初期化）
  const onUpload = (file) => {
    setImgFile(file)
    const url = URL.createObjectURL(file)
    setImgUrl(url)
    setFrangiOutUrl('')
    setUnetOutUrl('')
  }

  // スクリブル描画
  const drawLine = (x, y) => {
    const scr = scribbleRef.current
    if (!scr) return
    const ctx = scr.getContext('2d')

    const rect = scr.getBoundingClientRect()
    const displayScale = rect.width / scr.width

    // 描画設定
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.globalCompositeOperation = penType === 'scribble' ? 'source-over' : 'destination-out'
    ctx.strokeStyle = penType === 'scribble' ? scribbleColor : 'rgba(255,255,255,1)'
    ctx.lineWidth = lineWidth / displayScale
    ctx.beginPath()

    const lastPoint = last.current
    if (lastPoint) ctx.moveTo(lastPoint.x, lastPoint.y)
    else ctx.moveTo(x, y)

    ctx.lineTo(x, y)
    ctx.stroke()

    last.current = { x, y }
  }

  // カーソル位置取得
  const getPos = (e) => {
    const scr = scribbleRef.current
    if (!scr) return { x: 0, y: 0 }

    const rect = scr.getBoundingClientRect()
    const scale = rect.width / scr.width

    return {
      x: (e.clientX - rect.left) / scale,
      y: (e.clientY - rect.top) / scale
    }
  }

  // スクリブル消去
  const clearScribble = () => {
    // 履歴を保存
    saveState()
    const scr = scribbleRef.current
    if (scr) scr.getContext('2d').clearRect(0, 0, scr.width, scr.height)
    const out = outRef.current
    if (out) out.getContext('2d').clearRect(0, 0, out.width, out.height)
    setUnetOutUrl(null)
  }

  // Frangi応答を計算+パーセンタイルを適用
  const computeFrangi = async () => {
    if (!imgFile) {
      alert('画像がありません．')
      return
    }

    const formData = new FormData()
    formData.append('image', imgFile)
    formData.append('use_clahe', useClahe)
    formData.append('clahe_clip', claheClip)
    formData.append('clahe_grid', claheGrid)
    formData.append('max_size', maxSize)
    const res = await fetch('http://127.0.0.1:8000/api/compute_frangi', {
      method: 'POST',
      body: formData
    })
    if (!res.ok) {
      const data = await res.json()
      alert(data.error)
      return
    }

    // 初回のpercentile適用（履歴に追加）
    await _applyPercentile(frangiPercentile, true)
  }

  // Frangi応答のパーセンタイルを適用
  const _applyPercentile = async (p, addToHistory = false) => {
    const formData = new FormData()
    formData.append('percentile', p)
    const res = await fetch('http://127.0.0.1:8000/api/apply_frangi_percentile', {
      method: 'POST',
      body: formData
    })
    if (!res.ok) {
      const data = await res.json()
      alert(data.error)
      return
    }
    const url = URL.createObjectURL(await res.blob())
    setFrangiOutUrl(url)
    if (addToHistory) {
      setFrangiOutImages((prev) => [...prev, url])
    }
  }

  // Frangi応答のパーセンタイルを適用（スライダー用）
  const applyPercentile = (p, addToHistory = false) => {
    if (!frangiOutUrl) {
      alert('Frangi応答がありません．')
      return
    }
    _applyPercentile(p, addToHistory)
  }

  // スクリブルの線画化（自動のみ）
  const refineScribble = useCallback(async () => {
    const reqSeq = ++refineReqSeqRef.current
    const scr = scribbleRef.current
    if (!scr) {
      alert('スクリブルがありません．')
      return
    }
    if (!imgFile) {
      alert('画像がありません．')
      return
    }

    const blob = await new Promise((resolve) => scr.toBlob((b) => resolve(b), 'image/png'))
    if (!blob) {
      alert('スクリブル画像の取得に失敗しました')
      return
    }

    const formData = new FormData()
    formData.append('image', imgFile)
    formData.append('scribble', new File([blob], 'scribble.png', { type: 'image/png' }))
    formData.append('use_clahe', useClahe)
    formData.append('clahe_clip', claheClip)
    formData.append('clahe_grid', claheGrid)
    formData.append('max_size', maxSize)
    const res = await fetch('http://127.0.0.1:8000/api/refine_scribble', {
      method: 'POST',
      body: formData
    })
    // レスポンスが返ってきたときに、リクエストが最新かどうかを確認
    if (reqSeq !== refineReqSeqRef.current) return

    if (!res.ok) {
      const data = await res.json()
      alert(data.error)
      return
    }

    const refinedBlob = await res.blob()
    if (reqSeq !== refineReqSeqRef.current) return
    const url = URL.createObjectURL(refinedBlob)
    setUnetOutUrl(url)
  }, [imgFile, useClahe, claheClip, claheGrid, maxSize])

  // 全体の線画を予測
  const predict = async () => {
    const scr = scribbleRef.current
    if (!imgFile) {
      alert('画像がありません．')
      return
    }
    if (!scr) {
      alert('スクリブルがありません．')
      return
    }
    // refineScribbleを待ってから予測を開始
    await refineScribble()

    // プログレスバー更新
    progressIntervalRef.current = setInterval(async () => {
      const res = await fetch('http://127.0.0.1:8000/api/progress')
      if (!res.ok) {
        const data = await res.json()
        alert(data.error)
        clearInterval(progressIntervalRef.current)
        return
      }
      const data = await res.json()
      setProgress(data)
    }, 300)

    previewRef.current = setInterval(async () => {
      const res = await fetch('http://127.0.0.1:8000/api/preview')
      if (!res.ok) {
        const data = await res.json()
        alert(data.error)
        clearInterval(previewRef.current)
        return
      }
      const url = URL.createObjectURL(await res.blob())
      setUnetOutUrl(url)
    }, 1000)

    // 全体線画を予測
    const formData = new FormData()
    formData.append('lr', lr)
    formData.append('iters', iters)
    formData.append('max_size', maxSize)
    const res = await fetch('http://127.0.0.1:8000/api/predict_line', {
      method: 'POST',
      body: formData
    })

    if (!res.ok) {
      const data = await res.json()
      alert(data.error)
      clearInterval(progressIntervalRef.current)
      clearInterval(previewRef.current)
      return
    }

    // 完了時にインターバルを止める
    clearInterval(progressIntervalRef.current)
    clearInterval(previewRef.current)

    const outBlob = await res.blob()
    const url = URL.createObjectURL(outBlob)
    setUnetOutUrl(url)
    // 生成履歴に追加
    setUnetOutImages((prev) => [...prev, url])
  }

  // 生成キャンセル
  const cancelPrediction = async () => {
    try {
      await fetch('http://127.0.0.1:8000/api/cancel_prediction', { method: 'POST' })
    } catch (error) {
      alert('キャンセルリクエストの送信に失敗しました．: ' + error.message)
    }
    // インターバルを止める
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current)
    }
    if (previewRef.current) {
      clearInterval(previewRef.current)
    }
  }

  // Undo処理
  const undo = useCallback(() => {
    if (historyRef.current.length === 0) return
    const lastState = historyRef.current.pop()
    const scr = scribbleRef.current
    if (scr) {
      const ctx = scr.getContext('2d')
      redoHistoryRef.current.push(ctx.getImageData(0, 0, scr.width, scr.height))
      ctx.putImageData(lastState, 0, 0)
      // 履歴の長さを更新
      setHistLength(historyRef.current.length)
      setRedoHistLength(redoHistoryRef.current.length)
      refineScribble()
    }
  }, [refineScribble])

  // Redo処理
  const redo = useCallback(() => {
    if (redoHistoryRef.current.length === 0) return
    const nextState = redoHistoryRef.current.pop()
    const scr = scribbleRef.current
    if (scr) {
      const ctx = scr.getContext('2d')
      historyRef.current.push(ctx.getImageData(0, 0, scr.width, scr.height))
      ctx.putImageData(nextState, 0, 0)
      setHistLength(historyRef.current.length)
      setRedoHistLength(redoHistoryRef.current.length)
      refineScribble()
    }
  }, [refineScribble])

  const saveAlphaPng = async () => {
    const src = outRef.current
    if (!src || !src.width || !src.height) {
      alert('保存する画像がありません')
      return
    }

    // 画像データを取得
    const w = src.width
    const h = src.height
    const sctx = src.getContext('2d')
    const srcData = sctx.getImageData(0, 0, w, h)
    const d = srcData.data

    // アルファ付きキャンバスを作成
    const dstCanvas = document.createElement('canvas')
    dstCanvas.width = w
    dstCanvas.height = h
    const dctx = dstCanvas.getContext('2d')
    const out = dctx.createImageData(w, h)
    const o = out.data

    // probをalphaに変換
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i]
      const g = d[i + 1]
      const b = d[i + 2]
      // Rec.709 luma approximation in 0..255
      const y = 0.2126 * r + 0.7152 * g + 0.0722 * b
      const a = Math.max(0, Math.min(255, Math.round(255 - y)))

      o[i] = r
      o[i + 1] = g
      o[i + 2] = b
      o[i + 3] = a
    }
    dctx.putImageData(out, 0, 0)

    const blob = await new Promise((resolve) => dstCanvas.toBlob(resolve, 'image/png'))
    if (!blob) {
      alert('画像の保存に失敗しました．')
      return
    }

    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'line_alpha.png'
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const changeMethod = (nextMethod) => {
    setMethod(nextMethod)
    if (nextMethod === 'frangi') {
      setCursorPos(null)
      setDrawing(false)
      last.current = null
    }
  }

  // ここからuseEffect
  // 画面更新時，キーイベントリスナーを設定
  useEffect(() => {
    const handleKeyDown = (e) => {
      // ユーザーが Input や Textarea にフォーカスしている場合はネイティブの Undo に任せる
      // if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

      if ((e.ctrlKey || e.metaKey) && e.code === 'KeyY') {
        e.preventDefault()
        redo()
      } else if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.code === 'KeyZ') {
        e.preventDefault()
        redo()
      } else if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.code === 'KeyZ') {
        e.preventDefault()
        undo()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [redo, undo])

  // 画像読み込み時の処理
  useEffect(() => {
    const img = imgRef.current
    const scr = scribbleRef.current
    const out = outRef.current
    // 画像がなければ何もしない
    if (!img || !scr || !out) return

    const onLoad = () => {
      // 内部サイズを画像の元サイズに合わせる
      scr.width = img.naturalWidth
      scr.height = img.naturalHeight
      out.width = img.naturalWidth
      out.height = img.naturalHeight

      // スクリブル初期化
      const ctx = scr.getContext('2d')
      ctx.clearRect(0, 0, scr.width, scr.height)

      // 出力初期化
      const octx = out.getContext('2d')
      octx.clearRect(0, 0, out.width, out.height)

      // キャンバスサイズ
      const canvasW = (windowSize.width - 156 - 78) / 2
      const canvasH = windowSize.height - 240
      // 画像の元サイズ
      const imgW = img.naturalWidth
      const imgH = img.naturalHeight
      // 縮小スケールを計算
      const scale = Math.min(canvasW / imgW, canvasH / imgH)
      setImgSize({
        width: imgW * scale,
        height: imgH * scale
      })

      // 中央においたときの余白
      const offset = {
        x: (canvasW - imgW * scale) / 2,
        y: (canvasH - imgH * scale) / 2
      }
      setCanvasTransform({ clicked: false, scale: 1, offset: offset })

      // 履歴リセット
      historyRef.current = []
      redoHistoryRef.current = []
    }

    // 読み込まれたときにonLoadを実行
    img.addEventListener('load', onLoad)
    return () => img.removeEventListener('load', onLoad)
  }, [windowSize.height, windowSize.width, imgUrl])

  // probをout canvasに表示
  useEffect(() => {
    const out = outRef.current
    if (!out) return

    // 切替先に出力が無い場合は、前の結果が残らないようクリアする
    if (!probUrl) {
      const ctx = out.getContext('2d')
      ctx.clearRect(0, 0, out.width, out.height)
      return
    }

    // probUrl が変わったときprobを表示
    const img = new Image()
    img.onload = () => {
      out.width = img.naturalWidth
      out.height = img.naturalHeight
      const ctx = out.getContext('2d')
      ctx.clearRect(0, 0, out.width, out.height)
      ctx.drawImage(img, 0, 0)
    }
    img.src = probUrl
  }, [probUrl])

  // 画面サイズ変更時の処理
  useEffect(() => {
    const calcViewSize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight
      })
    }
    window.addEventListener('resize', calcViewSize)
    return () => window.removeEventListener('resize', calcViewSize)
  })

  useEffect(() => {
    if (!historyScrollRef.current) return
    historyScrollRef.current.scrollLeft = historyScrollRef.current.scrollWidth
  }, [outImages, method])

  // UI描画
  return (
    <div className="app-container">
      <h2 className="app-title">Scribble-Supervised Line Extractor &mdash; 線画抽出AI</h2>
      {/* 生成進捗*/}
      <div
        className="progress-panel"
        style={{
          opacity: method === 'frangi' ? 0.5 : 1,
          pointerEvents: method === 'frangi' ? 'none' : 'auto'
        }}
      >
        <span className="progress-label">生成進捗</span>
        <progress
          value={progress.it}
          max={progress.iters}
          style={{ width: windowSize.width - 1100 }}
        />
        <span className="progress-stat">
          {progress.it} / {progress.iters} (損失: {progress.loss.toFixed(4)})
        </span>
        <button onClick={cancelPrediction} disabled={!imgUrl}>
          生成をキャンセル
        </button>
      </div>
      <div
        ref={historyScrollRef}
        className="history-panel"
        onWheel={(e) => {
          e.preventDefault()
          e.currentTarget.scrollLeft += e.deltaY
        }}
        style={{
          width: (windowSize.width - 156 - 108) / 2,
          height: 102
        }}
      >
        <span className="history-label" style={{opacity: outImages.length > 0 ? 0 : 1}}>
          生成履歴
        </span>
        {outImages.map((url, index) => (
          <img
            key={index}
            src={url}
            className="history-thumb"
            style={{
              border:
                url === probUrl ? '2px solid var(--color-accent)' : '1px solid var(--color-border)',
              opacity: url === probUrl ? 1 : 0.7
            }}
            onClick={() => {
              method === 'frangi' ? setFrangiOutUrl(url) : setUnetOutUrl(url)
            }}
          />
        ))}
      </div>

      {/* ファイル選択 */}
      <div className="controls-row" style={{ marginTop: 6, marginBottom: 6 }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            const f = e.target.files && e.target.files[0]
            if (f) onUpload(f)
          }}
        />

        <span className="controls-label" style={{opacity: imgUrl ? 1 : 0.5}}>
          最大解像度
        </span>
        <input
          type="number"
          min={256}
          step={256}
          value={maxSize}
          onChange={(e) => setMaxSize(e.target.value)}
          onBlur={() => refineScribble()}
          disabled={!imgUrl}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.target.blur() // Enterを押したらフォーカスを外す
            }
          }}
          style={{ width: 90 }}
        />
        <button onClick={saveAlphaPng} disabled={!probUrl}>
          線画を保存
        </button>
      </div>

      <div className="controls-row" style={{ marginBottom: 8 }}>
        <span className="controls-label">方式:</span>
        <button
          className={`method-btn${method === 'unet' ? ' active' : ''}`}
          onClick={() => changeMethod('unet')}
        >
          U-Net（AI）
        </button>
        <button
          className={`method-btn${method === 'frangi' ? ' active' : ''}`}
          onClick={() => changeMethod('frangi')}
        >
          Frangiフィルタ
        </button>
      </div>

      <div className="controls-row" style={{ marginBottom: 8 }}>
        {method === 'frangi' ? (
          <>
            <button className="btn-primary" onClick={computeFrangi} disabled={!imgUrl}>
              線画抽出
            </button>
            <span
              className="controls-label"
              style={{
                opacity: frangiOutUrl ? 1 : 0.5,
                pointerEvents: frangiOutUrl ? 'auto' : 'none'
              }}
            >
              パーセンタイル
            </span>
            <input
              type="range"
              min={80}
              max={100}
              step={0.1}
              value={frangiPercentile}
              disabled={!frangiOutUrl}
              pointerEvents={frangiOutUrl ? 'auto' : 'none'}
              onChange={(e) => {
                setFrangiPercentile(e.target.value)
                applyPercentile(e.target.value)
              }}
              onMouseUp={(e) => applyPercentile(e.currentTarget.value, true)}
              onTouchEnd={(e) => applyPercentile(e.currentTarget.value, true)}
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
                setFrangiPercentile(e.target.value)
                applyPercentile(e.target.value)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.target.blur() // Enterを押したらフォーカスを外す
                }
              }}
              style={{ width: 70 }}
            />
          </>
        ) : (
          <>
            <button className="btn-primary" onClick={predict} disabled={!imgUrl}>
              全体の線画を生成
            </button>
            <span className="controls-label">学習率</span>
            <input
              type="number"
              min={1e-7}
              step="any"
              value={lr}
              onChange={(e) => setLr(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.target.blur() // Enterを押したらフォーカスを外す
                }
              }}
              style={{ width: 90 }}
            />
            <span className="controls-label">学習ステップ数</span>
            <input
              type="number"
              min={100}
              step={100}
              value={iters}
              onChange={(e) => setIters(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.target.blur() // Enterを押したらフォーカスを外す
                }
              }}
              style={{ width: 90 }}
            />
          </>
        )}
      </div>

      <div style={{ display: 'flex', gap: 16 }}>
        {/* 描画ツール */}
        <div
          className="tool-panel"
          style={{
            width: '140px',
            height: '500px',
            opacity: method === 'frangi' ? 0.5 : 1,
            pointerEvents: method === 'frangi' ? 'none' : 'auto'
          }}
        >
          <div className="tool-group">
            <span className="tool-group-label">ペンの太さ</span>
            <input
              type="range"
              min={1}
              max={100}
              value={lineWidth}
              onChange={(e) => setLineWidth(parseInt(e.target.value, 10))}
            />
            <span style={{ fontSize: '0.8em', color: 'var(--color-text-muted)' }}>
              {lineWidth}px
            </span>
          </div>
          <div className="tool-sep" />
          <div className="tool-group">
            <span className="tool-group-label">ツール</span>
            <label>
              <input
                type="radio"
                value="scribble"
                checked={penType === 'scribble'}
                onChange={(e) => setPenType(e.target.value)}
              />{' '}
              ペン
            </label>
            <label>
              <input
                type="radio"
                value="erase"
                checked={penType === 'erase'}
                onChange={(e) => setPenType(e.target.value)}
              />{' '}
              消しゴム
            </label>
          </div>

          <div className="tool-sep" />
          <div
            className="tool-group"
            style={{
              opacity: penType === 'erase' ? 0.5 : 1,
              pointerEvents: penType === 'erase' ? 'none' : 'auto'
            }}
          >
            <span className="tool-group-label">ペン種類</span>
            <label>
              <input
                type="radio"
                value="rgb(0,255,0)"
                checked={scribbleColor === 'rgb(0,255,0)'}
                onChange={(e) => setScribbleColor(e.target.value)}
              />{' '}
              線画
            </label>
            <label>
              <input
                type="radio"
                value="rgb(255,0,0)"
                checked={scribbleColor === 'rgb(255,0,0)'}
                onChange={(e) => setScribbleColor(e.target.value)}
              />{' '}
              背景
            </label>
          </div>
          <div className="tool-sep" />
          {/* <button style={{ width: '140px' }} onClick={refineScribble} disabled={!imgUrl}>スクリブル<br />線画化</button> */}
          <button style={{ width: '140px' }} onClick={clearScribble}>
           すべて消去
          </button>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
            <button
              style={{ width: '62px', fontSize: '0.78em' }}
              onClick={undo}
              disabled={!imgUrl || histLength === 0}
            >
              取り消し
            </button>
            <button
              style={{ width: '62px', fontSize: '0.78em' }}
              onClick={redo}
              disabled={!imgUrl || redoHistLength === 0}
            >
              やり直し
            </button>
          </div>
        </div>

        {/* 左側: 入力画像とスクリブル */}
        <div
          className="viewer-panel"
          style={{
            position: 'relative',
            display: 'inline-block',
            width: (windowSize.width - 156 - 78) / 2,
            height: windowSize.height - 240,
            borderRadius: 12,
            overflow: 'hidden',
            cursor: method === 'frangi' || !imgUrl ? 'auto' : 'none'
          }}
          // スクロール時
          onWheel={(e) => {
            if (!imgUrl) return
            e.preventDefault()
            const rect = e.currentTarget.getBoundingClientRect()
            // マウス位置をキャンバス内の座標に変換
            const cx = e.clientX - rect.left
            const cy = e.clientY - rect.top
            setCanvasTransform((prev) => {
              const scalePrev = prev.scale
              // キャンバス内での画像位置
              const zoomOffsetPrev = prev.offset
              const newScale = Math.max(0.1, Math.min(10, scalePrev * Math.exp(-e.deltaY * 0.0004)))
              // スクロール後のマウスと画像の距離
              const distX = (cx - zoomOffsetPrev.x) * (newScale / scalePrev)
              const distY = (cy - zoomOffsetPrev.y) * (newScale / scalePrev)

              return {
                ...prev,
                scale: newScale,
                offset: {
                  x: cx - distX,
                  y: cy - distY
                }
              }
            })
          }}
          // クリック時
          onPointerDown={(e) => {
            if (!imgUrl) return
            e.preventDefault()
            if (e.button === 1 || (e.button === 0 && method === 'frangi')) {
              e.currentTarget.setPointerCapture(e.pointerId)
              // 中ボタンドラッグでパン開始
              setCanvasTransform((prev) => ({
                ...prev,
                clicked: true,
                offset: {
                  x: prev.offset.x,
                  y: prev.offset.y
                }
              }))
            } else if (e.button === 0 && method === 'unet') {
              e.currentTarget.setPointerCapture(e.pointerId)
              saveState() // 描画開始前に履歴を保存
              setDrawing(true)
              last.current = null
              const p = getPos(e)
              drawLine(p.x, p.y)
            }
          }}
          // ドラッグ時
          onPointerMove={(e) => {
            if (!imgUrl) return
            setCursorPos({ x: e.clientX, y: e.clientY, size: lineWidth })
            if (e.target.hasPointerCapture(e.pointerId)) {
              if (drawing) {
                const p = getPos(e)
                drawLine(p.x, p.y)
              }
              setCanvasTransform((prev) => {
                if (!prev.clicked) return prev
                const dx = e.movementX
                const dy = e.movementY
                return {
                  ...prev,
                  offset: {
                    x: prev.offset.x + dx,
                    y: prev.offset.y + dy
                  }
                }
              })
            }
          }}
          // ドラッグ終了
          onPointerUp={(e) => {
            if (!imgUrl) return
            e.preventDefault()
            if (e.currentTarget.hasPointerCapture(e.pointerId)) {
              e.currentTarget.releasePointerCapture(e.pointerId)
            }
            if (e.button === 1 || (e.button === 0 && method === 'frangi')) {
              // ドラッグ終了
              setCanvasTransform((prev) => ({
                ...prev,
                clicked: false
              }))
            } else if (e.button === 0 && method === 'unet') {
              if (drawing) refineScribble() // 描画完了ごとに線画を更新
              setDrawing(false)
              last.current = null
            }
          }}
          // キャンバス外に出たとき
          onPointerLeave={() => {
            if (!imgUrl) return
            setDrawing(false)
            last.current = null
            setCursorPos(null)
          }}
        >
          {/* 画像 */}
          <img
            ref={imgRef}
            src={imgUrl}
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              width: imgSize.width,
              height: imgSize.height,
              objectFit: 'contain',
              display: imgUrl ? 'block' : 'none',
              transform: `translate(${canvasTransform.offset.x}px, ${canvasTransform.offset.y}px) scale(${canvasTransform.scale})`,
              transformOrigin: '0 0'
            }}
          />

          {/* スクリブル */}
          <canvas
            ref={scribbleRef}
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              width: imgSize.width,
              height: imgSize.height,
              objectFit: 'contain',
              opacity: method === 'frangi' || !imgUrl ? 0 : 0.7,
              transform: `translate(${canvasTransform.offset.x}px, ${canvasTransform.offset.y}px) scale(${canvasTransform.scale})`,
              transformOrigin: '0 0'
            }}
          />

          {/* カスタムカーソル */}
          {method !== 'frangi' && cursorPos && imgUrl && (
            <div
              style={{
                position: 'fixed',
                left: cursorPos.x,
                top: cursorPos.y,
                width: cursorPos.size,
                height: cursorPos.size,
                borderRadius: '50%',
                border: '1px solid black',
                backgroundColor: penType === 'erase' ? 'rgba(0,0,0,0.2)' : scribbleColor,
                opacity: 0.3,
                transform: 'translate(-50%, -50%)',
                pointerEvents: 'none',
                zIndex: 9999
              }}
            />
          )}
        </div>

        {/* 右側: 出力結果 */}
        <div
          className="viewer-panel"
          style={{
            position: 'relative',
            display: 'inline-block',
            flex: 1,
            borderRadius: 12,
            overflow: 'hidden'
          }}
          // スクロール時
          onWheel={(e) => {
            if (!imgUrl) return
            e.preventDefault()
            const rect = e.currentTarget.getBoundingClientRect()
            // マウス位置をキャンバス内の座標に変換
            const cx = e.clientX - rect.left
            const cy = e.clientY - rect.top
            setCanvasTransform((prev) => {
              const scalePrev = prev.scale
              // キャンバス内での画像位置
              const zoomOffsetPrev = prev.offset
              const newScale = Math.max(0.1, Math.min(10, scalePrev * Math.exp(-e.deltaY * 0.0004)))
              // スクロール後のマウスと画像の距離
              const distX = (cx - zoomOffsetPrev.x) * (newScale / scalePrev)
              const distY = (cy - zoomOffsetPrev.y) * (newScale / scalePrev)

              return {
                ...prev,
                scale: newScale,
                offset: {
                  x: cx - distX,
                  y: cy - distY
                }
              }
            })
          }}
          // クリック時
          onPointerDown={(e) => {
            if (!imgUrl) return
            e.preventDefault()
            if (e.button === 1 || e.button === 0) {
              e.currentTarget.setPointerCapture(e.pointerId)
              // 中ボタンドラッグでパン開始
              setCanvasTransform((prev) => ({
                ...prev,
                clicked: true,
                offset: {
                  x: prev.offset.x,
                  y: prev.offset.y
                }
              }))
            }
          }}
          // ドラッグ時
          onPointerMove={(e) => {
            if (!imgUrl) return
            if (e.target.hasPointerCapture(e.pointerId)) {
              setCanvasTransform((prev) => {
                if (!prev.clicked) return prev
                const dx = e.movementX
                const dy = e.movementY
                return {
                  ...prev,
                  offset: {
                    x: prev.offset.x + dx,
                    y: prev.offset.y + dy
                  }
                }
              })
            }
          }}
          // ドラッグ終了
          onPointerUp={(e) => {
            if (!imgUrl) return
            e.preventDefault()
            if (e.currentTarget.hasPointerCapture(e.pointerId)) {
              e.currentTarget.releasePointerCapture(e.pointerId)
            }
            if (e.button === 1 || e.button === 0) {
              // ドラッグ終了
              setCanvasTransform((prev) => ({
                ...prev,
                clicked: false
              }))
            }
          }}
        >
          {/* 出力（二値化結果） */}
          <canvas
            ref={outRef}
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              width: imgSize.width,
              height: imgSize.height,
              objectFit: 'contain',
              transform: `translate(${canvasTransform.offset.x}px, ${canvasTransform.offset.y}px) scale(${canvasTransform.scale})`,
              transformOrigin: '0 0',
              backgroundColor: 'white'
            }}
          />
        </div>
      </div>
    </div>
  )
}
