import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'

import { spawn } from 'child_process'
import { createWriteStream, mkdirSync, existsSync, rmSync } from 'fs'

let serverProcess = null
let setupWindow = null

// ユーザーデータ内のvenvパス（本番用）
function getVenvDir() {
  return join(app.getPath('userData'), 'venv')
}

// Python実行ファイルのパス
function getVenvPython() {
  if (is.dev) {
    // 開発時: プロジェクトルートの .venv を使う、なければ system python
    const devVenv = join(__dirname, '../../.venv')
    const devPython = process.platform === 'win32'
      ? join(devVenv, 'Scripts', 'python.exe')
      : join(devVenv, 'bin', 'python')
    return existsSync(devPython) ? devPython : (process.platform === 'win32' ? 'python' : 'python3')
  }
  const venvDir = getVenvDir()
  return process.platform === 'win32'
    ? join(venvDir, 'Scripts', 'python.exe')
    : join(venvDir, 'bin', 'python')
}

// Pythonソースファイルのディレクトリ
function getPythonSrcDir() {
  if (is.dev) return join(__dirname, '../../src/python')
  return join(process.resourcesPath, 'python')
}

// requirements.txtのパス
function getRequirementsPath() {
  if (is.dev) return join(__dirname, '../../requirements.txt')
  return join(process.resourcesPath, 'python', 'requirements.txt')
}

function createServerLogStream() {
  const logDir = app.getPath('userData')
  mkdirSync(logDir, { recursive: true })
  return createWriteStream(join(logDir, 'server.log'), { flags: 'a' })
}

function sendSetupLog(msg) {
  if (setupWindow && !setupWindow.isDestroyed()) {
    setupWindow.webContents.send('setup-log', msg)
  }
  console.log('[Setup]', msg)
}

// venv作成 → pip install を順に実行
function runSetup() {
  return new Promise((resolve, reject) => {
    const venvDir = getVenvDir()
    const pip = process.platform === 'win32'
      ? join(venvDir, 'Scripts', 'pip.exe')
      : join(venvDir, 'bin', 'pip')

    const systemPython = process.platform === 'win32' ? 'python' : 'python3'
    sendSetupLog('Python仮想環境を作成中...')
    const venvCreate = spawn(systemPython, ['-m', 'venv', venvDir])
    venvCreate.on('error', err => reject(new Error(`Pythonが見つかりません (${systemPython}): ${err.message}`)))
    venvCreate.stdout.on('data', d => sendSetupLog(d.toString().trim()))
    venvCreate.stderr.on('data', d => sendSetupLog(d.toString().trim()))
    venvCreate.on('close', code => {
      if (code !== 0) return reject(new Error(`venvの作成に失敗しました： (exit ${code})`))

      sendSetupLog('依存パッケージをインストール中...')
      const pipProc = spawn(pip, [
        'install', '-r', getRequirementsPath(),
        '--extra-index-url', 'https://download.pytorch.org/whl/cu126'
      ])
      pipProc.on('error', err => reject(new Error(`pipの実行に失敗しました (${pip}): ${err.message}`)))
      pipProc.stdout.on('data', d => sendSetupLog(d.toString().trim()))
      pipProc.stderr.on('data', d => sendSetupLog(d.toString().trim()))
      pipProc.on('close', code => {
        if (code !== 0) return reject(new Error(`パッケージインストールに失敗しました： (exit ${code})`))
        resolve()
      })
    })
  })
}

function startPythonServer() {
  return new Promise((resolve, reject) => {
    const python = getVenvPython()
    const srcDir = getPythonSrcDir()
    const logStream = createServerLogStream()
    const ts = () => new Date().toISOString()
    let resolved = false

    const onReady = () => {
      if (resolved) return
      resolved = true
      resolve()
    }

    serverProcess = spawn(python, [join(srcDir, 'server.py')], { cwd: srcDir })

    serverProcess.on('error', err => {
      logStream.write(`[${ts()}][fatal] ${err.message}\n`)
      console.error('[Python ERROR]', err.message)
      if (!resolved) reject(new Error(`Pythonサーバーの起動に失敗しました (${python}): ${err.message}`))
    })

    serverProcess.stdout.on('data', d => {
      const msg = d.toString()
      logStream.write(`[${ts()}][out] ${msg}`)
      console.log('[Python]', msg)
      if (msg.includes('Running on http://127.0.0.1')) onReady()
    })
    serverProcess.stderr.on('data', d => {
      const msg = d.toString()
      logStream.write(`[${ts()}][err] ${msg}`)
      console.error('[Python ERROR]', msg)
      if (msg.includes('Running on http://127.0.0.1')) onReady()
    })
    serverProcess.on('close', code => {
      logStream.write(`[${ts()}][exit] ${code}\n`)
      console.log('[Python] exited with code', code)
      if (!resolved) reject(new Error(`Pythonサーバーが起動前に終了しました： (exit ${code})`))
    })

    // 30秒以内に起動しなければタイムアウトで開く
    setTimeout(onReady, 30000)
  })
}

function createSetupWindow() {
  setupWindow = new BrowserWindow({
    width: 620,
    height: 420,
    resizable: false,
    autoHideMenuBar: true,
    title: 'セットアップ中...',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  })
  const htmlPath = is.dev
    ? join(__dirname, '../../resources/setup.html')
    : join(process.resourcesPath, 'setup.html')
  setupWindow.loadFile(htmlPath)
}

async function doSetup() {
  try {
    await runSetup()
    sendSetupLog('\nセットアップが完了しました。サーバーを起動しています...')
    await startPythonServer()
    sendSetupLog('サーバーが起動しました。アプリを開いています...')
    createWindow()
    setTimeout(() => {
      if (setupWindow && !setupWindow.isDestroyed()) setupWindow.close()
      setupWindow = null
    }, 1500)
  } catch (err) {
    sendSetupLog(`\nエラー: ${err.message}`)
    if (setupWindow && !setupWindow.isDestroyed()) {
      setupWindow.webContents.send('setup-error', err.message)
    }
  }
}

// 再試行: 部分的なvenvを削除して再実行
ipcMain.handle('retry-setup', async () => {
  try { rmSync(getVenvDir(), { recursive: true, force: true }) } catch (_) {}
  await doSetup()
})

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1600,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(async () => {
  electronApp.setAppUserModelId('com.kouki.ssle')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  ipcMain.on('ping', () => console.log('pong'))

  if (is.dev) {
    // 開発時: セットアップ不要、そのままサーバーを起動
    await startPythonServer()
    createWindow()
  } else {
    const venvPython = getVenvPython()
    if (existsSync(venvPython)) {
      // セットアップ済み: そのまま起動
      await startPythonServer()
      createWindow()
    } else {
      // 初回: セットアップウィンドウを表示してからvenv構築
      createSetupWindow()
      setupWindow.webContents.once('did-finish-load', () => doSetup())
    }
  }

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (serverProcess) serverProcess.kill()  // サーバープロセスを終了
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
