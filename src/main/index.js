import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'

import { spawn } from 'child_process'

let serverProcess = null

function getPythonPath() {
  if (is.dev) {
    // 開発時: src/python/server.py を直接 python で実行
    return null  // 後述
  }
  // 本番: asar 展開済みの exe
  return join(process.resourcesPath, 'server.exe')
}

function createWindow() {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
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

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  const pythonPath = getPythonPath()
  if (pythonPath) {
    // 本番: exe を直接実行 (cwd を resourcesPath に設定して temp/ を正しく作成)
    serverProcess = spawn(pythonPath, [], { cwd: process.resourcesPath })
    serverProcess.stdout.on('data', (data) => console.log('[Python]', data.toString()))
    serverProcess.stderr.on('data', (data) => console.error('[Python ERROR]', data.toString()))
    serverProcess.on('close', (code) => console.log('[Python] exited with code', code))
  } else {
    // 開発: python コマンドで src/python/server.py を実行
    const serverDir = join(__dirname, '../../src/python')
    serverProcess = spawn('python', [join(serverDir, 'server.py')], { cwd: serverDir })
    serverProcess.stdout.on('data', (data) => console.log('[Python]', data.toString()))
    serverProcess.stderr.on('data', (data) => console.error('[Python ERROR]', data.toString()))
    serverProcess.on('close', (code) => console.log('[Python] exited with code', code))
  }

  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // IPC test
  ipcMain.on('ping', () => console.log('pong'))

  createWindow()

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
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
