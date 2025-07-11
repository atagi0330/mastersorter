// main.js
const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

let mainWindow;
let pythonProcess = null; // Pythonプロセスを管理するための変数

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1000,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'), // プリロードスクリプトを使用
            contextIsolation: true, // コンテキスト分離を有効に
            nodeIntegration: false // Node.jsをレンダラープロセスで無効に
        }
    });

    mainWindow.loadFile('indexUI.html');

    // 開発ツールを開く (デバッグ用)
    // mainWindow.webContents.openDevTools();
}

// Electronアプリが準備完了したらウィンドウを作成
app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

// 全てのウィンドウが閉じられたときにアプリを終了
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
    // アプリケーション終了時にPythonプロセスも終了させる
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
    }
});

// Pythonスクリプトを実行するIPCハンドラー
ipcMain.handle('start-python-script', async (event, scriptName) => {
    // 既存のPythonプロセスがあれば終了させる
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
        console.log('既存のPythonプロセスを終了しました。');
    }

    const scriptPath = path.join(__dirname, scriptName); // scriptName が新しいファイル名になる // スクリプトのフルパス
    const pythonExecutable = 'C:\\Users\\ok230075\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'; // または 'python3' など、環境に合わせて変更

    console.log(`Pythonスクリプトを起動中: ${scriptPath}`);

    return new Promise((resolve, reject) => {
        // Pythonスクリプトを子プロセスとして起動
        pythonProcess = spawn(pythonExecutable, [scriptPath]);

        let stdout = '';
        let stderr = '';

        // 標準出力からのデータを受信
        pythonProcess.stdout.on('data', (data) => {
            const output = data.toString();
            stdout += output;
            console.log(`Python stdout: ${output}`);
            // レンダラープロセスにPythonの出力を送信
            mainWindow.webContents.send('python-output', output);
        });

        // 標準エラー出力からのデータを受信
        pythonProcess.stderr.on('data', (data) => {
            const error = data.toString();
            stderr += error;
            console.error(`Python stderr: ${error}`);
            // レンダラープロセスにPythonのエラーを送信
            mainWindow.webContents.send('python-error', error);
        });

        // プロセスが終了したとき
        pythonProcess.on('close', (code) => {
            console.log(`Pythonプロセスが終了しました。終了コード: ${code}`);
            if (code === 0) {
                resolve({ success: true, stdout, stderr });
            } else {
                reject({ success: false, stdout, stderr, code });
            }
            pythonProcess = null; // プロセス終了後に参照をクリア
        });

        // プロセスがエラーを発生したとき（例: pythonが見つからない）
        pythonProcess.on('error', (err) => {
            console.error('Pythonプロセス起動エラー:', err);
            reject({ success: false, error: err.message });
            pythonProcess = null;
        });
    });
});

// Pythonプロセスを停止するIPCハンドラー
ipcMain.handle('stop-python-script', () => {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
        console.log('Pythonプロセスを手動で終了しました。');
        return { message: 'Pythonプロセスを停止しました。' };
    }
    return { message: '実行中のPythonプロセスはありません。' };
});