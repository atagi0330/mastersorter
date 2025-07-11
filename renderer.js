// renderer.js
document.addEventListener('DOMContentLoaded', () => {
    const calibrateButton = document.getElementById('calibrateButton');
    const detectButton = document.getElementById('detectButton');
    const stopButton = document.getElementById('stopButton');
    const outputConsole = document.getElementById('outputConsole');

    // コンソールにメッセージを追加するヘルパー関数
    function appendToConsole(message, isError = false) {
        const p = document.createElement('p');
        p.textContent = message;
        if (isError) {
            p.style.color = '#f00'; // エラーは赤色
        }
        outputConsole.appendChild(p);
        outputConsole.scrollTop = outputConsole.scrollHeight; // 自動スクロール
    }

    // Pythonスクリプトからの出力をリッスン
    window.electronAPI.onPythonOutput((data) => {
        appendToConsole(data);
    });

    // Pythonスクリプトからのエラーをリッスン
    window.electronAPI.onPythonError((data) => {
        appendToConsole(data, true);
    });

    // キャリブレーションボタンのクリックハンドラー
    calibrateButton.addEventListener('click', async () => {
        appendToConsole('キャリブレーションスクリプトを起動中...');
        try {
            const result = await window.electronAPI.startPythonScript("config_saver.py");
            if (result.success) {
                appendToConsole('キャリブレーションスクリプトが正常に終了しました。');
            } else {
                appendToConsole(`キャリブレーションスクリプトがエラーで終了しました: ${result.code}`, true);
                if (result.stderr) appendToConsole(result.stderr, true);
            }
        } catch (error) {
            appendToConsole(`スクリプト起動エラー: ${error.message}`, true);
        }
    });

    // 検出ボタンのクリックハンドラー
    detectButton.addEventListener('click', async () => {
        appendToConsole('検出スクリプトを起動中...');
        try {
            const result = await window.electronAPI.startPythonScript("detector.py");
            if (result.success) {
                appendToConsole('検出スクリプトが正常に終了しました。');
            } else {
                appendToConsole(`検出スクリプトがエラーで終了しました: ${result.code}`, true);
                if (result.stderr) appendToConsole(result.stderr, true);
            }
        } catch (error) {
            appendToConsole(`スクリプト起動エラー: ${error.message}`, true);
        }
    });

    // 停止ボタンのクリックハンドラー
    stopButton.addEventListener('click', async () => {
        appendToConsole('Pythonプロセスを停止しようとしています...');
        try {
            const result = await window.electronAPI.stopPythonScript();
            appendToConsole(result.message);
        } catch (error) {
            appendToConsole(`プロセス停止エラー: ${error.message}`, true);
        }
    });
});