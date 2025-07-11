// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    startPythonScript: (scriptName) => ipcRenderer.invoke('start-python-script', scriptName),
    stopPythonScript: () => ipcRenderer.invoke('stop-python-script'),
    onPythonOutput: (callback) => ipcRenderer.on('python-output', (_event, value) => callback(value)),
    onPythonError: (callback) => ipcRenderer.on('python-error', (_event, value) => callback(value))
});