import fs from 'fs';
import path from 'path';
import childProcess from 'child_process';
import { decode } from 'iconv-lite';
import { app, BrowserWindow, ipcMain, dialog } from 'electron';

/**
 * Set `__static` path to static files in production
 * https://simulatedgreg.gitbooks.io/electron-vue/content/en/using-static-assets.html
 */
if (process.env.NODE_ENV !== 'development') {
  global.__static = require('path')
    .join(__dirname, '/static')
    .replace(/\\/g, '\\\\'); // eslint-disable-line
}

let mainWindow;
const winURL =
  process.env.NODE_ENV === 'development'
    ? 'http://localhost:9080'
    : `file://${__dirname}/index.html`;

ipcMain.on('open-directory-dialog', (event, args) => {
  dialog
    .showOpenDialog({
      properties: [args],
      title: '请选择文件路径',
      buttonLabel: '选择',
    })
    .then((result) => {
      event.sender.send('selectFilePath', result.filePaths[0]);
    });
});

ipcMain.on('getResult', (event, args) => {
  fs.readFile(`${args[0]}/demo_fig.jpg`, (err, data) => {
    if (err) return;
    const b64 = Buffer.from(data).toString('base64');
    event.sender.send('result', `data:image/jpg;base64,${b64}`);
  });
});

ipcMain.on('start', (event, args) => {
  const [index, params] = args;
  console.log(`参数: ${JSON.stringify(params)}`);
  console.log('————————开始PY进程————————');

  // let pyPath = `${path.join(__static, './FarmZone_2022_11_13_serial.py')}`;
  // if (process.env.NODE_ENV !== 'development') {
  //   pyPath = path
  //     .join(__static, '/FarmZone_2022_11_13_serial.py')
  //     .replace('\\app.asar\\dist\\electron', '');
  // }
  // const workerProcess = childProcess.spawn('python', [
  //   `${pyPath}`,
  //   `${JSON.stringify({
  //     ...params,
  //   })}`,
  // ]);

  let pyPath = `${path.join(
    __static,
    './dist/FarmZone_2022_11_13_serial/FarmZone_2022_11_13_serial.exe',
  )}`;
  if (process.env.NODE_ENV !== 'development') {
    pyPath = path
      .join(
        __static,
        '/dist/FarmZone_2022_11_13_serial/FarmZone_2022_11_13_serial.exe',
      )
      .replace('\\app.asar\\dist\\electron', '');
  }
  const workerProcess = childProcess.spawn(`${pyPath}`, [
    JSON.stringify(params),
  ]);

  const encoding = 'cp936';
  const binaryEncoding = 'binary';

  workerProcess.stdout.on('data', (data) => {
    const result = decode(Buffer.from(data, binaryEncoding), encoding);
    console.log(`stdout: ${result.toString()}`);
    event.sender.send(`stdout_${index}`, result.toString());
  });
  workerProcess.stderr.on('data', (data) => {
    const result = decode(Buffer.from(data, binaryEncoding), encoding);
    console.log(`stderr: ${result}`);
    event.sender.send(`stderr_${index}`, result.toString());
  });
  workerProcess.on('close', (code) => {
    console.log(`计算结束，退出码 ${code}`);
    event.sender.send(`close_${index}`, `计算结束，退出码 ${code}`);
  });
});

function createWindow() {
  /**
   * Initial window options
   */
  mainWindow = new BrowserWindow({
    height: 768,
    useContentSize: true,
    width: 1600,
    webPreferences: {
      nodeIntegration: true,
    },
    show: false,
    icon: 'src/logo.ico',
  });

  mainWindow.loadURL(winURL);
  global.mainWindow = mainWindow;

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  mainWindow.on('ready-to-show', () => {
    mainWindow.maximize();
    mainWindow.show();
  });
}

const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    // 当运行第二个实例时,将会聚焦到myWindow这个窗口
    if (mainWindow) {
      mainWindow.show();
      if (mainWindow.isMinimized()) {
        mainWindow.restore();
      }
      mainWindow.focus();
    }
  });

  // 创建 myWindow, 加载应用的其余部分, etc...
  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit();
    }
  });

  app.on('before-quit', () => {
    if (global.server) {
      global.server.close();
    }
  });

  app.on('activate', () => {
    if (mainWindow === null) {
      createWindow();
    }
  });

  app.on('ready', async () => {
    createWindow();
  });
}
