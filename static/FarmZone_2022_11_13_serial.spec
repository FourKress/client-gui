# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['FarmZone_2022_11_13_serial.py','FlowField_2022_11_13.py','Obtain_wind_probability_2022_11_13.py','Optimizer_2022_11_13.py', 'WindTurbine_2022_11_13.py'],
    pathex=['C:\\Users\wudong\\WebstormProjects\\client-demo\\static'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FarmZone_2022_11_13_serial',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FarmZone_2022_11_13_serial',
)
