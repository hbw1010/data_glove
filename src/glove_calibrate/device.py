"""发现本机手套；仅显式准备请求启动板端已安装的采集程序。"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import os
import re
import subprocess
import time

from .paths import ROOT
from .usb.usb_transport import discover_usb_devices

_SCRIPT = "/usr/sbin/dataglove-usb-composite"


def _adb(*arguments: str, timeout: float = 8) -> str:
    # 使用Pixi锁定wheel内的二进制，不借用旧工程或自动下载工具。
    executable = Path(str(files("adbutils").joinpath("binaries", "adb")))
    if not executable.is_file():
        raise RuntimeError("当前adbutils分发包不含Linux ADB，请重新运行pixi install --locked")
    home = ROOT / ".cache" / "adb"
    home.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "ANDROID_USER_HOME": str(home)}
    try:
        result = subprocess.run([str(executable), *arguments], capture_output=True, text=True,
                                timeout=timeout, env=environment, check=False)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("ADB设备查询超时；请检查USB连接") from error
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip() or "ADB请求失败")
    return result.stdout


def _debug_serials() -> set[str]:
    """只纳入实际RV1106手套，不把任意Android设备当成待启动手套。"""
    serials = set()
    for device in Path("/sys/bus/usb/devices").iterdir():
        try:
            if ((device / "idVendor").read_text().strip() == "2207"
                    and (device / "manufacturer").read_text().strip() == "RV1106 Hand Development"):
                serials.add((device / "serial").read_text().strip())
        except (FileNotFoundError, PermissionError, NotADirectoryError):
            continue
    return serials


def discover_devices() -> dict:
    devices, warnings = {}, []
    try:
        for device in discover_usb_devices(vid=0x2207):
            devices[device.serial] = {"serial": device.serial, "mode": "dgst"}
    except (RuntimeError, OSError) as error:
        warnings.append(str(error))
    serials = _debug_serials() - devices.keys()
    if serials:
        try:
            for line in _adb("devices", "-l").splitlines():
                columns = line.split()
                if len(columns) >= 2 and columns[0] in serials:
                    if columns[1] == "device":
                        devices[columns[0]] = {"serial": columns[0], "mode": "debug"}
                    else:
                        warnings.append(f"手套 {columns[0]} ADB状态为 {columns[1]}")
        except (RuntimeError, OSError, TimeoutError) as error:
            warnings.append(str(error))
    return {"devices": list(devices.values()), "warnings": warnings}


def prepare_device(serial: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{8,64}", serial):
        raise ValueError("请选择有效的手套USB序列号")
    devices = discover_devices()
    selected = next((item for item in devices["devices"] if item["serial"] == serial), None)
    if selected is None:
        raise ValueError("所选手套未连接或不可访问：" + "；".join(devices["warnings"]))
    if selected["mode"] == "dgst":
        return {"serial": serial, "ready": True, "message": "DGST接口已经就绪，未重复启动"}
    status = _adb("-s", serial, "shell", _SCRIPT, "status")
    if "dataglove-usbd running pid=" in status:
        raise RuntimeError("板端采集程序已运行但DGST接口未就绪；不会自动停止或覆盖现有服务。" + status)
    if "dataglove-usbd not running" not in status:
        raise RuntimeError("板端缺少兼容的采集启动程序；不会自动部署或刷写固件。" + status)
    # 脚本自带脱离ADB的监督启动；重枚举期间ADB短暂中断是预期现象。
    _adb("-s", serial, "shell", _SCRIPT, "start", timeout=12)
    deadline = time.monotonic() + 35
    last_error = "未发现DGST接口"
    while time.monotonic() < deadline:
        try:
            if any(item.serial == serial for item in discover_usb_devices(vid=0x2207)):
                return {"serial": serial, "ready": True, "message": "板端DGST接口就绪，可以连接实时手套"}
        except (RuntimeError, OSError) as error:
            last_error = str(error)
        time.sleep(0.5)
    try:
        detail = _adb("-s", serial, "shell", "cat", "/tmp/dgst/composite.log")[-4000:]
    except (RuntimeError, OSError, TimeoutError) as error:
        detail = str(error)
    raise RuntimeError(f"板端采集未就绪：{last_error}；{detail}")
