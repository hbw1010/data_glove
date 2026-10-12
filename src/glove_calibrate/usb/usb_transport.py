"""DGST USB bulk 传输；仅在发现或打开设备时加载系统 libusb-1.0。

Linux 运行库为 libusb-1.0-0。访问设备需要对应 /dev/bus/usb 节点的
udev 权限；本模块不重置 USB、不切换配置、不解绑内核驱动。
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import math
import time

from .dgst_protocol import ProtocolError


_TIMEOUT = -7
_ROLES = {
    "DGST Control": ("control", 1, 1),
    "DGST Video": ("video", 0, 1),
    "DGST Encoder": ("encoder", 0, 1),
}


@dataclass(frozen=True)
class UsbDeviceInfo:
    vid: int
    pid: int
    serial: str
    port_path: str
    bus: int
    address: int
    speed: int


class _DeviceDescriptor(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8),
        ("bcdUSB", ctypes.c_uint16), ("bDeviceClass", ctypes.c_uint8),
        ("bDeviceSubClass", ctypes.c_uint8), ("bDeviceProtocol", ctypes.c_uint8),
        ("bMaxPacketSize0", ctypes.c_uint8), ("idVendor", ctypes.c_uint16),
        ("idProduct", ctypes.c_uint16), ("bcdDevice", ctypes.c_uint16),
        ("iManufacturer", ctypes.c_uint8), ("iProduct", ctypes.c_uint8),
        ("iSerialNumber", ctypes.c_uint8), ("bNumConfigurations", ctypes.c_uint8),
    ]


class _EndpointDescriptor(ctypes.Structure):
    _fields_ = [
        ("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8),
        ("bEndpointAddress", ctypes.c_uint8), ("bmAttributes", ctypes.c_uint8),
        ("wMaxPacketSize", ctypes.c_uint16), ("bInterval", ctypes.c_uint8),
        ("bRefresh", ctypes.c_uint8), ("bSynchAddress", ctypes.c_uint8),
        ("extra", ctypes.POINTER(ctypes.c_ubyte)), ("extra_length", ctypes.c_int),
    ]


class _InterfaceDescriptor(ctypes.Structure):
    _fields_ = [
        ("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8),
        ("bInterfaceNumber", ctypes.c_uint8), ("bAlternateSetting", ctypes.c_uint8),
        ("bNumEndpoints", ctypes.c_uint8), ("bInterfaceClass", ctypes.c_uint8),
        ("bInterfaceSubClass", ctypes.c_uint8), ("bInterfaceProtocol", ctypes.c_uint8),
        ("iInterface", ctypes.c_uint8),
        ("endpoint", ctypes.POINTER(_EndpointDescriptor)),
        ("extra", ctypes.POINTER(ctypes.c_ubyte)), ("extra_length", ctypes.c_int),
    ]


class _Interface(ctypes.Structure):
    _fields_ = [
        ("altsetting", ctypes.POINTER(_InterfaceDescriptor)),
        ("num_altsetting", ctypes.c_int),
    ]


class _ConfigDescriptor(ctypes.Structure):
    _fields_ = [
        ("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8),
        ("wTotalLength", ctypes.c_uint16), ("bNumInterfaces", ctypes.c_uint8),
        ("bConfigurationValue", ctypes.c_uint8), ("iConfiguration", ctypes.c_uint8),
        ("bmAttributes", ctypes.c_uint8), ("MaxPower", ctypes.c_uint8),
        ("interface", ctypes.POINTER(_Interface)),
        ("extra", ctypes.POINTER(ctypes.c_ubyte)), ("extra_length", ctypes.c_int),
    ]


class _LibUsb:
    def __init__(self):
        path = ctypes.util.find_library("usb-1.0")
        if not path:
            raise ProtocolError("USB 需要系统 libusb-1.0 运行库（Linux: libusb-1.0-0）")
        try:
            self.lib = ctypes.CDLL(path)
        except OSError as exc:
            raise ProtocolError(f"无法加载系统 libusb-1.0: {exc}") from exc
        pointer = ctypes.c_void_p
        byte_pointer = ctypes.POINTER(ctypes.c_ubyte)
        signatures = {
            "init": ([ctypes.POINTER(pointer)], ctypes.c_int),
            "exit": ([pointer], None),
            "get_device_list": ([pointer, ctypes.POINTER(ctypes.POINTER(pointer))], ctypes.c_ssize_t),
            "free_device_list": ([ctypes.POINTER(pointer), ctypes.c_int], None),
            "get_device_descriptor": ([pointer, ctypes.POINTER(_DeviceDescriptor)], ctypes.c_int),
            "get_active_config_descriptor": ([pointer, ctypes.POINTER(ctypes.POINTER(_ConfigDescriptor))], ctypes.c_int),
            "free_config_descriptor": ([ctypes.POINTER(_ConfigDescriptor)], None),
            "open": ([pointer, ctypes.POINTER(pointer)], ctypes.c_int),
            "close": ([pointer], None),
            "get_device_speed": ([pointer], ctypes.c_int),
            "get_bus_number": ([pointer], ctypes.c_uint8),
            "get_device_address": ([pointer], ctypes.c_uint8),
            "get_port_numbers": ([pointer, byte_pointer, ctypes.c_int], ctypes.c_int),
            "get_string_descriptor_ascii": ([pointer, ctypes.c_uint8, byte_pointer, ctypes.c_int], ctypes.c_int),
            "claim_interface": ([pointer, ctypes.c_int], ctypes.c_int),
            "release_interface": ([pointer, ctypes.c_int], ctypes.c_int),
            "bulk_transfer": ([pointer, ctypes.c_ubyte, byte_pointer, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_int), ctypes.c_uint], ctypes.c_int),
            "error_name": ([ctypes.c_int], ctypes.c_char_p),
        }
        try:
            for name, (arguments, result) in signatures.items():
                function = getattr(self.lib, f"libusb_{name}")
                function.argtypes = arguments
                function.restype = result
        except AttributeError as exc:
            raise ProtocolError(f"系统 libusb-1.0 缺少必要接口: {exc}") from exc
        self.context = pointer()
        self.check(self.lib.libusb_init(ctypes.byref(self.context)), "libusb_init")

    def check(self, result: int, operation: str) -> int:
        if result < 0:
            name = self.lib.libusb_error_name(result).decode("ascii", "replace")
            hint = "；请配置设备 /dev/bus/usb 的 udev 访问权限" if result == -3 else ""
            if result == -6:
                hint = "；接口正被其他进程或内核驱动占用（不会自动解绑）"
            raise ProtocolError(f"{operation}: {name} ({result}){hint}")
        return result

    def close(self) -> None:
        if self.context:
            self.lib.libusb_exit(self.context)
            self.context = None

    def string(self, handle, index: int) -> str:
        if not index:
            return ""
        buffer = (ctypes.c_ubyte * 256)()
        count = self.check(self.lib.libusb_get_string_descriptor_ascii(
            handle, index, buffer, len(buffer),
        ), "读取 USB 字符串描述符")
        try:
            return ctypes.string_at(buffer, count).decode("utf-8", "strict")
        except UnicodeError as exc:
            raise ProtocolError("USB 字符串描述符不是有效 UTF-8") from exc


def _vendor_interfaces(config):
    for index in range(config.bNumInterfaces):
        interface = config.interface[index]
        for alternate_index in range(interface.num_altsetting):
            alternate = interface.altsetting[alternate_index]
            if (alternate.bInterfaceClass, alternate.bInterfaceSubClass,
                    alternate.bInterfaceProtocol) == (0xFF, 0, 1):
                yield alternate


def _layout(usb, handle, config) -> tuple[list[int], dict[str, int]] | None:
    interfaces: list[int] = []
    endpoints: dict[str, int] = {}
    roles: set[str] = set()
    addresses: set[int] = set()
    for alternate in _vendor_interfaces(config):
        name = usb.string(handle, alternate.iInterface)
        if name not in _ROLES:
            continue
        # 固件使用默认 alternate 0；不擅自切换接口状态。
        if alternate.bAlternateSetting != 0:
            raise ProtocolError(f"{name} 使用不支持的 alternate setting {alternate.bAlternateSetting}")
        role, wanted_out, wanted_in = _ROLES[name]
        number = alternate.bInterfaceNumber
        if role in roles or number in interfaces:
            raise ProtocolError(f"DGST 接口名称或编号重复: {name}, {number}")
        ins, outs = [], []
        for index in range(alternate.bNumEndpoints):
            endpoint = alternate.endpoint[index]
            address = endpoint.bEndpointAddress
            if endpoint.bmAttributes & 3 != 2:
                raise ProtocolError(f"{name} 包含非 bulk 端点")
            if not address & 0x0F or address & 0x70 or address in addresses:
                raise ProtocolError(f"{name} 的端点地址无效或重复: 0x{address:02x}")
            if not endpoint.wMaxPacketSize:
                raise ProtocolError(f"{name} 的端点包大小为零")
            addresses.add(address)
            (ins if address & 0x80 else outs).append(address)
        if (len(outs), len(ins)) != (wanted_out, wanted_in):
            raise ProtocolError(f"{name} 的 bulk 端点方向或数量不匹配")
        roles.add(role)
        interfaces.append(number)
        endpoints[f"{role}_in"] = ins[0]
        if outs:
            endpoints[f"{role}_out"] = outs[0]
    if not roles:
        return None
    if roles != {"control", "video", "encoder"}:
        raise ProtocolError("USB 设备缺少完整的 DGST Control/Video/Encoder 接口")
    return interfaces, endpoints


@dataclass
class _Candidate:
    info: UsbDeviceInfo
    handle: ctypes.c_void_p
    interfaces: list[int]
    endpoints: dict[str, int]


def _scan(usb, *, vid=None, pid=None, serial=None, port_path=None) -> list[_Candidate]:
    for name, value in (("vid", vid), ("pid", pid)):
        if value is not None and (type(value) is not int or not 0 <= value <= 0xFFFF):
            raise ValueError(f"{name} 必须是 0..65535 的整数")
    devices = ctypes.POINTER(ctypes.c_void_p)()
    count = usb.check(usb.lib.libusb_get_device_list(usb.context, ctypes.byref(devices)),
                      "枚举 USB 设备")
    candidates: list[_Candidate] = []
    try:
        for index in range(count):
            device = devices[index]
            descriptor = _DeviceDescriptor()
            usb.check(usb.lib.libusb_get_device_descriptor(device, ctypes.byref(descriptor)),
                      "读取 USB 设备描述符")
            if ((vid is not None and descriptor.idVendor != vid)
                    or (pid is not None and descriptor.idProduct != pid)):
                continue
            bus = usb.lib.libusb_get_bus_number(device)
            ports = (ctypes.c_ubyte * 8)()
            depth = usb.check(usb.lib.libusb_get_port_numbers(device, ports, len(ports)),
                              "读取 USB 物理端口路径")
            physical_path = f"{bus}-" + ".".join(str(ports[i]) for i in range(depth))
            if port_path is not None and port_path != physical_path:
                continue
            config = ctypes.POINTER(_ConfigDescriptor)()
            result = usb.lib.libusb_get_active_config_descriptor(device, ctypes.byref(config))
            if result == -5:  # 未配置设备没有可用接口。
                continue
            usb.check(result, f"读取 USB {physical_path} 的活动配置")
            handle = ctypes.c_void_p()
            retained = False
            try:
                if not any(_vendor_interfaces(config.contents)):
                    continue
                usb.check(usb.lib.libusb_open(device, ctypes.byref(handle)),
                          f"打开 USB {descriptor.idVendor:04x}:{descriptor.idProduct:04x} {physical_path}")
                actual_serial = usb.string(handle, descriptor.iSerialNumber)
                if serial is not None and actual_serial != serial:
                    continue
                layout = _layout(usb, handle, config.contents)
                if layout is None:
                    continue
                interfaces, endpoints = layout
                info = UsbDeviceInfo(
                    descriptor.idVendor, descriptor.idProduct, actual_serial, physical_path,
                    bus, usb.lib.libusb_get_device_address(device),
                    usb.lib.libusb_get_device_speed(device),
                )
                candidates.append(_Candidate(info, handle, interfaces, endpoints))
                retained = True
            finally:
                usb.lib.libusb_free_config_descriptor(config)
                if handle and not retained:
                    usb.lib.libusb_close(handle)
        return candidates
    except BaseException:
        for candidate in candidates:
            usb.lib.libusb_close(candidate.handle)
        raise
    finally:
        usb.lib.libusb_free_device_list(devices, 1)


def discover_usb_devices(*, vid: int | None = None, pid: int | None = None) -> list[UsbDeviceInfo]:
    """读取描述符确认 DGST 接口，不 claim；权限错误不会被当成无设备。"""
    usb = _LibUsb()
    try:
        candidates = _scan(usb, vid=vid, pid=pid)
        try:
            return [candidate.info for candidate in candidates]
        finally:
            for candidate in candidates:
                usb.lib.libusb_close(candidate.handle)
    finally:
        usb.close()


class UsbBulkDevice:
    """独立端点可并发传输；调用 close 前，会话必须停止全部读写线程。"""

    def __init__(self, usb, candidate: _Candidate):
        self._usb = usb
        self._handle = candidate.handle
        self._claimed: list[int] = []
        self.info = candidate.info
        self.endpoints = candidate.endpoints

    @classmethod
    def open(cls, *, vid: int | None = None, pid: int | None = None,
             serial: str | None = None, port_path: str | None = None) -> UsbBulkDevice:
        usb = _LibUsb()
        candidates: list[_Candidate] = []
        opened = None
        try:
            candidates = _scan(usb, vid=vid, pid=pid, serial=serial, port_path=port_path)
            if not candidates:
                raise ProtocolError(
                    f"未找到匹配的 DGST USB 设备（vid={vid!r}, pid={pid!r}, "
                    f"serial={serial!r}, port_path={port_path!r}）；请检查线缆、固件 USB 模式及选择条件"
                )
            if len(candidates) != 1:
                choices = "; ".join(
                    f"{item.info.vid:04x}:{item.info.pid:04x} serial={item.info.serial!r} "
                    f"port_path={item.info.port_path}" for item in candidates
                )
                raise ProtocolError(f"DGST USB 设备选择有歧义: {choices}；请指定唯一 serial 或 port_path")
            selected = candidates.pop()
            opened = cls(usb, selected)
            for interface in selected.interfaces:
                usb.check(usb.lib.libusb_claim_interface(opened._handle, interface),
                          f"claim DGST 接口 {interface} ({opened.info.port_path})")
                opened._claimed.append(interface)
            return opened
        except BaseException:
            if opened is not None:
                opened.close()
            else:
                for candidate in candidates:
                    usb.lib.libusb_close(candidate.handle)
                usb.close()
            raise

    def _check_transfer(self, endpoint: int, timeout_ms: int, *, incoming: bool) -> None:
        if self._handle is None:
            raise ProtocolError("USB 设备已关闭")
        if endpoint not in self.endpoints.values() or bool(endpoint & 0x80) != incoming:
            raise ValueError("端点不属于所选 DGST 设备或方向不匹配")
        if type(timeout_ms) is not int or not 1 <= timeout_ms <= 0xFFFFFFFF:
            raise ValueError("timeout_ms 必须是正整数且不超过 uint32")

    def read(self, endpoint: int, *, size: int = 65536, timeout_ms: int = 250) -> bytes:
        self._check_transfer(endpoint, timeout_ms, incoming=True)
        if type(size) is not int or not 1 <= size <= 0x7FFFFFFF:
            raise ValueError("读取大小必须是正整数且不超过 int32")
        buffer = (ctypes.c_ubyte * size)()
        transferred = ctypes.c_int()
        result = self._usb.lib.libusb_bulk_transfer(
            self._handle, endpoint, buffer, size, ctypes.byref(transferred), timeout_ms,
        )
        if result != _TIMEOUT:
            self._usb.check(result, f"bulk IN 0x{endpoint:02x}")
        if not 0 <= transferred.value <= size:
            raise ProtocolError("libusb 返回无效的传输长度")
        # libusb 超时仍可能交付完整 USB packet，不能丢掉这些字节。
        return ctypes.string_at(buffer, transferred.value)

    def write(self, endpoint: int, data: bytes, *, timeout_ms: int = 1000) -> None:
        self._check_transfer(endpoint, timeout_ms, incoming=False)
        if len(data) > 0x7FFFFFFF:
            raise ValueError("写入大小不能超过 int32")
        buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        deadline = time.monotonic() + timeout_ms / 1000
        offset = 0
        while offset < len(data):
            remaining_ms = min(timeout_ms, math.ceil((deadline - time.monotonic()) * 1000))
            if remaining_ms <= 0:
                raise ProtocolError(f"USB 写入超时，已发送 {offset}/{len(data)} 字节")
            transferred = ctypes.c_int()
            pointer = ctypes.cast(ctypes.byref(buffer, offset), ctypes.POINTER(ctypes.c_ubyte))
            result = self._usb.lib.libusb_bulk_transfer(
                self._handle, endpoint, pointer, len(data) - offset,
                ctypes.byref(transferred), remaining_ms,
            )
            if not 0 <= transferred.value <= len(data) - offset:
                raise ProtocolError("libusb 返回无效的传输长度")
            offset += transferred.value
            if result == _TIMEOUT and offset == len(data):
                return
            self._usb.check(result, f"bulk OUT 0x{endpoint:02x}（已发送 {offset}/{len(data)} 字节）")
            if not transferred.value:
                raise ProtocolError("USB bulk OUT 成功返回但未发送任何字节")

    def close(self) -> None:
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            for interface in reversed(self._claimed):
                self._usb.lib.libusb_release_interface(handle, interface)
        finally:
            self._claimed.clear()
            self._usb.lib.libusb_close(handle)
            self._usb.close()

    def __enter__(self) -> UsbBulkDevice:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
