import psutil
import platform
import sys

def get_core_hardware_info():
    """获取核心硬件配置：CPU+内存+固态硬盘"""
    hardware_info = {}
    # ===== 1. 处理器信息（跨平台适配）=====
    if sys.platform == "win32":
        cpu_model = platform.processor() or "Unknown CPU Model (Windows)"
    elif sys.platform == "linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                cpu_model = [line for line in f if "model name" in line][0].split(":", 1)[1].strip()
        except:
            cpu_model = "Unknown CPU Model (Linux)"
    else:  # macOS
        cpu_model = platform.processor() or "Unknown CPU Model (macOS)"
    hardware_info["CPU"] = cpu_model

    # ===== 2. 内存信息（总大小，GB保留2位小数）=====
    mem_total = psutil.virtual_memory().total / (1024 ** 3)
    hardware_info["Total RAM"] = f"{mem_total:.2f} GB"

    # ===== 3. 固态硬盘信息（总容量，自动过滤机械硬盘，GB保留2位小数）=====
    disk_ssd_total = 0.0
    # 遍历所有磁盘分区，Windows跳过系统保留分区，Linux/Mac跳过临时分区
    for part in psutil.disk_partitions(all=False):
        try:
            disk_usage = psutil.disk_usage(part.mountpoint)
            disk_name = part.device.split(":")[0] if sys.platform == "win32" else part.device
            # 判定固态硬盘（Win：名称含SSD/PCIe/NVMe；Linux/Mac：路径含nvme/sda）
            is_ssd = any(key in disk_name.upper() for key in ["SSD", "NVME", "PCIe"]) or \
                     any(key in disk_name for key in ["nvme", "sda"])
            if is_ssd:
                disk_ssd_total += disk_usage.total / (1024 ** 3)
        except (PermissionError, OSError):
            continue  # 跳过无权限访问的分区
    hardware_info["Total SSD Storage"] = f"{disk_ssd_total:.2f} GB" if disk_ssd_total > 0 else "No SSD Detected"

    return hardware_info

# 主程序：打印硬件信息（适配实验报告格式）
if __name__ == "__main__":
    print("="*50)
    print("        Core Hardware Configuration of Running Env")
    print("="*50)
    core_info = get_core_hardware_info()
    for k, v in core_info.items():
        print(f"{k:<20}: {v}")
    print("="*50)