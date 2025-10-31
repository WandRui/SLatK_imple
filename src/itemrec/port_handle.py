import socket

def is_port_available(port: int) -> bool:
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def find_available_port(start_port: int) -> int:
    """查找可用的端口号"""
    port = start_port
    while port < 65535:
        if is_port_available(port) and is_port_available(port + 1):
            return port
        port += 1
    raise RuntimeError("No available ports!")