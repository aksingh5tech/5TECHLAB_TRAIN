import socket

def run_server(host='0.0.0.0', port=5000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                conn.sendall(b"Hello, client!")
                break  # Close after one connection for simplicity, remove this line to keep listening.

if __name__ == "__main__":
    run_server(host='0.0.0.0', port=5000)
