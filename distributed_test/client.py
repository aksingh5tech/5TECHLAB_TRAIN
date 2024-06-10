import socket

def test_connection(host='217.18.53.77', port=5000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            data = s.recv(1024)
            print(f"Received: {data.decode()}")
        except socket.error as e:
            print(f"Failed to connect to {host}:{port}")
            print(e)

if __name__ == "__main__":
    test_connection(host='217.18.53.77', port=5000)
