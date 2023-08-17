import socket

HOST = '127.0.0.1'
PORT = 5680

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected to TCP Server")
        
        while True:
            data = s.recv(2084)
            if not data:
                break
            print("Received Data (Raw):", data)

if __name__ == "__main__":
    main()
