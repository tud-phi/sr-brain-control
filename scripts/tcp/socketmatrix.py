import socket

# Define the server address and port
HOST = "127.0.0.1"
PORT = 5680


def decode_stimulation(byte_data):
    # Decode the first byte to determine the stimulation type
    stimulation_type = byte_data[0]

    # Return the decoded stimulation type
    return stimulation_type


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connected to TCP Server")

        while True:
            data = s.recv(8)  # Here we cna adjust the buffer size as needed
            if not data:
                break

            # Decode the received data
            decoded_stimulation = decode_stimulation(data)

            # Print the decoded stimulation type
            print("Decoded Stimulation:", decoded_stimulation)


if __name__ == "__main__":
    main()
