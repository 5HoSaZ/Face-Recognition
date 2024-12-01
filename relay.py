import asyncio
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
import uuid

# RELAY_IP, RELAY_PORT = "192.168.1.108", 8765
RELAY_IP, RELAY_PORT = "192.168.35.144", 8765
SERVER_IP, SERVER_PORT = "172.26.162.157", 8888
CHUNK = 2048

# List of connected clients
connected = dict()


# Create handler for each connections
async def handler(client_socket, path: str = None):
    client_id = uuid.uuid4()
    try:
        async with connect(f"ws://{SERVER_IP}:{SERVER_PORT}") as relay_socket:
            connected[client_id] = (client_socket, relay_socket)
            print("Relay a client to server")
            async for msg in client_socket:
                await relay_socket.send(msg)
            await relay_socket.close()
            del connected[client_id]
            print("A client has disconnected")
    except Exception:
        print("Error connecting to server!")


async def main():
    async with serve(handler, RELAY_IP, RELAY_PORT):
        print(f"Relay started at ws://{RELAY_IP}:{RELAY_PORT}.")
        print(f"Relay to server at ws://{SERVER_IP}:{SERVER_PORT}.")
        await asyncio.get_running_loop().create_future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Terminating...")
