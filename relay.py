import asyncio
import websockets
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
import uuid

RELAY_IP, RELAY_PORT = "localhost", 8765
SERVER_IP, SERVER_PORT = "localhost", 8000
CHUNK = 2048

# List of connected clients
connected = dict()


# Create handler for each connections
async def handler(client_socket, path: str = None):
    async with connect(f"ws://{SERVER_IP}:{SERVER_PORT}") as relay_socket:
        try:
            client_id = str(uuid.uuid4())
            print(f"Client {client_id} has connected")
            connected[client_id] = (client_socket, relay_socket)
            await relay_socket.send(client_id)
            await relay(client_socket, relay_socket)
        except websockets.exceptions.ConnectionClosed as e:
            print(e)
        finally:
            del connected[client_id]
            print(f"Client {client_id} has disconnected")


async def relay(client_socket, relay_socket):
    async for msg in client_socket:
        await relay_socket.send(msg)


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
