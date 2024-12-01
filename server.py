import os
import math
import asyncio
import websockets
import numpy as np
from PIL import Image

SERVER_IP, SERVER_PORT = "172.26.162.157", 8888
CHUNK = 2048

# List of connected clients
connected = dict()


# Create handler for each connections
async def handler(websocket, path: str = None):
    try:
        client_id = await websocket.recv()
        connected[client_id] = websocket
        print(f"Client {client_id} has connected")
        await process(websocket)
    except websockets.exceptions.ConnectionClosed as e:
        print(e)
    finally:
        del connected[client_id]
        print(f"Client {client_id} has disconnected")


async def process(websocket):
    while True:
        image_properties = (await websocket.recv())[8:]
        height = int.from_bytes(image_properties[0:2], byteorder="little")
        width = int.from_bytes(image_properties[2:4], byteorder="little")
        with open("tmp.bin", "wb") as tmp:
            for i in range(chunk_count := math.ceil(height * width * 3 / CHUNK)):
                image_chunk = (await websocket.recv())[8:]
                tmp.write(image_chunk)
                print(
                    f"Receiving image - Progress: {(i + 1) / chunk_count :.0%}",
                    end="\r",
                )
                await asyncio.sleep(0.01)
        image_data = np.fromfile("tmp.bin", dtype=np.uint8)
        os.remove("tmp.bin")
        image_data = image_data.reshape(height, width, 3)
        image = Image.fromarray(image_data)
        print("\nSaving image, shape:", image_data.shape)
        image.save("./images/Image.jpg")
        await websocket.send("Server received image")


async def main():
    async with websockets.serve(handler, SERVER_IP, SERVER_PORT):
        print(f"Server started at ws://{SERVER_IP}:{SERVER_PORT}.")
        await asyncio.get_running_loop().create_future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Terminating...")
