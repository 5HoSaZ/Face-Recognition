import os
import math
import asyncio
import websockets
import numpy as np
from PIL import Image

import torch
import pandas as pd

from face_detection import FaceDetectionPipeline
from models.net import ConvolutionNet
from models.transform import PILToNormalizedTensor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FaceRecognition:
    def __init__(self, model_path: str, device: str):
        self.device = device
        # Get mapper
        self.mapper = {
            int(row["Target"]): str(row["Name"])
            for _, row in pd.read_csv("./data/processed/mapping.csv").iterrows()
        }
        # Get pipeline
        self.pipeline = FaceDetectionPipeline(device)
        self.normalize = PILToNormalizedTensor()
        # Get model
        self.num_classes = len(self.mapper)
        self.model = ConvolutionNet(self.num_classes, 256, 0.2)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.to(device)
        self.model.eval()

    def predicts(self, image: Image):
        image = self.pipeline(image)
        image = self.normalize(image)
        image = image.unsqueeze(0).to(self.device)
        outputs = torch.exp(self.model(image)).max(dim=1)
        pred, prob = outputs.indices, outputs.values
        pred, prob = self.mapper[int(pred.item())], float(prob.item())
        return pred, prob


recognition = FaceRecognition("./models/convolution_net_1.pth", DEVICE)

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
        await websocket.send("Server received image!")
        print("Predicting...")
        name, probability = recognition.predicts(image)
        response = f"{name}, Confidence: {probability:.2%}"
        print(response)
        # await websocket.send(response)


async def main():
    async with websockets.serve(handler, SERVER_IP, SERVER_PORT):
        print(f"Server started at ws://{SERVER_IP}:{SERVER_PORT}.")
        await asyncio.get_running_loop().create_future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Terminating...")
