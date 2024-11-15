import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fer import FER

app = FastAPI()
detector = FER()

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:  # Keep the connection open for continuous frames
            try:
                payload = await websocket.receive_text()  # Receive frame from the client
                payload = json.loads(payload)

                if 'data' not in payload or 'image' not in payload['data']:
                    raise ValueError("Missing image data in the payload")

                imageByt64 = payload['data']['image'].split(',')[1]  

                # Decode and convert into image
                image = np.frombuffer(base64.b64decode(imageByt64), np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                if image is None:
                    raise ValueError("Failed to decode the image")

                print("Frame received")

                # Detect emotion via FER model
                prediction = detector.detect_emotions(image)

                if not prediction:
                    raise ValueError("No emotions detected")

                # Extract emotions and calculate percentages
                emotions = prediction[0]['emotions']
                total = sum(emotions.values())
                emotion_percentages = {emotion: round((value / total) * 100, 2) for emotion, value in emotions.items()}

                # Log the emotion percentages
                print(emotion_percentages)

                # Find the dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                response = {
                    "predictions": emotions,
                    "emotion": dominant_emotion,
                    "percentages": emotion_percentages
                }

                # Send the prediction to the client
                await websocket.send_json(response)
                print(f"Prediction sent: {response['emotion']}")

            except Exception as e:
                print(f"Error processing frame: {e}")
                # Send error message to the client
                error_response = {"error": str(e)}
                await websocket.send_json(error_response)
                break  # Optional: break the loop if there's an error

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        print("WebSocket connection closed")