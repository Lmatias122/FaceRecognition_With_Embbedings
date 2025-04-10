import asyncio
import cv2
from comando import aguardar_comando
from FaceRecognition_RealTime import RecognitionRealTime

async def main():
    camera = cv2.VideoCapture(0)

    estado = {"modo": "reconhecimento"}  # modo: "reconhecimento" ou "coleta"

    await asyncio.gather(
        RecognitionRealTime(camera, estado),
        aguardar_comando(camera, estado)
    )

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
