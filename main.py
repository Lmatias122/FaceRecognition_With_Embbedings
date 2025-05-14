import asyncio
import os
import threading
import aioconsole
import cv2
import torch
from Face_extract import FaceExtract
from comando import aguardar_comando
from FaceRecognition_RealTime import RecognitionRealTime
from app import get_shared_state, start_flask
from app import app, set_camera_instance

async def main():

    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    camera = cv2.VideoCapture(0)
    set_camera_instance(camera)
    
    embeddings_path = 'embeddings.pt'
    estado = get_shared_state()

    
    if os.path.exists(embeddings_path):
        await asyncio.gather(
        RecognitionRealTime(camera, estado),
        # aguardar_comando(camera, estado)
    )        
    else:         
         nome =  input("Não existe um modelo de reconhecimento Criado. Digite o nome da pessoa que deseja inserir no modelo: ")
         await FaceExtract(nome, camera)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
