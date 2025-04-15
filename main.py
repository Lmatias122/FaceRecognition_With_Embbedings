import asyncio
import os
import aioconsole
import cv2
import torch
from Face_extract import FaceExtract
from comando import aguardar_comando
from FaceRecognition_RealTime import RecognitionRealTime

async def main():
    camera = cv2.VideoCapture(0)
    embeddings_path = 'embeddings.pt'
    estado = {
    "modo": "reconhecimento",
    "recarregar_embeddings": False,
    "ultimo_reconhecido": None}

    
    if os.path.exists(embeddings_path):
        await asyncio.gather(
        RecognitionRealTime(camera, estado),
        aguardar_comando(camera, estado)
    )        
    else:         
         nome =  input("Não existe um modelo de reconhecimento Criado. Digite o nome da pessoa que deseja inserir no modelo: ")
         await FaceExtract(nome, camera)

   

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
