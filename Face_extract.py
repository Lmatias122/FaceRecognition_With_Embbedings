import asyncio
import os
import time
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
import Extraction_Emb
import sys

async def FaceExtract(nome_pessoa, camera):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device)

    nome_pasta = os.path.join("dataset", nome_pessoa)
    os.makedirs(nome_pasta, exist_ok=True)

    contador = 0
    total_imagens = 35

    print(f"Iniciando coleta para: {nome_pessoa}")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Erro ao acessar a câmera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]

        # Moldura centralizada
        box_width, box_height = 250, 300
        x1_ref = (frame_width - box_width) // 2
        y1_ref = (frame_height - box_height) // 2
        x2_ref = x1_ref + box_width
        y2_ref = y1_ref + box_height

        cv2.rectangle(frame, (x1_ref, y1_ref), (x2_ref, y2_ref), (255, 255, 0), 2)
        cv2.putText(frame, "Centralize o rosto na moldura", (x1_ref - 40, y1_ref - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        face = mtcnn(rgb_frame)

        if face is not None and isinstance(face, torch.Tensor):
            face = face.squeeze(0)
            face = face.clamp(0, 1)
            face_image = (face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            alpha = 1.4
            beta = 60
            face_image = cv2.convertScaleAbs(face_image, alpha=alpha, beta=beta)
            bgr_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(f"{nome_pasta}/imagem_{contador}.jpg", bgr_face_image)
            print(f"Imagem {contador + 1}/{total_imagens} salva.")
            contador += 1

            cv2.imshow("Face Detectada", bgr_face_image)
            time.sleep(0.3)
        else:
            cv2.putText(frame, "Nenhum rosto detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or contador >= total_imagens:
            break
        elif key == ord('z'):
            print("Encerrando o sistema.")
            camera.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    print(f"Captura Finalizada! {contador} imagens salvas em {nome_pasta}")
    Extraction_Emb.CreateEmb()

    cv2.destroyAllWindows()
    await asyncio.sleep(0.1)
