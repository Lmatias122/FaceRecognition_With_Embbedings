import os
import time
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
import Extraction_Emb


def FaceExtract(nome_pessoa):
    # Define o dispositivo (GPU se disponível)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inicializa o detector de rostos MTCNN
    mtcnn = MTCNN(keep_all=False, device=device)

    # Pasta onde as imagens serão salvas
    nome_pasta = os.path.join("dataset/",nome_pessoa)
    os.makedirs(nome_pasta, exist_ok=True)

    # Captura da câmera
    camera = cv2.VideoCapture(0)

    contador = 0
    total_imagens = 35

    print("Iniciando captura. Pressione 'q' para sair...")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Erro ao acessar a câmera.")
            break

        # Converte de BGR (OpenCV) para RGB (necessário para MTCNN)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecta e extrai o rosto
        face = mtcnn(rgb_frame)

        if face is not None and isinstance(face, torch.Tensor):
            time.sleep(0.3)
            face = face.squeeze(0)  # Remove dimensão de batch (caso exista)
            face = face.clamp(0, 1)  # Garante que os valores estão entre 0 e 1

            # Converte tensor para imagem RGB
            face_image = (face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Aplica correção de brilho/contraste mais forte
            alpha = 1.4  # contraste
            beta = 60    # brilho
            face_image = cv2.convertScaleAbs(face_image, alpha=alpha, beta=beta)

            # Converte RGB para BGR para salvar corretamente com OpenCV
            bgr_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

            # Salva a imagem
            cv2.imwrite(f"{nome_pasta}/imagem_{contador}.jpg", bgr_face_image)
            print(f"Imagem {contador + 1}/{total_imagens} salva.")
            contador += 1
           

            # Mostra a imagem da face extraída
            cv2.imshow("Face Detectada", bgr_face_image)

        # Mostra o frame da webcam
        cv2.imshow("Webcam", frame)

        # Interrompe com 'q' ou se atingir o total de imagens
        if cv2.waitKey(1) & 0xFF == ord('q') or contador >= total_imagens:
            break

    # Libera a câmera e fecha as janelas
    camera.release()
    cv2.destroyAllWindows()
    Extraction_Emb.CreateEmb()
    print(f"Captura Finalizada! {contador} imagens salvas em {nome_pasta}")
