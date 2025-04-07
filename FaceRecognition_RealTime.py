import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carrega modelo e embeddings
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
embeddings = torch.load('embeddings.pt')

# Inicia webcam
camera = cv2.VideoCapture(0)
threshold = 0.8  # quanto menor, mais rigoroso

print("Pressione 'q' para sair.")

while True:
    ret, frame = camera.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(img_rgb)


    # Detecta face e bbox
    boxes, _ = mtcnn.detect(img_pil)

    # Define a área da moldura (centralizada)
    frame_height, frame_width = frame.shape[:2]
    box_width, box_height = 250, 300
    x1_ref = (frame_width - box_width) // 2
    y1_ref = (frame_height - box_height) // 2
    x2_ref = x1_ref + box_width
    y2_ref = y1_ref + box_height

    if boxes is not None:
        box = boxes[0]  # como keep_all=False, só tem uma
        x1_face, y1_face, x2_face, y2_face = box

        # Verifica se a face está dentro da moldura
        if x1_face >= x1_ref and y1_face >= y1_ref and x2_face <= x2_ref and y2_face <= y2_ref:
            face = mtcnn(img_pil)
            if face is not None:
                face = face.unsqueeze(0).to(device)
                emb = resnet(face).detach().cpu()

                menor_dist = float('inf')
                nome = "Desconhecido"

                for pessoa, emb_salvo in embeddings.items():
                    dist = torch.nn.functional.pairwise_distance(emb, emb_salvo)
                    if dist < menor_dist:
                        menor_dist = dist
                        nome = pessoa if dist < threshold else "Desconhecido"

                # Exibe resultado
                cv2.putText(frame, f"{nome} ({float(menor_dist):.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "Aproxime o rosto da moldura", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


        # Desenha nome
        # Define a área da moldura (centralizada)
        frame_height, frame_width = frame.shape[:2]
        box_width, box_height = 250, 300
        x1 = (frame_width - box_width) // 2
        y1 = (frame_height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height

        # Desenha o retângulo (verde) como guia na imagem original
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Adiciona texto explicativo
        cv2.putText(frame, "Posicione seu rosto dentro da moldura", (x1 - 30, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        


    cv2.imshow("Reconhecimento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
