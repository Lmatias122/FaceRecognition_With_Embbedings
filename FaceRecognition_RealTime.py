import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

threshold = 0.8

def RecognitionRealTime(camera, estado):
    print("Bem-Vindo ao sistema de reconhecimento!")

    embeddings = {}
    modo_anterior = None
    contador = 0
    while True:
        if estado["modo"] != "reconhecimento":
            continue

        # Recarrega embeddings se necessário
        if embeddings == {} or estado.get("recarregar_embeddings") or modo_anterior != "reconhecimento":
            try:
                embeddings = torch.load('embeddings.pt')     
                estado["recarregar_embeddings"] = False
            except Exception as e:
                print("Erro ao carregar embeddings:", e)
                embeddings = {}

        modo_anterior = estado["modo"]

        ret, frame = camera.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        boxes, _ = mtcnn.detect(img_pil)

        frame_height, frame_width = frame.shape[:2]
        box_width, box_height = 250, 300
        x1_ref = (frame_width - box_width) // 2
        y1_ref = (frame_height - box_height) // 2
        x2_ref = x1_ref + box_width
        y2_ref = y1_ref + box_height

        if boxes is not None:
            box = boxes[0]
            x1_face, y1_face, x2_face, y2_face = box

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
                            if dist < threshold:
                                nome = pessoa
                            else:
                                nome = "Rosto desconhecido"

                    if nome != "Desconhecido":
                        estado["ultimo_reconhecido"] = nome
                        print(f"Reconhecido: {nome} com distância {menor_dist.item():.4f}")
                    else:
                        print("Rosto não reconhecido.")

                cv2.putText(frame, f"Reconhecido: {nome}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
