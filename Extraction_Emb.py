import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

def CreateEmb():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = 'dataset'
    embeddings_path = 'embeddings.pt'

    # Carrega embeddings existentes (se houver)
    if os.path.exists(embeddings_path):
        data = torch.load(embeddings_path)
    else:
        data = {}

    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    for pessoa in os.listdir(dataset_path):
        pasta_pessoa = os.path.join(dataset_path, pessoa)
        if not os.path.isdir(pasta_pessoa):
            continue

        embeddings = []
        for img_nome in os.listdir(pasta_pessoa):
            img_path = os.path.join(pasta_pessoa, img_nome)
            img = Image.open(img_path)
            face = mtcnn(img)
            if face is not None:
                face = face.unsqueeze(0).to(device)
                emb = resnet(face).detach().cpu()
                embeddings.append(emb)

        if embeddings:
            media = torch.cat(embeddings).mean(0, keepdim=True)
            data[pessoa] = media  # substitui ou adiciona

    torch.save(data, embeddings_path)
    print("Embeddings atualizados e salvos com sucesso.")

    # Remove imagens após gerar os embeddings
    for pessoa in os.listdir(dataset_path):
        pasta_pessoa = os.path.join(dataset_path, pessoa)
        for img_file in os.listdir(pasta_pessoa):
            os.remove(os.path.join(pasta_pessoa, img_file))
        os.rmdir(pasta_pessoa)  # remove pasta da pessoa

async def delete_emb(rg,estado):
    try:
        embeddings = torch.load("embeddings.pt")
        if rg in embeddings:
            del embeddings[rg]
            torch.save(embeddings, "embeddings.pt")
            print(f"{rg} removido com sucesso.")
            estado["recarregar_embeddings"] = True
        else:
            print(f"{rg} não encontrado.")
    except Exception as e:
        print(f"Erro ao remover pessoa: {e}")

