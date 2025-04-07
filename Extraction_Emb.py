import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

def CreateEmb():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_path = 'dataset'  # pasta com subpastas por pessoa
    embeddings_path = 'embeddings.pt'
    
    # Inicializa detector e extrator
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    data = {}
    
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
            data[pessoa] = media
    
    # Salva os embeddings
    torch.save(data, embeddings_path)
    print("Embeddings salvos com sucesso.")
