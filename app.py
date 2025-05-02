from flask import Flask, jsonify, request
import cv2
import os
import torch
from Face_extract import FaceExtract
from FaceRecognition_RealTime import RecognitionRealTime
from Extraction_Emb import delete_all, delete_emb
from shared import modo_reconhecimento

app = Flask(__name__)

# Caminho de embeddings
embeddings_path = 'embeddings.pt'

# Estado global
estado = {
    "modo": "reconhecimento",
    "recarregar_embeddings": False,
    "ultimo_reconhecido": None
}

# Inicializa a câmera
camera = cv2.VideoCapture(0)

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({"modo": estado["modo"], "ultimo_reconhecido": estado["ultimo_reconhecido"]})

@app.route('/set_modo', methods=['POST'])
def set_modo():
    modo = request.json.get("modo")
    if modo not in ["reconhecimento", "coleta", "pausado"]:
        return jsonify({"error": "Modo inválido"}), 400
    estado["modo"] = modo
    return jsonify({"message": f"Modo alterado para {modo}"}), 200

@app.route('/coletar', methods=['POST'])
def coletar():
    nome = request.json.get("nome")
    if not nome:
        return jsonify({"error": "Nome não fornecido"}), 400
    FaceExtract(nome, camera)  # Processo de coleta de imagens para o FaceNet
    estado["modo"] = "reconhecimento"
    return jsonify({"message": "Coleta finalizada e modo alterado para reconhecimento."}), 200

@app.route('/deletar', methods=['POST'])
def deletar():
    rg = request.json.get("rg")
    if not rg:
        return jsonify({"error": "RG não fornecido"}), 400
    delete_emb(rg, estado)
    estado["modo"] = "reconhecimento"
    return jsonify({"message": f"Embutimento de {rg} deletado."}), 200

@app.route('/deletar_todos', methods=['POST'])
def deletar_todos():
    delete_all(estado)
    estado["modo"] = "reconhecimento"
    return jsonify({"message": "Todos os embutimentos foram deletados."}), 200

@app.route('/reconhecimento', methods=['GET'])
def reconhecimento():
    # Exemplo de como iniciar o reconhecimento facial
    RecognitionRealTime(camera, estado)
    return jsonify({"message": "Reconhecimento iniciado."}), 200

if __name__ == '__main__':
    if os.path.exists(embeddings_path):
        app.run(debug=True, use_reloader=False)
    else:
        print("Não existem embeddings criados, execute a coleta de amostras primeiro.")
