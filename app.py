from venv import logger
from flask import Flask, jsonify, request
import cv2
import os
import requests
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
    try:
        data = request.get_json()
        rg = data.get('rg')
        callback_url = data.get('callbackUrl')
        
        if not rg:
            return jsonify({"error": "RG não fornecido"}), 400
        if not callback_url:
            return jsonify({"error": "URL de callback não fornecida"}), 400

        FaceExtract(rg, camera)

        for attempt in range(3):
            try:
                response = requests.post(
                    callback_url,
                    json={"rg": rg, "success": True},
                    timeout=5
                )
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                if attempt == 2: 
                    raise
        
        if response.status_code == 200:
            print("Processamento concluído com sucesso, encerrando...")
            return jsonify({"status": "completed"}), 200
        else:
            return jsonify({"error": "Callback failed"}), 500

    except Exception as e:
        print(f"Erro fatal: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if 'camera' in globals():
            camera.release()
        
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
