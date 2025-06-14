import time
from flask import Flask, jsonify, request
import cv2
import os
import requests
import threading  # Adicionado para gerenciamento de threads
from Face_extract import FaceExtract
from FaceRecognition_RealTime import RecognitionRealTime
from Extraction_Emb import delete_all, delete_emb

app = Flask(__name__)

# Caminho de embeddings
embeddings_path = 'embeddings.pt'

# Estado global com thread lock
estado = {
    "modo": "reconhecimento",
    "recarregar_embeddings": True,
    "ultimo_reconhecido": None,
    "lock": threading.Lock()  # Único lock para toda aplicação
}

def get_shared_state():
    return estado 

@app.route('/status', methods=['GET'])
def get_status():
    with estado["lock"]:  # Acesso thread-safe
        return jsonify({
            "modo": estado["modo"],
            "ultimo_reconhecido": estado["ultimo_reconhecido"]
        })

@app.route('/set_modo', methods=['POST'])
def set_modo():
    modo = request.json.get("modo")
    if modo not in ["reconhecimento", "coleta", "pausado"]:
        return jsonify({"error": "Modo inválido"}), 400
    
    with estado["lock"]:  # Acesso thread-safe
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

        # Tenta callback (3 tentativas)
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

        # Atualiza estado com thread safety
        with estado["lock"]:
            estado["recarregar_embeddings"] = True
            estado["modo"] = "reconhecimento"
            
            # Novo: Garantir que o valor persista
            estado["_ultima_atualizacao"] = time.time()

        return jsonify({"status": "completed"}), 200

    except Exception as e:
        print(f"Erro fatal: {str(e)}")
        return jsonify({"error": str(e)}), 500


    
    # delete_all(estado)
    
    # with estado["lock"]:
    #     estado["recarregar_embeddings"] = True  # Sinaliza para recarregar
    #     estado["modo"] = "reconhecimento"
    
    # return jsonify({"message": "Todos os embutimentos foram deletados."}), 200
        
@app.route('/deletar', methods=['POST'])
def deletar():
    rg = request.json.get("rg")
    if not rg:
        return jsonify({"error": "RG não fornecido"}), 400
    
    delete_emb(rg, estado)
    
    with estado["lock"]:
        estado["recarregar_embeddings"] = True  # Sinaliza para recarregar
        estado["modo"] = "reconhecimento"
    
    return jsonify({"message": f"Embutimento de {rg} deletado."}), 200

@app.route('/deletar_todos', methods=['POST'])
def deletar_todos():
    delete_all(estado)
    
    with estado["lock"]:
        estado["recarregar_embeddings"] = True  # Sinaliza para recarregar
        estado["modo"] = "reconhecimento"
    
    return jsonify({"message": "Todos os embutimentos foram deletados."}), 200

@app.route('/reconhecimento', methods=['GET'])
def reconhecimento():
    # Inicia o reconhecimento facial (não é a melhor prática, mas mantive como estava)
    RecognitionRealTime(camera, estado)
    return jsonify({"message": "Reconhecimento iniciado."}), 200

# Gerenciamento da câmera
camera = None

def set_camera_instance(cam):
    global camera
    camera = cam

def start_flask():
    if os.path.exists(embeddings_path):
        app.run(debug=False, use_reloader=False)
    else:
        print("Não existem embeddings criados, execute a coleta de amostras primeiro.")