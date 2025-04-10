import aioconsole
from Face_extract import FaceExtract

async def aguardar_comando(camera, estado):
    while True:
        comando = await aioconsole.ainput("Digite 'coletar' para coletar amostras, ou 'voltar' para retornar ao reconhecimento: ")
        if comando.strip().lower() == 'coletar':
            nome = await aioconsole.ainput("Digite o nome da pessoa: ")
            estado["modo"] = "coleta"
            await FaceExtract(nome, camera)
            estado["modo"] = "reconhecimento"
            print("Reconhecimento retomado.")
        elif comando.strip().lower() == 'voltar':
            estado["modo"] = "reconhecimento"
