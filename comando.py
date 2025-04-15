import aioconsole
from Face_extract import FaceExtract
from Extraction_Emb import delete_emb

async def aguardar_comando(camera, estado):
    while True:
        comando = await aioconsole.ainput("Digite 'coletar' para coletar amostras,'voltar' para retornar ao reconhecimento ou 'deletar' para deletar um usuario: ")
        if comando.strip().lower() == 'coletar':
            nome = await aioconsole.ainput("Digite o nome da pessoa: ")
            estado["modo"] = "coleta"
            await FaceExtract(nome, camera)
            estado["modo"] = "reconhecimento"
            print("Reconhecimento retomado.")
        elif comando.strip().lower() == 'voltar':
            estado["modo"] = "reconhecimento"
        elif comando.strip().lower() == 'deletar':
            rg = await aioconsole.ainput("Digite o rg da pessoa: ")
            await delete_emb(rg, estado)
            estado["modo"] = "reconhecimento"
