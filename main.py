from Face_extract import FaceExtract
from FaceRecognition_RealTime import RecognitionRealTime

def main():
    print("Bem-Vindo ao sistema de reconhecimento!")
    resposta = input("Deseja adicionar um novo morador? (s/n): ").strip().lower()

    if resposta == 's':
        nome = input("Digite o nome do novo usuario e se prepare para a coleta de faces: ").strip().lower()

        FaceExtract(nome)
    else:
        print("Iniciando reconhecimento facial em tempo real... ")
        RecognitionRealTime()

if __name__ == "__main__":
    main()