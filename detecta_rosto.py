# Sistema de Detecção e Autenticação Facial com OpenCV
# Sprint 3 - Nexora - Iot & IoB - 3ESPY
# Autores: 
# 
# - Gabriel Machado - RM 99880
# - Lourenzo Ramos - RM 99951
# - Vitor Hugo Rodrigues - RM 97758
# - Leticia Resina - RM 98069
#   
# Requisitos: pip install opencv-contrib-python
# Arquivo do cascade: haarcascade_frontalface_default.xml no mesmo diretório

import os
import cv2
from datetime import datetime

# ==== CONFIGS ====
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATA_DIR = 'data'
CADASTRO_IMG = os.path.join(DATA_DIR, 'usuario.jpg')
FACE_SIZE = (200, 200)   # tamanho padrão para treinar/comparar
LBPH_THRESHOLD = 60      # quanto menor o valor retornado, maior a confiança

# ==== UTILS ====
def garantir_cascade():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError(
            f"Erro: não foi possível carregar '{CASCADE_PATH}'. "
            "Baixe o haarcascade_frontalface_default.xml do OpenCV e deixe no mesmo diretório."
        )
    return face_cascade

def abrir_camera(index=0, largura=640, altura=480):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Erro: não foi possível acessar a webcam.")
    # tentar setar resolução (nem sempre aplica)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    return cap

def detectar_faces(gray, face_cascade):
    # retorna (x,y,w,h)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )
    return faces

def maior_face(faces):
    if len(faces) == 0:
        return None
    # escolhe a maior área (w*h)
    return max(faces, key=lambda f: f[2] * f[3])

def preparar_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

# ==== CAPTURA DE CADASTRO ====
def capturar_rosto_cadastro():
    """
    Captura 1 foto de rosto e salva em data/usuario.jpg.
    Se já existir, pergunta no console:
      0 = manter
      1 = apagar e cadastrar novo
    """
    face_cascade = garantir_cascade()
    preparar_dir()

    # Se já existir imagem, perguntar o que fazer
    if os.path.exists(CADASTRO_IMG):
        print(f"Já existe uma imagem cadastrada em: {CADASTRO_IMG}")
        escolha = input("Digite 1 para apagar e cadastrar nova, ou 0 para manter: ").strip()
        if escolha == "0":
            print("Mantendo a imagem atual.")
            return False
        elif escolha == "1":
            os.remove(CADASTRO_IMG)
            print("Imagem anterior apagada.")
        else:
            print("Opção inválida. Cancelando cadastro.")
            return False

    cap = abrir_camera()
    print("[Cadastro] Posicione o rosto. 'c'=capturar | 'q'=sair sem salvar")
    salvo = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Erro na webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detectar_faces(gray, face_cascade)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        cv2.imshow("Cadastro de Rosto", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            if len(faces) == 0:
                print("Nenhum rosto detectado. Tente de novo.")
                continue

            (x, y, w, h) = maior_face(faces)
            rosto = gray[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, FACE_SIZE)
            rosto = cv2.equalizeHist(rosto)

            cv2.imwrite(CADASTRO_IMG, rosto)
            print(f"Foto salva em {CADASTRO_IMG}")
            salvo = True
            cv2.imshow("Rosto Salvo", rosto)
            cv2.waitKey(700)
            break

    cap.release()
    cv2.destroyAllWindows()
    return salvo


# ==== TREINO / CARREGAMENTO DO MODELO ====
def treinar_modelo_lbph():
    """
    Treina um reconhecedor LBPH com a imagem de cadastro.
    Retorna o reconhecedor pronto para 'predict'.
    """
    import numpy as np
    if not os.path.exists(CADASTRO_IMG):
        raise FileNotFoundError(
            f"Imagem de cadastro não encontrada em '{CADASTRO_IMG}'. "
            "Use a opção (1) para cadastrar antes."
        )

    img = cv2.imread(CADASTRO_IMG, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Falha ao ler imagem de cadastro. Refaça o cadastro.")

    # normaliza o tamanho e melhora contraste (robustez)
    rosto = cv2.resize(img, FACE_SIZE)
    rosto = cv2.equalizeHist(rosto)

    # precisa do pacote contrib
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "LBPH Face Recognizer indisponível. Instale:\n"
            "pip install opencv-contrib-python"
        )

    # Você pode ajustar os parâmetros se quiser:
    # radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=DBL_MAX
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train([rosto], np.array([1], dtype=np.int32))
    return recognizer


# ==== AUTENTICAÇÃO EM TEMPO REAL ====
def autenticar_ao_vivo():
    """
    Abre a webcam e, para cada rosto detectado:
      - Desenha retângulo VERMELHO e texto 'Rosto Detectado' se não reconhecer (distância >= threshold)
      - Desenha retângulo VERDE e 'Autenticado' se reconhecer (distância < threshold)
    Pressione 'q' para encerrar.
    """
    face_cascade = garantir_cascade()
    recognizer = treinar_modelo_lbph()

    cap = abrir_camera()
    print("[Autenticação] Pressione 'q' para encerrar.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Falha ao ler frame da webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detectar_faces(gray, face_cascade)

        for (x, y, w, h) in faces:
            rosto = gray[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, FACE_SIZE)
            rosto = cv2.equalizeHist(rosto)

            # predict retorna (label, distance). Quanto MENOR a distance, melhor (mais parecido).
            label, distance = recognizer.predict(rosto)

            if distance < LBPH_THRESHOLD:
                # Autenticado
                cor = (0, 255, 0)  # verde
                texto = f"Autenticado (dist={distance:.1f})"
            else:
                # Não autenticado
                cor = (0, 0, 255)  # vermelho
                texto = f"Rosto Detectado (dist={distance:.1f})"

            cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
            cv2.putText(frame, texto, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        # sempre mostrar a janela, mesmo sem rosto (retângulos só quando tiver)
        cv2.imshow("Autenticacao Facial (LBPH)", frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==== MENU CLI ====
def menu():
    print("="*60)
    print("        >>> Sistema de Detecção & Autenticação Facial <<<")
    print("="*60)
    print("[1] Tirar/atualizar foto de cadastro")
    print("[2] Iniciar autenticação em tempo real")
    print("[0] Sair")
    print("-"*60)

def main():
    while True:
        menu()
        op = input("Escolha uma opção: ").strip()
        if op == '1':
            try:
                capturar_rosto_cadastro()
            except Exception as e:
                print(f"Erro no cadastro: {e}")
        elif op == '2':
            try:
                autenticar_ao_vivo()
            except Exception as e:
                print(f"Erro na autenticação: {e}")
        elif op == '0':
            print("Encerrando...")
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()