# Sistema de Detecção e Autenticação Facial com OpenCV  
**Sprint 3 – Nexora – IoT & IoB – Turma 3ESPY**

## 👥 Autores
- **Gabriel Machado** – RM 99880  
- **Lourenzo Ramos** – RM 99951  
- **Vitor Hugo Rodrigues** – RM 97758  
- **Leticia Resina** – RM 98069  

---

## 🎯 Objetivo
Aplicação **local (desktop/notebook)** para **detecção e autenticação facial** do usuário, utilizando **OpenCV + Haar Cascade** para detecção e **LBPH** para reconhecimento.  
Feedback visual em tempo real:  
- 🔴 **Vermelho**: rosto detectado, **não autenticado**.  
- 🟢 **Verde + “Autenticado”**: rosto bate com a imagem cadastrada.

---

## 🧰 Tecnologias
- Python 3.10+  
- OpenCV (`opencv-python`, `opencv-contrib-python`)  
- Haar Cascade (detecção)  
- LBPH (autenticação)

---

## ⚙️ Instalação
```bash
pip install opencv-python opencv-contrib-python numpy
```

Coloque o arquivo **`haarcascade_frontalface_default.xml`** na mesma pasta do script.  
(Disponível no repositório oficial do OpenCV.)

---

## ▶️ Execução
```bash
python detecta_rosto.py
```

**Menu:**
```
[1] Tirar/atualizar foto de cadastro
[2] Iniciar autenticação em tempo real
[0] Sair
```

**Fluxo:**
- **[1] Cadastro**:  
  - Se já existir imagem: digite **1** para apagar e cadastrar nova ou **0** para manter.  
  - Na janela: **c** para capturar, **q** para sair sem salvar.
- **[2] Autenticação**:  
  - Exibe retângulos coloridos conforme o reconhecimento.

---

## 🗂️ Estrutura
```
├── detecta_rosto.py
├── haarcascade_frontalface_default.xml
├── data/
│   └── usuario.jpg
└── README.md
```

---

## 🔧 Parâmetros (e impacto)
- `FACE_SIZE = (200, 200)` – padroniza entrada do LBPH.  
- `LBPH_THRESHOLD = 60` – limite de distância (quanto **menor**, mais parecido).  
- Equalização de histograma usada no treino e na predição para maior robustez.

**Dicas de calibração:**  
- Falso negativo (não autentica): aumentar levemente o threshold (ex.: 65–70) e refazer o cadastro com boa iluminação.  
- Falso positivo (autentica errado): diminuir o threshold (ex.: 50–55).

---

## 🚫 Limitações & ➕ Próximos Passos
**Limitações**
- Apenas **um usuário** cadastrado (uma imagem).  
- Sensível à iluminação/posição frontal.

**Próximos passos**
- Múltiplos usuários (IDs/labels distintos e várias amostras por pessoa).  
- GUI (Tkinter/Qt) e logs.  
- Pipelines IA (Mediapipe/FaceNet) para embeddings mais robustos.  
- Integração com IoT (controle de acesso, trancas, etc.).

---

## ⚖️ Ética, Privacidade & Uso Responsável
Este projeto é **exclusivamente educacional e de pesquisa acadêmica** (FIAP – IoT & IoB, 3ESPY).  
Não deve ser utilizado em ambientes de produção ou para **processos decisórios sensíveis** (controle de acesso real, validação legal/biométrica, vigilância, etc.).

**Princípios adotados:**
- Todo o processamento e armazenamento são **locais**; a imagem de cadastro fica apenas em `data/usuario.jpg`.  
- **Mínima coleta**: o sistema persiste **apenas uma imagem** estritamente necessária para autenticação.  
- **Transparência**: o algoritmo usado (Haar + LBPH), parâmetros e limitações são descritos neste README.  
- **Reversibilidade**: o usuário pode apagar a imagem a qualquer momento (opção **1** no menu de cadastro).    
- **Conformidade**: boas práticas alinhadas à **LGPD**, evitando uso não autorizado, reuso indevido ou transferência de dados pessoais.  
- **Finalidade educacional**: este projeto é voltado para **estudo e conhecimento**, proporcionando aprendizado prático em **visão computacional** e **reconhecimento facial**.