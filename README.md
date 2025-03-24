---

# **Detecção de Tubarões com YOLO (Drone Shark Detection Dataset)**


<div alt="linguagens"><br>

 <img alt="GITHUB" src="https://img.shields.io/badge/GITHUB-800080?style=for-the-badge&logo=github&logoColor=white">

<img alt="PYTHON" src="https://img.shields.io/badge/PYTHON-1a4a3f?style=for-the-badge&logo=python&logoColor=white">

<img alt="LABEL-STUDIO" src="https://img.shields.io/badge/LABEL-STUDIO-he4a3f?style=for-the-badge&logo=LABEL-STUDIO&logoColor=white">  

</div><br>

## **1. Introdução**
Este notebook tem como objetivo treinar modelos de detecção de objetos usando a biblioteca **Ultralytics** com diferentes versões do modelo **YOLO** (You Only Look Once). O modelo será treinado para identificar tubarões em imagens capturadas por drones, utilizando o **[Drone Shark Detection Dataset](https://www.kaggle.com/datasets/erick88santos/drone-shark-detection-dataset)**.

## **2. Configuração do Ambiente**
Antes de iniciar o treinamento, foi necessário configurar o ambiente:
- **Instalação das dependências**: Foi instalada a biblioteca `ultralytics`, que contém a implementação oficial do YOLOv5, YOLOv8 e YOLO11.
- **Importação das bibliotecas**: Foram importadas as bibliotecas essenciais para o pré-processamento de dados, treinamento e avaliação do modelo, incluindo `torch`, `cv2` e `matplotlib`.
- **Verificação do ambiente**: Foi verificado se o notebook estava sendo executado com **GPU** para acelerar o treinamento.

## **3. Preparação e Pré-processamento do Conjunto de Dados**
Para treinar o modelo, foi necessário preparar o conjunto de dados:
- **Download do dataset**: O Drone Shark Detection Dataset foi baixado e extraído no diretório de trabalho.
- **Estruturação das pastas**: Os arquivos foram organizados seguindo o formato esperado pelo YOLO:
  - `train/images/` → Imagens para treinamento
  - `train/labels/` → Anotações no formato YOLO
  - `val/images/` → Imagens para validação
  - `val/labels/` → Anotações para validação
- **Conversão das anotações**: Como o YOLO exige anotações no formato `txt` contendo a classe e as coordenadas do bounding box normalizadas, os arquivos foram convertidos conforme necessário.

## **4. Treinamento do Modelo YOLO**
O treinamento foi realizado usando o framework **Ultralytics YOLO**, seguindo os passos abaixo:
- **Definição dos hiperparâmetros**:
  - Modelo: `YOLOv8n.pt` (versão nano para melhor desempenho em tempo real)
  - Épocas: 50
  - Tamanho do batch: 16
  - Taxa de aprendizado ajustada para evitar overfitting
- **Execução do treinamento**:
  - O modelo foi treinado utilizando os dados de treinamento.
  - Durante o processo, métricas como **perda de classificação, perda de bounding box e precisão mAP** foram monitoradas.
  - A cada época, os resultados foram armazenados para análise posterior.

## **5. Avaliação do Modelo**
Após o treinamento, o modelo foi avaliado utilizando o conjunto de validação. Os seguintes testes foram realizados:
- **Cálculo do mAP (mean Average Precision)** para avaliar a precisão da detecção dos tubarões.
- **Análise das métricas**:
  - **Precisão (Precision)**: Indica a porcentagem de detecções corretas.
  - **Recall**: Mede a capacidade do modelo de encontrar todos os objetos relevantes.
  - **F1-score**: Combinação da precisão e recall para avaliar o desempenho geral.
- **Comparação entre diferentes versões do YOLO**:
  - Foram realizados testes com YOLOv5 e YOLOv8 para comparar qual apresentava melhores resultados.

## **6. Inferência e Testes**
Após o treinamento e avaliação, o modelo foi aplicado para detectar tubarões em imagens e vídeos inéditos. Os seguintes experimentos foram feitos:
- **Teste com imagens individuais**: Foram carregadas imagens do conjunto de testes para visualizar a performance do modelo.
- **Inferência em vídeos**: O modelo foi aplicado em vídeos capturados por drones para detectar tubarões em tempo real.
- **Ajuste dos parâmetros de detecção**: Foram ajustados limiares de confiança e supressão de não-máximos para melhorar os resultados.

## **7. Conclusão**
O treinamento e teste do modelo YOLO para detecção de tubarões apresentou bons resultados, destacando-se os seguintes pontos:
- **YOLOv8** demonstrou um melhor desempenho em relação ao YOLOv5, com maior precisão e menor tempo de inferência.
- O modelo conseguiu detectar tubarões em imagens aéreas, mesmo em condições desafiadoras como variação de iluminação e reflexos na água.


---

