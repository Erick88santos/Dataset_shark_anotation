import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore

# Importar as bibliotecas necessárias: destaque para pytorch
import numpy as np
import PIL.Image
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from matplotlib import pyplot as plt
import os

# Tamanho da imagem
image_size = 100

# Transformando as imagens: para modelos mais robustos, é recomendável fazer data augmentation!
transformacoes_de_imagens = { 
    'images': transforms.Compose([
        transforms.Resize(size=[image_size, image_size]),
        transforms.ToTensor(),
    ])
}

# Determinar o caminho do dataset e das imagens
dataset = './dataset/'
pasta_images = os.path.join(dataset, 'images')

import os

# Determinar o caminho do dataset e das imagens
dataset = r'C:\Users\erick\OneDrive\Documentos\IFPE\dataset'
pasta_images = os.path.join(dataset, 'images')

# Verificar se a pasta existe ou criá-la se não existir
if not os.path.exists(pasta_images):
    os.makedirs(pasta_images)
    print(f"A pasta '{pasta_images}' foi criada.")
else:
    print(f"A pasta '{pasta_images}' foi encontrada.")

# Continue com o resto do código aqui...


# Tamanho do batch de treinamento
bs = 8



import os

# Caminho para a pasta de imagens
dataset = r'C:\Users\erick\OneDrive\Documentos\IFPE\dataset'
pasta_images = os.path.join(dataset, 'images')

# Verificar se há subpastas de classes dentro da pasta 'images'
subpastas = [d for d in os.listdir(pasta_images) if os.path.isdir(os.path.join(pasta_images, d))]
if not subpastas:
    raise FileNotFoundError(f"Não foram encontradas subpastas de classes em '{pasta_images}'. Verifique a estrutura de pastas.")

# Continue com o carregamento usando ImageFolder
data = {
    'images': datasets.ImageFolder(root=pasta_images, transform=transformacoes_de_imagens['images'])
}


# Criar DataLoaders
data_loader_treino = DataLoader(data['images'], batch_size=bs, shuffle=True)
data_loader_validacao = DataLoader(data['images'], batch_size=bs, shuffle=True)

# Mapear os índices com os nomes das classes
indice_para_classe = {v: k for k, v in data['images'].class_to_idx.items()}
print("Classes:", indice_para_classe)

# Quantidade de imagens para treino e validação
num_imagens_treino = len(data['images'])
num_imagens_validacao = len(data['images'])

# Configurar o modelo AlexNet pré-treinado
alexnet = models.alexnet(pretrained=True)

# Congelar os parâmetros da rede pré-treinada
for param in alexnet.parameters():
    param.requires_grad = False

# Modificar a última camada para ajustar o número de classes
numero_de_classes = len(indice_para_classe)
alexnet.classifier[6] = nn.Linear(4096, numero_de_classes)
alexnet.classifier.add_module("7", nn.LogSoftmax(dim=1))

# Definir a função de erro e o otimizador
funcao_erro = nn.NLLLoss()
otimizador = optim.Adam(alexnet.parameters())

# Verificar dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}\n")
alexnet.to(device)

# Função para treinar e validar o modelo
def treinar_e_validar(modelo, metrica_erro, otimizador_sgd, epocas=25):
    historico = []
    melhor_acuracia = 0.0
    melhor_modelo = None

    for epoca in range(epocas):
        inicio_epoca = time.time()
        print(f"\n\nÉpoca: {epoca+1}/{epocas}")
        
        modelo.train()
        erro_treino, acuracia_treino = 0.0, 0.0
        erro_validacao, acuracia_validacao = 0.0, 0.0

        for entradas, labels in data_loader_treino:
            entradas, labels = entradas.to(device), labels.to(device)
            otimizador_sgd.zero_grad()
            saidas = modelo(entradas)
            erro = metrica_erro(saidas, labels)
            erro.backward()
            otimizador_sgd.step()
            erro_treino += erro.item() * entradas.size(0)
            _, preds = torch.max(saidas, 1)
            acuracia_treino += torch.sum(preds == labels.data)

        modelo.eval()
        with torch.no_grad():
            for entradas, labels in data_loader_validacao:
                entradas, labels = entradas.to(device), labels.to(device)
                saidas = modelo(entradas)
                erro = metrica_erro(saidas, labels)
                erro_validacao += erro.item() * entradas.size(0)
                _, preds = torch.max(saidas, 1)
                acuracia_validacao += torch.sum(preds == labels.data)

        erro_medio_treino = erro_treino / num_imagens_treino
        acuracia_media_treino = acuracia_treino.double() / num_imagens_treino
        erro_medio_validacao = erro_validacao / num_imagens_validacao
        acuracia_media_validacao = acuracia_validacao.double() / num_imagens_validacao

        historico.append([erro_medio_treino, erro_medio_validacao, acuracia_media_treino, acuracia_media_validacao])

        fim_epoca = time.time()
        print(f"Época : {epoca+1}, Treino: Erro: {erro_medio_treino:.4f}, Acurácia: {acuracia_media_treino*100:.4f}%, "
              f"Validação: Erro: {erro_medio_validacao:.4f}, Acurácia: {acuracia_media_validacao*100:.4f}%, Tempo: {fim_epoca-inicio_epoca:.4f}s")

        if acuracia_media_validacao > melhor_acuracia:
            melhor_acuracia = acuracia_media_validacao
            torch.save(modelo, './modelos/melhor_modelo.pt')
            melhor_modelo = modelo
            
    return melhor_modelo, historico

# Treinar o modelo
numero_de_epocas = 20
modelo_treinado, historico = treinar_e_validar(alexnet, funcao_erro, otimizador, numero_de_epocas)

# Plotar o histórico de treino e validação
historico = np.array(historico)
plt.plot(historico[:, 0], label='Erro Treino')
plt.plot(historico[:, 1], label='Erro Validação')
plt.xlabel('Drone Image of Shark ')
plt.ylabel('Erro')
plt.legend()
plt.ylim(0, 1)
plt.show()


import torch
import PIL.Image
import matplotlib.pyplot as plt

def predicao_image_shark(modelo, arquivo_imagem_teste, transformacao, image_size, numero_de_classes, indice_para_classe):
    '''
    Função para realizar a predição do status do AR
    Parâmetros:
        :param modelo: modelo para testar
        :param arquivo_imagem_teste: caminho para imagem de teste
        :param transformacao: transformação aplicada à imagem de teste
        :param image_size: tamanho da imagem que o modelo espera
        :param numero_de_classes: número de classes de saída do modelo
        :param indice_para_classe: dicionário para mapear índices aos nomes das classes
    '''
    
    imagem_teste = PIL.Image.open(arquivo_imagem_teste).convert('RGB')
    plt.imshow(imagem_teste)
    plt.axis('off')
    plt.show()
    
    # Aplicar transformação e ajustar dimensões do tensor
    tensor_imagem_teste = transformacao(imagem_teste).unsqueeze(0)  # Adiciona dimensão para batch

    # Enviar imagem para GPU, se disponível
    tensor_imagem_teste = tensor_imagem_teste.to(device)
    
    dict_predicoes = {}
    
    # Desabilitar cálculo de gradiente
    with torch.no_grad():
        modelo.eval()
        
        # Fazer a previsão
        saida = modelo(tensor_imagem_teste)
        
        # Transformar a saída log-softmax em probabilidade
        ps = torch.exp(saida)
        
        # Obter as top-k classes e suas probabilidades
        topk, topclass = ps.topk(numero_de_classes, dim=1)
        for i in range(numero_de_classes):
            classe_predita = indice_para_classe[topclass.cpu().numpy()[0][i]]
            probabilidade = topk.cpu().numpy()[0][i]
            dict_predicoes[classe_predita] = probabilidade
    
    return dict_predicoes
