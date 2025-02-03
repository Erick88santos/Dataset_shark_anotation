# Dataset_shark_anotation
Classificação Dataset </br>
  👋


<div  alt="linguagens"style="display: inline_block"></br>

 <img align="center" alt="GITHUB" src="https://img.shields.io/badge/GITHUB-800080?style=for-the-badge&logo=github&logoColor=white"/>

<img align="center" alt="PYTHON" src="https://img.shields.io/badge/PYTHON-1a4a3f?style=for-the-badge&logo=python&logoColor=white"/>

<img align="center" alt="LABEL-STUDIO" src="https://img.shields.io/badge/LABEL-STUDIO-he4a3f?style=for-the-badge&logo=LABEL-STUDIO&logoColor=white"/>  

</div></br>



### Resumo da Análise do Código

O código realiza **treinamento e validação de um modelo de aprendizado de máquina (AlexNet)** para classificar imagens em diferentes categorias. As etapas principais incluem:

1. **Preparação do Dataset**: 
   - As imagens são carregadas e transformadas para um tamanho padrão (100x100 pixels) e convertidas em tensores para serem processadas pelo modelo.

2. **Configuração do Modelo**: 
   - Utiliza o modelo AlexNet pré-treinado, ajustando sua camada de saída para classificar o número de classes presente no dataset.

3. **Treinamento com Early Stopping**: 
   - O modelo é treinado por até 100 épocas, enquanto monitora a perda no conjunto de validação. Caso não haja melhoria na perda por 5 épocas consecutivas, o treinamento é interrompido para evitar overfitting.

4. **Métricas de Desempenho**:
   - Após cada época, são calculados:
     - *Loss* (Erro) de treinamento e validação.
     - *Accuracy* (Acurácia) de treinamento e validação.

5. **Visualização de Resultados**:
   - Um gráfico exibe as curvas de erro e acurácia para avaliar o desempenho do modelo ao longo das épocas.

6. **Predição em Imagem Única**:
   - O modelo é usado para prever a classe de uma imagem individual, exibindo o resultado com o nível de confiança.

**Objetivo Final**: Treinar e validar um modelo eficiente de classificação de imagens, otimizando seu desempenho com Early Stopping para evitar treinamento excessivo e melhorar a generalização.

# Passo a Passo:

1- Baixe o DATASET e descompacte-o: 

[DATASET-ZIP](https://drive.google.com/drive/folders/1MwLEkfiph4amjvWXvr2tkwrvrB7n77uS?usp=sharing)

2- Crie um arquivo dentro do diretório /images com o nome ´´index-early-stopping.py´´ e escreva:

# Importe os arquivos necessários:
````
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
````

# Configurações Gerais 
````
image_size = 100
batch_size = 32
num_epochs = 100
dataset_path = r'C:\Users\yourname\path\dataset'
images_path = os.path.join(dataset_path, 'images')
````

# Transformações de Imagem
````
image_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])
````

# Verificação da Estrutura de Diretórios
````
if not os.path.exists(images_path):
    os.makedirs(images_path)
    print(f"Pasta '{images_path}' criada.")
else:
    print(f"Pasta '{images_path}' encontrada.")
````

# Carregamento do Dataset
````
if not os.listdir(images_path):
    raise FileNotFoundError(f"Não foram encontradas subpastas de classes em '{images_path}'.")
    
data = datasets.ImageFolder(root=images_path, transform=image_transforms)
dataloaders = {
    'train': DataLoader(data, batch_size=batch_size, shuffle=True),
    'test': DataLoader(data, batch_size=batch_size, shuffle=False),
}
class_to_idx = data.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
````

# Configuração do Modelo AlexNet Pré-Treinado
````
alexnet = models.alexnet(pretrained=True)
for param in alexnet.parameters():
    param.requires_grad = False
num_classes = len(class_to_idx)
alexnet.classifier[6] = nn.Linear(4096, num_classes)
alexnet.classifier.add_module("7", nn.LogSoftmax(dim=1))
````

# Configuração de Dispositivo
````
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alexnet.to(device)
````

# Função de Erro e Otimizador
````
criterion = nn.NLLLoss()
optimizer = optim.Adam(alexnet.classifier[6].parameters())
````

# Função de Treinamento e Validação com Early Stopping
````
def train_and_evaluate_with_early_stopping(model, criterion, optimizer, dataloaders, epochs, patience=10):
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_loss = float('inf')  # Inicializa a melhor perda como infinito
    patience_counter = 0  # Contador de paciência para Early Stopping
    best_model_state = None  # Armazenar o melhor estado do modelo

    for epoch in range(epochs):
        print(f"\nÉpoca {epoch + 1}/{epochs}")
        start_time = time.time()
````        
# Treinamento
````
        model.train()
        train_loss, train_corrects = 0.0, 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
````        

 # Validação/Teste
 ````        
        model.eval()
        test_loss, test_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)
````        

# Resultados da Época
````       
        train_loss = train_loss / len(dataloaders['train'].dataset)
        train_acc = train_corrects.double() / len(dataloaders['train'].dataset)
        test_loss = test_loss / len(dataloaders['test'].dataset)
        test_acc = test_corrects.double() / len(dataloaders['test'].dataset)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc.item())
        
        epoch_time = time.time() - start_time
        print(f"Treino: Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Teste: Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Tempo: {epoch_time:.2f}s")
````

# Early Stopping: verifica se a perda de teste melhorou
````        
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Salva o melhor estado do modelo
        else:
            patience_counter += 1
            print(f"Sem melhoria na perda de validação por {patience_counter} época(s).")
            if patience_counter >= patience:
                print("\nTreinamento interrompido antecipadamente devido ao Early Stopping.")
                model.load_state_dict(best_model_state)  # Restaura o melhor modelo
                return model, history
````

# Restaura o melhor modelo ao final do treinamento
````
    
    model.load_state_dict(best_model_state)
    return model, history
````

# Treinamento com Early Stopping
````
trained_model, training_history = train_and_evaluate_with_early_stopping(
    alexnet, criterion, optimizer, dataloaders, num_epochs, patience=10
)
````

# Plotagem do Histórico 
````
def plot_training_history(history):
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
````

# Loss
````  
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Erro de Treino')
    plt.plot(epochs_range, history['test_loss'], label='Erro de Teste')
    plt.title('Erro')
    plt.xlabel('Épocas - Early Stopping')
    plt.ylabel('Loss')
    plt.legend()
````    

# Accuracy
````    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Treino de Acurácia')
    plt.plot(epochs_range, history['test_acc'], label='Teste de Acurácia')
    plt.title('Acurácia')
    plt.xlabel('Épocas - Early Stopping')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

plot_training_history(training_history)

````
# RESULTADO 

![Image](https://github.com/user-attachments/assets/86a2074b-7715-414b-82db-5cc3213d1b74)
