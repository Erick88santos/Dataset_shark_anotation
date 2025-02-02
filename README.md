# Dataset_shark_anotation
Classificação Dataset </br>
  👋


<div  alt="linguagens"style="display: inline_block"></br>

 <img align="center" alt="GITHUB" src="https://img.shields.io/badge/GITHUB-800080?style=for-the-badge&logo=github&logoColor=white"/>

<img align="center" alt="PYTHON" src="https://img.shields.io/badge/PYTHON-1a4a3f?style=for-the-badge&logo=python&logoColor=white"/>
  
</div></br>


### Resumo da Análise do Código

O código realiza **treinamento e validação de um modelo de aprendizado de máquina (AlexNet)** para classificar imagens em diferentes categorias. As etapas principais incluem:

1. **Preparação do Dataset**: 
   - As imagens são carregadas e transformadas para um tamanho padrão (100x100 pixels) e convertidas em tensores para serem processadas pelo modelo.

2. **Configuração do Modelo**: 
   - Utiliza o modelo AlexNet pré-treinado, ajustando sua camada de saída para classificar o número de classes presente no dataset.

3. **Treinamento com Early Stopping**: 
   - O modelo é treinado por até 20 épocas, enquanto monitora a perda no conjunto de validação. Caso não haja melhoria na perda por 5 épocas consecutivas, o treinamento é interrompido para evitar overfitting.

4. **Métricas de Desempenho**:
   - Após cada época, são calculados:
     - *Loss* (Erro) de treinamento e validação.
     - *Accuracy* (Acurácia) de treinamento e validação.

5. **Visualização de Resultados**:
   - Um gráfico exibe as curvas de erro e acurácia para avaliar o desempenho do modelo ao longo das épocas.

6. **Predição em Imagem Única**:
   - O modelo é usado para prever a classe de uma imagem individual, exibindo o resultado com o nível de confiança.

**Objetivo Final**: Treinar e validar um modelo eficiente de classificação de imagens, otimizando seu desempenho com Early Stopping para evitar treinamento excessivo e melhorar a generalização.
