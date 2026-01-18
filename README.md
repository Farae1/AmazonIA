# APS - Sistema de Análise de Desmatamento (U-Net + PyTorch)
Este projeto implementa um sistema de segmentação semântica para detecção de áreas desmatadas em imagens de satélite (GeoTIFF) utilizando a arquitetura U-Net em PyTorch. O sistema é otimizado para processar imagens multiespectrais de 4 bandas (RGB + NIR).

# Fluxo de pipeline

O script main.py executa o fluxo completo do projeto:

-**Divisão do Dataset**: Os dados são divididos automaticamente em 60% para treino, 20% para validação e 20% para teste.

-**Treinamento**: Utiliza o otimizador Adam com taxa de aprendizado de 1×10−4 e a função de perda CrossEntropyLoss.

-**Avaliação de Teste**: Após o treinamento (ou carregamento de pesos), o modelo é testado em dados inéditos, calculando Acurácia média, IoU e Dice.

-**Visualização**: O sistema seleciona uma imagem aleatória do conjunto de teste e exibe o comparativo entre os canais RGB, a máscara real e a predição gerada.

# Instalação e utilização
Clone o repositório na sua máquina e o acesse:

    https://github.com/Farae1/AmazonIA.git
    cd AmazonIA

Após isso, para testar a geração de uma mask à partir de uma imagem aleatória GeoTIFF:

    python main.py

# Parte teórica
No repositorio, também estará a pesquisa teórica realizada sobre modelos clássicos de Machine Learning, Redes Convulocionais e outros tipos de modelos de Deep Learning
