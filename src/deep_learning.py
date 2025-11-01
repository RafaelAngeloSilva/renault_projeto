"""
Algoritmo de Deep Learning para detecção de defeitos em carrocerias.
Utiliza uma rede neural convolucional (CNN) para classificação de imagens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from typing import List, Tuple, Optional
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class CNNDefeitoDetector(nn.Module):
    """
    Rede neural convolucional para detecção de defeitos em carrocerias.
    Baseada em uma arquitetura CNN simples mas eficaz.
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Inicializa a rede neural.
        
        Args:
            num_classes (int): Número de classes (2: defeito/ok)
        """
        super(CNNDefeitoDetector, self).__init__()
        
        # Primeira camada convolucional
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Segunda camada convolucional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Terceira camada convolucional
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Quarta camada convolucional
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Camadas totalmente conectadas
        # Tamanho da feature map após 4 pools: 224/2^4 = 14x14
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass da rede.
        
        Args:
            x: Tensor de entrada (batch_size, 3, 224, 224)
            
        Returns:
            Tensor de saída (batch_size, num_classes)
        """
        # Bloco 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Bloco 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Bloco 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Bloco 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class AlgoritmoDeepLearning:
    """
    Algoritmo de Deep Learning para detecção de defeitos em carrocerias.
    Utiliza transfer learning ou treina uma CNN do zero.
    """
    
    def __init__(self, usar_transfer_learning: bool = True):
        """
        Inicializa o algoritmo de Deep Learning.
        
        Args:
            usar_transfer_learning (bool): Se True, usa ResNet pré-treinado
        """
        self.usar_transfer_learning = usar_transfer_learning
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ["ok", "defeito"]
        
        # Transformações para as imagens
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Carrega o modelo
        self.model = self._carregar_modelo()
        self.model.eval()  # Modo de inferência
        
    def _carregar_modelo(self):
        """
        Carrega o modelo de Deep Learning.
        Se não existir um modelo treinado, usa pesos aleatórios.
        
        Returns:
            Modelo carregado
        """
        if self.usar_transfer_learning:
            # Usa ResNet18 pré-treinado
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)  # 2 classes
        else:
            # Usa CNN customizada
            model = CNNDefeitoDetector(num_classes=2)
        
        model = model.to(self.device)
        
        # Tenta carregar pesos se existirem
        caminho_modelo = "modelos_deep_learning/modelo_treinado.pth"
        if os.path.exists(caminho_modelo):
            try:
                model.load_state_dict(torch.load(caminho_modelo, map_location=self.device))
                print(f"[DL] Modelo treinado carregado de {caminho_modelo}")
            except Exception as e:
                print(f"[DL] Aviso: Não foi possível carregar modelo existente: {e}")
                print("[DL] Usando pesos aleatórios (modelo não treinado)")
        else:
            print("[DL] Aviso: Nenhum modelo treinado encontrado. Usando pesos aleatórios.")
            print("[DL] O modelo pode não fornecer predições precisas.")
        
        return model
    
    def _preprocessar_imagem(self, imagem: np.ndarray) -> torch.Tensor:
        """
        Pré-processa uma imagem para o modelo.
        
        Args:
            imagem: Array numpy com a imagem (BGR do OpenCV)
            
        Returns:
            Tensor pré-processado
        """
        # Converte BGR para RGB
        if len(imagem.shape) == 3 and imagem.shape[2] == 3:
            imagem_rgb = imagem[:, :, ::-1]  # BGR -> RGB
        else:
            # Se já estiver em escala de cinza, converte para RGB
            imagem_rgb = np.stack([imagem] * 3, axis=2)
        
        # Converte para PIL Image
        pil_image = Image.fromarray(imagem_rgb)
        
        # Aplica transformações
        tensor = self.transform(pil_image)
        
        # Adiciona dimensão de batch
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def prever(self, imagens: List[np.ndarray]) -> List[str]:
        """
        Classifica uma lista de imagens usando Deep Learning.
        
        Args:
            imagens (List[np.ndarray]): Lista de imagens para classificar
            
        Returns:
            List[str]: Lista de previsões ("defeito" ou "ok")
        """
        previsoes = []
        
        for imagem in imagens:
            previsao = self.prever_uma(imagem)
            previsoes.append(previsao)
        
        return previsoes
    
    def prever_uma(self, imagem: np.ndarray) -> str:
        """
        Classifica uma única imagem usando Deep Learning.
        
        Args:
            imagem (np.ndarray): Imagem para classificar
            
        Returns:
            str: Previsão ("defeito" ou "ok")
        """
        try:
            # Pré-processa a imagem
            tensor = self._preprocessar_imagem(imagem)
            
            # Faz a inferência
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Converte predição para string
                indice_classe = predicted.item()
                predicao = self.classes[indice_classe]
                
            return predicao
            
        except Exception as e:
            print(f"[DL] Erro na predição: {e}")
            # Em caso de erro, retorna classe aleatória como fallback
            import random
            return random.choice(self.classes)
    
    def prever_com_confianca(self, imagem: np.ndarray) -> Tuple[str, float]:
        """
        Classifica uma imagem e retorna a confiança.
        
        Args:
            imagem (np.ndarray): Imagem para classificar
            
        Returns:
            Tuple[str, float]: (previsão, confiança)
        """
        try:
            # Pré-processa a imagem
            tensor = self._preprocessar_imagem(imagem)
            
            # Faz a inferência
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                confianca, predicted = torch.max(probabilities, 1)
                
                # Converte predição para string
                indice_classe = predicted.item()
                predicao = self.classes[indice_classe]
                confianca_valor = confianca.item()
                
            return predicao, confianca_valor
            
        except Exception as e:
            print(f"[DL] Erro na predição: {e}")
            import random
            return random.choice(self.classes), 0.5
    
    def obter_caracteristicas_detalhadas(self, imagem: np.ndarray) -> dict:
        """
        Retorna características detalhadas da predição.
        
        Args:
            imagem (np.ndarray): Imagem para analisar
            
        Returns:
            dict: Dicionário com informações detalhadas
        """
        predicao, confianca = self.prever_com_confianca(imagem)
        
        return {
            "predicao": predicao,
            "confianca": confianca,
            "usar_transfer_learning": self.usar_transfer_learning,
            "device": str(self.device)
        }


def treinar_modelo(dataset_dir: str, epochs: int = 10, batch_size: int = 32):
    """
    Função para treinar o modelo de Deep Learning.
    
    Args:
        dataset_dir: Diretório com pastas 'defeito' e 'ok'
        epochs: Número de épocas
        batch_size: Tamanho do batch
    """
    print("Esta funcionalidade de treinamento precisa de implementação completa.")
    print("Para usar um modelo pré-treinado, coloque o arquivo .pth em:")
    print("  modelos_deep_learning/modelo_treinado.pth")
    print("\nO sistema funcionará com pesos aleatórios até que um modelo seja fornecido.")


if __name__ == "__main__":
    # Teste simples do algoritmo
    print("Algoritmo de Deep Learning para Detecção de Defeitos")
    print("=" * 60)
    
    # Cria uma imagem de teste
    import cv2
    imagem_teste = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Testa o algoritmo
    algo_dl = AlgoritmoDeepLearning(usar_transfer_learning=True)
    predicao = algo_dl.prever_uma(imagem_teste)
    caracteristicas = algo_dl.obter_caracteristicas_detalhadas(imagem_teste)
    
    print(f"Predição: {predicao}")
    print(f"Características: {caracteristicas}")
    print("\nAlgoritmo inicializado com sucesso!")

