"""
Esteira Scan - Algoritmos de Detecção de Defeitos
Implementação 1 - Projeto

Este módulo contém os algoritmos básicos para detecção de defeitos em carrocerias:
1. Algoritmo Aleatório (baseline)
2. Algoritmo Heurístico (baseado em características visuais)
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Union


class AlgoritmoAleatorio:
    """
    Algoritmo baseline que gera previsões aleatórias.
    Serve como referência de desempenho mínimo.
    """
    
    def __init__(self, seed: int = 42):
        """
        Inicializa o algoritmo aleatório com uma seed para reprodutibilidade.
        
        Args:
            seed (int): Seed para o gerador de números aleatórios
        """
        random.seed(seed)
        self.classes = ["defeito", "ok"]
    
    def prever(self, imagens: List[np.ndarray]) -> List[str]:
        """
        Gera previsões aleatórias para uma lista de imagens.
        
        Args:
            imagens (List[np.ndarray]): Lista de imagens para classificar
            
        Returns:
            List[str]: Lista de previsões ("defeito" ou "ok")
        """
        return [random.choice(self.classes) for _ in imagens]
    
    def prever_uma(self, imagem: np.ndarray) -> str:
        """
        Gera uma previsão aleatória para uma única imagem.
        
        Args:
            imagem (np.ndarray): Imagem para classificar
            
        Returns:
            str: Previsão ("defeito" ou "ok")
        """
        return random.choice(self.classes)


class AlgoritmoHeuristico:
    """
    Algoritmo heurístico baseado em características visuais simples.
    Utiliza detecção de bordas e análise de intensidade para identificar defeitos.
    """
    
    def __init__(self, threshold_bordas: float = 30.0, threshold_intensidade: float = 0.3):
        """
        Inicializa o algoritmo heurístico com parâmetros configuráveis.
        
        Args:
            threshold_bordas (float): Limiar para detecção de bordas
            threshold_intensidade (float): Limiar para análise de intensidade
        """
        self.threshold_bordas = threshold_bordas
        self.threshold_intensidade = threshold_intensidade
    
    def _extrair_caracteristicas(self, imagem: np.ndarray) -> Tuple[float, float]:
        """
        Extrai características visuais da imagem.
        
        Args:
            imagem (np.ndarray): Imagem de entrada
            
        Returns:
            Tuple[float, float]: (intensidade_bordas, variacao_intensidade)
        """
        # Converte para escala de cinza
        if len(imagem.shape) == 3:
            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        else:
            gray = imagem
        
        # Detecta bordas usando Canny
        bordas = cv2.Canny(gray, 50, 150)
        intensidade_bordas = bordas.mean()
        
        # Calcula variação de intensidade
        variacao_intensidade = gray.std() / 255.0
        
        return intensidade_bordas, variacao_intensidade
    
    def prever(self, imagens: List[np.ndarray]) -> List[str]:
        """
        Classifica uma lista de imagens usando heurísticas visuais.
        
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
        Classifica uma única imagem usando heurísticas visuais.
        
        Args:
            imagem (np.ndarray): Imagem para classificar
            
        Returns:
            str: Previsão ("defeito" ou "ok")
        """
        intensidade_bordas, variacao_intensidade = self._extrair_caracteristicas(imagem)
        
        # Lógica heurística: defeitos tendem a ter mais bordas e variação de intensidade
        if (intensidade_bordas > self.threshold_bordas or 
            variacao_intensidade > self.threshold_intensidade):
            return "defeito"
        else:
            return "ok"
    
    def obter_caracteristicas_detalhadas(self, imagem: np.ndarray) -> dict:
        """
        Retorna características detalhadas da imagem para análise.
        
        Args:
            imagem (np.ndarray): Imagem para analisar
            
        Returns:
            dict: Dicionário com características extraídas
        """
        intensidade_bordas, variacao_intensidade = self._extrair_caracteristicas(imagem)
        
        return {
            "intensidade_bordas": intensidade_bordas,
            "variacao_intensidade": variacao_intensidade,
            "threshold_bordas": self.threshold_bordas,
            "threshold_intensidade": self.threshold_intensidade,
            "previsao": self.prever_uma(imagem)
        }
    
    def detectar_regioes_defeito(self, imagem: np.ndarray) -> list:
        """
        Detecta regiões com possíveis defeitos usando análise de bordas.
        
        Args:
            imagem (np.ndarray): Imagem para analisar
            
        Returns:
            list: Lista de coordenadas (x, y, w, h) das regiões com defeitos
        """
        # Converte para escala de cinza
        if len(imagem.shape) == 3:
            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        else:
            gray = imagem.copy()
        
        # Detecta bordas usando Canny
        bordas = cv2.Canny(gray, 50, 150)
        
        # Encontra contornos
        contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtra contornos por tamanho e área
        regioes = []
        h, w = gray.shape
        min_area = (w * h) * 0.001  # Área mínima de 0.1% da imagem
        max_area = (w * h) * 0.5   # Área máxima de 50% da imagem
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if min_area < area < max_area:
                x, y, w_box, h_box = cv2.boundingRect(contorno)
                regioes.append((x, y, w_box, h_box))
        
        return regioes


# Este arquivo contém apenas os algoritmos de detecção de defeitos
# Para usar o sistema, execute: python sistema_simples.py
