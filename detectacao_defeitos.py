"""
Esteira Scan - Sistema de Detecção de Defeitos
Sistema automatizado para análise de imagens de carrocerias
"""

import os
import cv2
import numpy as np
import sys

# Adiciona o diretório src ao path para importar os módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from algoritmos_deteccao import AlgoritmoAleatorio, AlgoritmoHeuristico

# Importa Deep Learning se disponível
try:
    from deep_learning import AlgoritmoDeepLearning
    DEEP_LEARNING_DISPONIVEL = True
except ImportError:
    print("[AVISO] Deep Learning não disponível. Instale torch e torchvision.")
    DEEP_LEARNING_DISPONIVEL = False


class SistemaDetecaoDefeitos:
    """
    Sistema principal para detecção de defeitos em imagens de carrocerias.
    """
    
    def __init__(self):
        self.pasta_imagens = "imagens_para_analisar"
        self.algo_aleatorio = AlgoritmoAleatorio(seed=42)
        self.algo_heuristico = AlgoritmoHeuristico()
        
        # Inicializa Deep Learning se disponível
        if DEEP_LEARNING_DISPONIVEL:
            try:
                self.algo_deep_learning = AlgoritmoDeepLearning(usar_transfer_learning=True)
                print("[DL] Algoritmo de Deep Learning inicializado com sucesso!")
            except Exception as e:
                print(f"[DL] Erro ao inicializar Deep Learning: {e}")
                self.algo_deep_learning = None
        else:
            self.algo_deep_learning = None
        
        # Cria pasta se não existir
        os.makedirs(self.pasta_imagens, exist_ok=True)
        
        # Cria pasta para modelos de Deep Learning
        os.makedirs("modelos_deep_learning", exist_ok=True)
        
        # Extensões de imagem suportadas
        self.extensoes_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def listar_imagens_disponiveis(self):
        """Lista todas as imagens disponíveis na pasta."""
        imagens = []
        
        if not os.path.exists(self.pasta_imagens):
            return imagens
        
        for arquivo in os.listdir(self.pasta_imagens):
            if any(arquivo.lower().endswith(ext) for ext in self.extensoes_validas):
                caminho_completo = os.path.join(self.pasta_imagens, arquivo)
                imagens.append(caminho_completo)
        
        return sorted(imagens)
    
    def analisar_imagem(self, caminho_imagem):
        """Analisa uma imagem específica."""
        try:
            # Carrega a imagem
            imagem = cv2.imread(caminho_imagem)
            if imagem is None:
                return None, "Erro ao carregar imagem"
            
            # Redimensiona para os algoritmos
            imagem_redim = cv2.resize(imagem, (224, 224))
            
            # Executa análises
            pred_aleatorio = self.algo_aleatorio.prever_uma(imagem_redim)
            pred_heuristico = self.algo_heuristico.prever_uma(imagem_redim)
            
            # Executa análise de Deep Learning se disponível
            pred_deep_learning = None
            caracteristicas_dl = None
            if self.algo_deep_learning is not None:
                try:
                    pred_deep_learning, confianca_dl = self.algo_deep_learning.prever_com_confianca(imagem_redim)
                    caracteristicas_dl = self.algo_deep_learning.obter_caracteristicas_detalhadas(imagem_redim)
                except Exception as e:
                    print(f"[DL] Erro na análise: {e}")
            
            # Obtém características detalhadas
            caracteristicas = self.algo_heuristico.obter_caracteristicas_detalhadas(imagem_redim)
            
            # Detecta regiões com defeitos para visualização
            regioes_defeito = []
            if pred_heuristico == "defeito":
                regioes_defeito = self.algo_heuristico.detectar_regioes_defeito(imagem)
            
            return {
                'arquivo': os.path.basename(caminho_imagem),
                'pred_aleatorio': pred_aleatorio,
                'pred_heuristico': pred_heuristico,
                'pred_deep_learning': pred_deep_learning,
                'caracteristicas': caracteristicas,
                'caracteristicas_dl': caracteristicas_dl,
                'dimensoes': imagem.shape,
                'imagem': imagem,
                'regioes_defeito': regioes_defeito
            }, None
            
        except Exception as e:
            return None, str(e)
    
    def mostrar_resultado(self, resultado):
        """Exibe o resultado da análise de forma organizada."""
        print(f"\n{'='*60}")
        print(f"ANALISE: {resultado['arquivo']}")
        print(f"{'='*60}")
        
        print(f"Dimensoes: {resultado['dimensoes'][1]}x{resultado['dimensoes'][0]} pixels")
        
        print(f"\nRESULTADOS:")
        status_aleatorio = "[DEFEITO]" if resultado['pred_aleatorio'] == "defeito" else "[OK]"
        status_heuristico = "[DEFEITO]" if resultado['pred_heuristico'] == "defeito" else "[OK]"
        
        print(f"   Algoritmo Aleatorio: {status_aleatorio} {resultado['pred_aleatorio'].upper()}")
        print(f"   Algoritmo Heuristico: {status_heuristico} {resultado['pred_heuristico'].upper()}")
        
        # Mostra resultados de Deep Learning se disponível
        if resultado['pred_deep_learning'] is not None:
            status_dl = "[DEFEITO]" if resultado['pred_deep_learning'] == "defeito" else "[OK]"
            if resultado['caracteristicas_dl']:
                confianca = resultado['caracteristicas_dl']['confianca'] * 100
                print(f"   Deep Learning (CNN): {status_dl} {resultado['pred_deep_learning'].upper()} (Confiança: {confianca:.1f}%)")
            else:
                print(f"   Deep Learning (CNN): {status_dl} {resultado['pred_deep_learning'].upper()}")
        
        print(f"\nCARACTERISTICAS TECNICAS (Heuristico):")
        char = resultado['caracteristicas']
        print(f"   Intensidade de Bordas: {char['intensidade_bordas']:.2f}")
        print(f"   Variacao de Intensidade: {char['variacao_intensidade']:.3f}")
        print(f"   Threshold Bordas: {char['threshold_bordas']}")
        print(f"   Threshold Intensidade: {char['threshold_intensidade']}")
        
        # Mostra características de Deep Learning se disponível
        if resultado['caracteristicas_dl']:
            print(f"\nCARACTERISTICAS TECNICAS (Deep Learning):")
            char_dl = resultado['caracteristicas_dl']
            print(f"   Confianca: {char_dl['confianca']*100:.1f}%")
            print(f"   Modelo: {'ResNet18 Transfer Learning' if char_dl['usar_transfer_learning'] else 'CNN Customizada'}")
            print(f"   Device: {char_dl['device']}")
        
        print(f"\nINTERPRETACAO:")
        if char['intensidade_bordas'] > char['threshold_bordas']:
            print("   Bordas detectadas: Possivel defeito (riscos, arranhoes)")
        else:
            print("   Bordas baixas: Superficie lisa")
        
        if char['variacao_intensidade'] > char['threshold_intensidade']:
            print("   Variacao alta: Superficie irregular")
        else:
            print("   Variacao baixa: Superficie uniforme")
        
        print(f"\nRECOMENDACAO:")
        # Considera todos os algoritmos na recomendação
        votos_defeito = 0
        if resultado['pred_aleatorio'] == "defeito":
            votos_defeito += 1
        if resultado['pred_heuristico'] == "defeito":
            votos_defeito += 1
        if resultado['pred_deep_learning'] == "defeito":
            votos_defeito += 1
        
        num_algoritmos = 2 + (1 if resultado['pred_deep_learning'] is not None else 0)
        
        if votos_defeito >= num_algoritmos / 2:
            print("   ATENCAO: Defeito detectado por maioria dos algoritmos! Verificar manualmente.")
        else:
            print("   OK: Nenhum defeito detectado pela maioria dos algoritmos.")
        
        # Mostra visualização se houver defeito detectado
        if resultado.get('regioes_defeito') and len(resultado['regioes_defeito']) > 0:
            self._mostrar_visualizacao(resultado)
    
    def _mostrar_visualizacao(self, resultado):
        """Mostra a visualização da imagem com regiões de defeito marcadas."""
        imagem = resultado['imagem'].copy()
        regioes = resultado['regioes_defeito']
        
        # Desenha retângulos nas regiões de defeito
        for i, (x, y, w, h) in enumerate(regioes, 1):
            # Cor vermelha para defeitos
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Adiciona número da região
            cv2.putText(imagem, f"Defeito {i}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Salva a imagem com visualização
        nome_arquivo = resultado['arquivo']
        nome_base = os.path.splitext(nome_arquivo)[0]
        imagem_resultado = os.path.join("imagens_para_analisar", f"{nome_base}_resultado.jpg")
        cv2.imwrite(imagem_resultado, imagem)
        
        print(f"\nVISUALIZACAO:")
        print(f"   Imagem com defeitos marcados salva em: {imagem_resultado}")
        print(f"   Total de regioes com defeito detectadas: {len(regioes)}")
        
        # Pergunta se quer ver a imagem
        try:
            mostrar = input("   Deseja visualizar a imagem agora? (s/n): ").strip().lower()
            if mostrar == 's':
                # Tenta abrir a imagem
                if os.name == 'nt':  # Windows
                    os.startfile(imagem_resultado)
                elif os.name == 'posix':  # Linux/Mac
                    os.system(f'xdg-open {imagem_resultado}')
        except:
            pass  # Se não conseguir, continua
    
    def executar_sistema(self):
        """Executa o sistema principal."""
        print("ESTEIRA SCAN - Sistema de Deteccao de Defeitos")
        print("=" * 50)
        print(f"Pasta de imagens: {self.pasta_imagens}")
        print("\nINSTRUCOES:")
        print("1. Copie suas fotos de carros para a pasta 'imagens_para_analisar'")
        print("2. Execute este programa")
        print("3. Escolha qual imagem analisar")
        print("4. Veja os resultados!")
        
        while True:
            print(f"\n{'='*50}")
            
            # Lista imagens disponíveis
            imagens = self.listar_imagens_disponiveis()
            
            if not imagens:
                print(f"\nNenhuma imagem encontrada na pasta '{self.pasta_imagens}'")
                print("Copie suas fotos de carros para esta pasta e execute novamente!")
                
                input("\nPressione Enter para verificar novamente...")
                continue
            
            print(f"\nIMAGENS DISPONIVEIS ({len(imagens)}):")
            for i, caminho in enumerate(imagens, 1):
                nome = os.path.basename(caminho)
                print(f"   {i}. {nome}")
            
            print(f"\nOpcoes:")
            print(f"   0. Verificar novamente")
            print(f"   a. Analisar TODAS as imagens")
            print(f"   s. Sair")
            
            escolha = input(f"\nEscolha uma opcao (1-{len(imagens)}, 0, a, s): ").strip().lower()
            
            if escolha == 's':
                print("Encerrando sistema...")
                break
            elif escolha == '0':
                continue
            elif escolha == 'a':
                # Analisa todas as imagens
                print(f"\nAnalisando TODAS as {len(imagens)} imagens...")
                for caminho in imagens:
                    resultado, erro = self.analisar_imagem(caminho)
                    if erro:
                        print(f"Erro em {os.path.basename(caminho)}: {erro}")
                    else:
                        self.mostrar_resultado(resultado)
                
                input("\nPressione Enter para continuar...")
            else:
                try:
                    indice = int(escolha) - 1
                    if 0 <= indice < len(imagens):
                        caminho = imagens[indice]
                        print(f"\nAnalisando {os.path.basename(caminho)}...")
                        
                        resultado, erro = self.analisar_imagem(caminho)
                        if erro:
                            print(f"Erro: {erro}")
                        else:
                            self.mostrar_resultado(resultado)
                        
                        input("\nPressione Enter para continuar...")
                    else:
                        print("Opcao invalida!")
                except ValueError:
                    print("Digite um numero valido!")


def main():
    """Funcao principal."""
    sistema = SistemaDetecaoDefeitos()
    sistema.executar_sistema()


if __name__ == "__main__":
    main()