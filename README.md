# Relatório final do desafio

---
## Identificação 

**Nome:** Letícia Maria dos Santos Dias  
**Curso:** Engenharia de Software  
**Instituição:** Universidade Federal do Cariri (UFCA)  
**Categoria:** Discente

---

## 📌 Navegação Rápida

- 📂 [Resumo da Arquitetura do Modelo](#1-resumo-da-arquitetura-do-modelo)
- ⚙ [Bibliotecas Utilizadas](#2-bibliotecas-utilizadas)
  - 📚 [Tensorflow](#tensorflow)
  - 📚 [os](#os)
- 💻 [Técnica de Otimização do Modelo](#3-técnica-de-otimização-do-modelo)
  - 🎯 [Quantização de Faixa Dinâmica](#-quantização-de-faixa-dinâmica)
- 📤 [Resultados Obtidos](#4-resultados-obtidos)
- 📝 [Comentários Adicionais](#5-comentários-adicionais)

---

## 1️⃣ Resumo da Arquitetura do Modelo

Nesse modelo utilizei uma arquitetura de CNN simples, inspirada no modelo VGG, com uso de filtros 3x3, camadas de pooling e camadas densas para classificação. 
Foram utilizadas duas camadas convolucionais e duas camadas densas, além do max pooling, conforme solicitado.

A primeira camada convolucional é expressa por:

```python
layers.Conv2D(16, (3,3), activation='relu')
```
Nessa camada, foram utilizados 16 filtros diferentes, cada um analiza uma matriz 3x3 gerando 16 mapas de características. Com esse filtro o modelo aprende a identificar caracteríscas presentes nas bordas mais próximas dos dados de entrada.

Já a segunda camada convolucional já faz uma análise mais profunda sendo representada por:

```python
layers.Conv2D(32, (3,3), activation='relu')
```
Nessa etapa, 32 filtros foram utilizados e os dados de entrada foram os resultados da camada anterior, combinando os padrôes da primeira camada para produzir padrôes mais sofisticados identificando melhor curvas, linhas e detalhes mais singulares de cada um dos algarismos.

Além disso, fiz o emprego do max pooling 2x2 com o intuito de extrair caracteristicas mais relevantes em cada uma das camadas convulucionais.

```python
layers.MaxPooling2D(2,2)
```
O max pooling atua no código fazendo uma redução das quantidade de pixels da imagem, que é representada por matrizes. Ao empregá-lo, nesse caso, ele reorganizou as matrizes dividindo-as em partes menores(matrizes 2x2) e extraiu o maior valor de cada um, produzindo uma nova matriz apenas com os dados maiores filtrados, reduzindo o custo computacional, pois com isso os recursos de hardware serão poupados ao processar matrizes menores enquanto os padrôes são mantidos.

Em seguida, utilizo o Flaten para converter os valores da matriz expressa em um mapa de características(pixels) em uma array de valores lineares, para preparar o modelo para ser tratado pelas camadas mais densas onde cada neurônio precisa se conectar com todos os valores de entrada.

Depois disso utilizei duas camadas densas, sendo a primeira:

```python
Dense(64, activation='relu')
```
Essa camada utiliza 64 neurônios, cada um faz uma interpretação diferente dos valores do Flaten, essa camada identifica padrôes ainda mais completos a partir da combinação dos padrôes já identificados antes ele começa a juntar os detalhes como se fosse um quebra cabeças.

A segunda camada densa realiza a distinção dos dígitos:

```python
layers.Dense(10, activation='softmax')
```

Ela utiliza 10 neunônios cada um representa um número específico de 0 a 9. Cada neurônio vai calcular um valor com base na última saída advinda da primeira camada densa e esses valores são convertidos em probabilidades. Com isso, para cada valor apresentado será atribuído o dígito com maior probabilidade. 

#### ReLU 

O ReLU foi utilizado em três momentos no train_model.py:

1. Nas camadas convolucionais, ele zera os valores negativos e mantém os positivos.
2. Na primeira camada densa ele zera as combinações que não fazem sentido. Essas combinações são definidas pelo uso de algoritmos de otimização como o Adam que também foi empregado nesse modelo.

#### Softmax

O Softmax foi utilizado na segunda camada densa, para converter os valores nas probabilidades de cada dígito, facilitando uma análise mais matemática e assertiva com base nos dados de probabilidades. 

Ele interpreta as probabilidades como nesse exemplo:

| Dígito | Probabilidade |
|--------|---------------|
| 0      | 4%            |
| 1      | 6%            |
| 2      | 60%           |

Nesse caso, o valor atribuido ao dígito seria o 2, pois ele possui a maior probabilidade de ser o dígito correto. Com base em todas as análises anteriores e algoritmos aplicados.

---
## 2️⃣ Bibliotecas Utilizadas

### Tensorflow

A biblioteca que mais utilizei foi a Tensorflow na versão 2.21.0
dando ênfase principalmente para o módulo Keras na versão 3.14.0

O Keras foi usado para:

Carregar o dataset MNIST

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```
Criar a rede neural

```python
model = keras.Sequential([...
```
Treinar, avaliar e salvar o modelo de um modo geral.
  
O módulo lite também foi utilizado na fase de otimização para a conversão do modelo para utilização em dispositivos de borda (Edge AI)
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
```
### os

A biblioteca OS do sistema operacional também foi bastante utilizada, dentre suas aplicações teve mais destaque sua utilização no salvamento dos arquivos solicitados, verificando se esses já existiam na pasta para evitar erros durante a compilação.
```python
if os.path.exists("model.tflite"):
  ```

---
## 3️⃣ Técnica de Otimização do Modelo

A técnica de otimização escolhida para o modelo foi a Quantização de Faixa Dinâmica. 

### Quantização de Faixa Dinâmica

No optimize_model.py essa Quantização de Faixa Dinâmica reduziu o modelo com a intenção de deixá-lo mais leve para dispositivo embarcado. Ao ser aplicada, ela atua convertendo os valores de ponto flutuante de 32 bits em inteiros de 8 bits.
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
 ```
 Isso faz com que o modelo ocupe menos memória, sem perder a coerência das informações. Essa redução é muito importante pois os dispositivos embarcados possuem capacidade de memória limitada e cada KB precisa ser bem aproveitado.

---
## 4️⃣ Resultados Obtidos

Na tabela a seguir listo alguns dos principais resultados obtidos com o modelo. Vale destacar que a acurácia apresentou valores variáveis em cada treinamento, mas ficando sempre numa média 98%, considerando o formato inteiro de 8 bits.

| Critério | Resultado |
|--------|---------------|
|  Acurácia final de treinamento   | 98,81%            |
| Loss final de treinamento     | 0,0389           |
| Acurácia de validação     | 98,83%       |
| Loss de validação     | 0.0431      |


---
## 5️⃣ Comentários Adicionais

### Limitações do modelo  

Algumas das limitações do seguinte modelo estão mais voltadas a sua aplicação em casos divergentes do utilizado, porque ele foi treinado apenas com o dataset MNIST e foi utilizada uma arquitetura simples com poucas camadas e apenas 5 épocas, conforme solicitado. Desse modo o modelo não apresenta robustez para lidar com uma complexidade muito superior ou imagens com ofuscamento, mas pode ser usado facilmente em Edge Ai, é leve e rápido.

### Como executar

Para executar em seu computador você deve clonar esse projeto, instalar os requirements.txt:
```powershell
pip install -r requirements.txt
```
e compilar os arquivos da seguinte forma no terminal:
```powershell
python train_model.py
>> python optimize_model.py
```
### Estrutura
```python
processoseletivoIA/
├── .github/
│   └── workflows/
│       └── ci.yml            
├── .devcontainer/            
│   └── devcontainer.json
├── .vscode/ 
│   └── settings.json
├── models/                   # 🤖 Modelos do Keras
├── train_model.py            # ✏️ Treinamento do modelo
├── optimize_model.py         # ✏️ Conversão e otimização
├── requirements.txt          # 📄 Dependências do projeto
├── model.h5                  # 🤖 Modelo treinado (gerado)
├── model.tflite              # ⚡ Modelo otimizado (gerado)
└── README.md                 # 📝 Relatório final 
```
---
