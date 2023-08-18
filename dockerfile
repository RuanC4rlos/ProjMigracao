# Usando a imagem oficial do TensorFlow
FROM tensorflow/tensorflow:latest

# Definindo o diretório de trabalho
WORKDIR /app

# Copiando o código para o diretório de trabalho
#COPY . /app
COPY codigo.py .

# Instalando dependências do Python
RUN pip install numpy tqdm

# Instalando o scikit-learn
RUN pip install scikit-learn

# Executando o script Python
CMD ["python", "codigo.py"]
