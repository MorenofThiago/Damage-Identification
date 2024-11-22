# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:18:26 2024

@author: User
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from scipy.io import loadmat
import random
from sklearn.model_selection import train_test_split
import keras_tuner as kt  # Atualização para keras_tuner
import os
import shutil

# Limpa o diretório do tuner
tuner_directory = 'my_dir'
if os.path.exists(tuner_directory):
    shutil.rmtree(tuner_directory)

# Importa os dados
PosSensor = 'VG'
Vagao = 'PrimVag'

# Carregar os dados
DadosAll = loadmat(f'Data04-08_{PosSensor}_{Vagao}_Cut.mat') 
dataBaseline = DadosAll['Baseline']               # Sem dano
dataCincoP = DadosAll['CincoP']                   # 5% de dano
dataDezP = DadosAll['DezP']                        # 10% de dano
dataVinteP = DadosAll['VinteP']                    # 20% de dano

# Selecionar 100 dados para treinamento de cada cenário
train_samples = 100
test_samples = int(train_samples)

# Normalização dos dados
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# Aplicar normalização aos dados
dataBaseline = normalize_data(dataBaseline)
dataCincoP = normalize_data(dataCincoP)
dataDezP = normalize_data(dataDezP)
dataVinteP = normalize_data(dataVinteP)

# Amostragem dos dados
dadosRigidez = pd.DataFrame()
dataBaseline_df = pd.DataFrame(dataBaseline)
dataBaseline_df['y_Baseline'] = 1
dataBaseline_sampled_train = dataBaseline_df.sample(n=train_samples, random_state=42)
dataBaseline_sampled_test = dataBaseline_df.drop(dataBaseline_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataBaseline_sampled_train], ignore_index=True)

dataCincoP_df = pd.DataFrame(dataCincoP)
dataCincoP_df['y_CincoP'] = 1
dataCincoP_sampled_train = dataCincoP_df.sample(n=train_samples, random_state=42)
dataCincoP_sampled_test = dataCincoP_df.drop(dataCincoP_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCincoP_sampled_train], ignore_index=True)

dataDezP_df = pd.DataFrame(dataDezP)
dataDezP_df['y_DezP'] = 1
dataDezP_sampled_train = dataDezP_df.sample(n=train_samples, random_state=42)
dataDezP_sampled_test = dataDezP_df.drop(dataDezP_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataDezP_sampled_train], ignore_index=True)

dataVinteP_df = pd.DataFrame(dataVinteP)
dataVinteP_df['y_VinteP'] = 1
dataVinteP_sampled_train = dataVinteP_df.sample(n=train_samples, random_state=42)
dataVinteP_sampled_test = dataVinteP_df.drop(dataVinteP_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataVinteP_sampled_train], ignore_index=True)
dadosRigidezTeste = pd.concat([dataBaseline_sampled_test, dataCincoP_sampled_test, dataDezP_sampled_test, dataVinteP_sampled_test], ignore_index=True)

# Preencher valores nulos, se houver
dadosRigidez = dadosRigidez.fillna(0)
dadosRigidezTeste = dadosRigidezTeste.fillna(0)

# Definição da função de construção do modelo com Dropout
def build_model(hp):
    # Otimização do inicializador
    initializer = hp.Choice('initializer', ['glorot_uniform', 'he_normal'])
    
    # Definindo o batch size como um hiperparâmetro
    batch_size = hp.Int('batch_size', min_value=8, max_value=64, step=8)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(5830, 1)))  # Ajuste para a camada de entrada
    model.add(tf.keras.layers.Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32), 
                                     kernel_size=hp.Int('kernel_size', min_value=3, max_value=5, step=1),
                                     activation='relu',
                                     kernel_initializer=initializer))
    
    # Verificação da necessidade de Batch Normalization
    if hp.Boolean('batch_norm'):
        model.add(tf.keras.layers.BatchNormalization())
    
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    
    num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=5)
    for i in range(num_conv_layers):
        model.add(tf.keras.layers.Conv1D(filters=hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32),
                                         kernel_size=hp.Int(f'kernel_size_{i}', min_value=3, max_value=5, step=1),
                                         activation='relu',
                                         kernel_initializer=initializer))
        if hp.Boolean(f'batch_norm_{i}'):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    
    # Adicionando Dropout
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16), 
                                    activation='relu',
                                    kernel_initializer=initializer))
    
    # Adicionando a camada de saída
    model.add(tf.keras.layers.Dense(4, activation='softmax', kernel_initializer=initializer))

    # Otimização da taxa de aprendizado
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

# Função para criar o callback ReduceLROnPlateau
def build_reduce_lr_callback(hp):
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Valor fixo para o fator de redução de LR
        patience=10,  # Valor fixo para a paciência
        min_lr=1e-6  # Valor fixo para o mínimo de LR
    )

# Criação e configuração do tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=50,
    executions_per_trial=5,
    directory='my_dir',
    project_name='cnn_tuning'
)

# Divisão dos dados
x_train = dadosRigidez.drop(['y_Baseline', 'y_CincoP','y_DezP', 'y_VinteP'], axis=1).values
y_train = pd.DataFrame().assign(Baseline=dadosRigidez['y_Baseline'].values, Cinco=dadosRigidez['y_CincoP'].values, Dez=dadosRigidez['y_DezP'].values, Vinte=dadosRigidez['y_VinteP'].values).values

x_test = dadosRigidezTeste.drop(['y_Baseline', 'y_CincoP','y_DezP', 'y_VinteP'], axis=1).values
y_test = pd.DataFrame().assign(Baseline=dadosRigidezTeste['y_Baseline'].values, Cinco=dadosRigidezTeste['y_CincoP'].values, Dez=dadosRigidezTeste['y_DezP'].values, Vinte=dadosRigidezTeste['y_VinteP'].values).values

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Formatação dos dados
x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Treinamento do modelo
tuner.search(x_train, y_train,
             epochs=100,
             validation_data=(x_val, y_val),
             callbacks=[build_reduce_lr_callback(tuner)])

# Resumo dos resultados
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Melhor configuração de hiperparâmetros
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
hyperparameters_str = "\n".join([f"{key}: {value}" for key, value in best_hyperparameters.values.items()])

# Avaliação no conjunto de validação com os melhores hiperparâmetros
history = best_model.fit(x_train, y_train, 
                         epochs=100, 
                         validation_data=(x_val, y_val),
                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                                    build_reduce_lr_callback(best_hyperparameters)])

# Obter a acurácia de validação do modelo
val_accuracy = max(history.history['val_accuracy'])  # A acurácia de validação é a melhor durante o treinamento

# Avaliação no conjunto de teste
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)

file_path = 'results.txt'  # Substitua pelo caminho desejado para o arquivo de resultados

# Salvar os melhores hiperparâmetros e a acurácia de validação em um arquivo
with open(file_path, 'w') as file:
    file.write("Melhores hiperparâmetros:\n")
    file.write(hyperparameters_str)
    file.write("\n\n")
    file.write(f"Acurácia de Validação: {val_accuracy:.4f}\n")
    file.write(f"Teste de perda: {test_loss:.4f}\n")
    file.write(f"Teste de acurácia: {test_accuracy:.4f}\n")

print(f"Melhores hiperparâmetros e resultados salvos em {file_path}")
