# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
from scipy import stats
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers, layers, initializers


plt.rcParams['font.family'] = 'Times New Roman' # Fonte dos plots

start_time = time.time()  # Record the start time

# Número de rodadas
n_runs = 20

# Selecionar 200 dados para treinamento de cada cenário
train_samples = 100
test_samples = 100

# Carregar os dados
PosSensor = 'VG'
Vagao = 'PrimVag'
DadosAll = loadmat(f'Data04-08_{PosSensor}_{Vagao}_Cut.mat') 
dataBaseline = DadosAll['Baseline']               # Sem dano
dataCincoP = DadosAll['CincoP']                   # 5% de dano
dataDezP = DadosAll['DezP']                        # 10% de dano
dataVinteP = DadosAll['VinteP']                    # 20% de dano

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

dadosRigidez = pd.DataFrame()

# Amostragem dos dados
dataBaseline_df = pd.DataFrame(dataBaseline)
dataBaseline_df['y_Baseline'] = 0
dataBaseline_sampled_train = dataBaseline_df.sample(n=train_samples, random_state=42)
dataBaseline_sampled_test = dataBaseline_df.drop(dataBaseline_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataBaseline_sampled_train], ignore_index=True)

# Adicionar dataCincoP com coluna y_CincoP
dataCincoP_df = pd.DataFrame(dataCincoP)
dataCincoP_df['y_CincoP'] = 1
dataCincoP_sampled_train = dataCincoP_df.sample(n=train_samples, random_state=42)
dataCincoP_sampled_test = dataCincoP_df.drop(dataCincoP_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataCincoP_sampled_train], ignore_index=True)

# Adicionar dataDez com coluna y_DezP
dataDezP_df = pd.DataFrame(dataDezP)
dataDezP_df['y_DezP'] = 2
dataDezP_sampled_train = dataDezP_df.sample(n=train_samples, random_state=42)
dataDezP_sampled_test = dataDezP_df.drop(dataDezP_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataDezP_sampled_train], ignore_index=True)

# Adicionar dataVinte com coluna y_VinteP
dataVinteP_df = pd.DataFrame(dataVinteP)
dataVinteP_df['y_VinteP'] = 3
dataVinteP_sampled_train = dataVinteP_df.sample(n=train_samples, random_state=42)
dataVinteP_sampled_test = dataVinteP_df.drop(dataVinteP_sampled_train.index).sample(n=test_samples, random_state=42)
dadosRigidez = pd.concat([dadosRigidez, dataVinteP_sampled_train], ignore_index=True)
dadosRigidezTeste = pd.concat([dataBaseline_sampled_test, dataCincoP_sampled_test, dataDezP_sampled_test, dataVinteP_sampled_test], ignore_index=True)

# Preencher valores nulos, se houver
dadosRigidez = dadosRigidez.fillna(0)
dadosRigidezTeste = dadosRigidezTeste.fillna(0)

# Função para criar o modelo da rede neural com base no valor de PosSensor
def create_model(n_classes=4):
    
    initializer = tf.keras.initializers.GlorotNormal()
    #regularizer = regularizers.l2(0.01)


    global PosSensor
    
    if PosSensor == 'TF':
        model = tf.keras.Sequential([
            # Primeira camada Conv1D
            tf.keras.layers.Conv1D(128, 5, activation='relu', kernel_initializer=initializer, input_shape=(5830, 1)),
            tf.keras.layers.BatchNormalization(),  # batch_norm_0
            tf.keras.layers.MaxPooling1D(2),

            # Segunda camada Conv1D
            tf.keras.layers.Conv1D(96, 4, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),  # batch_norm_1
            tf.keras.layers.MaxPooling1D(2),

            # Terceira camada Conv1D
            tf.keras.layers.Conv1D(32, 5, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.MaxPooling1D(2),

            # Quarta camada Conv1D
            tf.keras.layers.Conv1D(96, 3, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),  # batch_norm_3
            tf.keras.layers.MaxPooling1D(2),

            # Quinta camada Conv1D
            tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),  # batch_norm_4
            tf.keras.layers.MaxPooling1D(2),

            # Flatten para converter em uma dimensão
            tf.keras.layers.Flatten(),

            # Dropout para regularização
            tf.keras.layers.Dropout(0.1),

            # Camada densa com 32 unidades
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),

            # Saída final (softmax para classificação)
            tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer=initializer)
        ])
        
    elif PosSensor == 'VG':
        # Arquitetura para 'VG' com os parâmetros otimizados
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer, input_shape=(5830, 1)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(48, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Dropout(0.0),
            tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer=initializer)
        ])
    else:
        raise ValueError("PosSensor value is not recognized")

    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015 if PosSensor == 'VG' else 0.000169), 
                  metrics=['accuracy'])

    return model




# Função para treinar o modelo e obter a matriz de confusão
def train_and_evaluate_confusion_matrix(model, x_train, y_train, x_test, y_test, i):
    best_accuracy = 0
    best_conf_matrix = None

    
    # Treinamento do modelo
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1, restore_best_weights=True)

    history = model.fit(x_train, y_train, epochs=400, batch_size=10, validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])
    
    # Plotando o decaimento da função custo
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.ylim((0.01,2.1))
    plt.grid(True)
    plt.savefig(f'Loss_WithoutSpeed_{PosSensor}_n{train_samples}_it{i}.png', dpi=600, bbox_inches='tight')     
    plt.show()

    # Predições
    ytestpred = model.predict(x_test)
    ytestpred_classes = np.argmax(ytestpred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculando a matriz de confusão e a precisão
    conf_matrix = confusion_matrix(y_test_classes, ytestpred_classes)
    accuracy = accuracy_score(y_test_classes, ytestpred_classes)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_conf_matrix = conf_matrix

    return best_conf_matrix, accuracy


        
# Inicializar variáveis para salvar as curvas ROC e a melhor matriz de confusão
best_overall_accuracy = 0
best_overall_conf_matrix = None
accuracies = {scenario: [] for scenario in ['Baseline', '5%', '10%', '20%']}


# Loop principal
for i in range(n_runs):
    print(f"Execução {i+1}/{n_runs}")

    # Divisão dos dados de treinamento e validação
    x_train = dadosRigidez.drop(['y_Baseline', 'y_CincoP', 'y_DezP', 'y_VinteP'], axis=1)
    y_train = dadosRigidez[['y_Baseline', 'y_CincoP', 'y_DezP', 'y_VinteP']].idxmax(axis=1)
    y_train = y_train.replace({'y_Baseline': 0, 'y_CincoP': 1, 'y_DezP': 2, 'y_VinteP': 3})
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
    
    x_test = dadosRigidezTeste.drop(['y_Baseline', 'y_CincoP', 'y_DezP', 'y_VinteP'], axis=1)
    y_test = dadosRigidezTeste[['y_Baseline', 'y_CincoP', 'y_DezP', 'y_VinteP']].idxmax(axis=1)
    y_test = y_test.replace({'y_Baseline': 0, 'y_CincoP': 1, 'y_DezP': 2, 'y_VinteP': 3})
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)
    
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=i)

    # Criar o modelo para cada rodada
    model = create_model(n_classes=4)

    # Treinar o modelo e obter a melhor matriz de confusão
    best_conf_matrix, accuracy = train_and_evaluate_confusion_matrix(model, x_train, y_train, x_test, y_test, i)

    # Comparar acurácia para obter a melhor matriz de confusão
    if accuracy > best_overall_accuracy:
        best_overall_accuracy = accuracy
        best_overall_conf_matrix = best_conf_matrix

    # Calcular as acurácias para cada cenário individualmente
    ytestpred = model.predict(x_test)
    ytestpred_classes = np.argmax(ytestpred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    for j, scenario in enumerate(['Baseline', '5%', '10%', '20%']):
        scenario_mask = (y_test_classes == j)
        scenario_accuracy = accuracy_score(y_test_classes[scenario_mask], ytestpred_classes[scenario_mask])
        accuracies[scenario].append(scenario_accuracy)


#Plots
# Normalizar a matriz de confusão
normalized_conf_matrix = best_overall_conf_matrix.astype('float') / best_overall_conf_matrix.sum(axis=1)[:, np.newaxis]

# Plots
class_names = ['Baseline', 'DC1', 'DC2', 'DC3']

# Plotar matriz de confusão normalizada
plt.figure(figsize=(8, 6))
sns.heatmap(normalized_conf_matrix, annot=True, fmt='.2f', cmap='Blues', annot_kws={"size": 20}, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted', fontsize=22)
plt.ylabel('True', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(f'ConfusionMatrix_WithoutSpeed_{PosSensor}_{train_samples}pass.png', dpi=600, bbox_inches='tight')
plt.show()

# Plotar boxplot das acurácias
plt.figure(figsize=(8, 6))
accuracies_df = pd.DataFrame(accuracies)
# Cria o boxplot com preenchimento ativado para personalização
box = plt.boxplot([accuracies_df[col] for col in accuracies_df.columns],
                   patch_artist=True)  # Permite a personalização das caixas

# Defina as cores para as bordas
colors = ['green', 'goldenrod', 'darkorange', 'darkred']

# Iterar sobre os elementos do boxplot e modificar a cor das bordas e mustaches
for i, (patch, color) in enumerate(zip(box['boxes'], colors)):
    # Configura a cor das caixas
    patch.set_edgecolor('black')    # Define a cor da borda
    patch.set_facecolor('none')  # Remove o preenchimento da caixa
    patch.set_linewidth(1)       # Aumentar a espessura do contorno para maior visibilidade

    # Configura a cor dos whiskers e caps
    whiskers = box['whiskers'][2*i:2*i+2]
    caps = box['caps'][2*i:2*i+2]
    for whisker in whiskers:
        whisker.set_color('black')     # Define a cor dos whiskers
        whisker.set_linewidth(1)     # Define a espessura dos whiskers
        whisker.set_linestyle((0, (8, 6)))
    for cap in caps:
        cap.set_color('black')         # Define a cor dos caps
        cap.set_linewidth(1)         # Define a espessura dos caps

    # Configura a cor da linha mediana
    median = box['medians'][i]
    median.set_color(color)      # Define a cor da linha mediana
    median.set_linewidth(2)     # Define a espessura da linha mediana

# Configurações adicionais
plt.xlabel('Scenario', fontsize=24, fontfamily='serif', fontname='Times New Roman')
plt.ylabel('Accuracy', fontsize=24, fontfamily='serif', fontname='Times New Roman')
plt.xticks(ticks=range(1, len(accuracies_df.columns) + 1),
           labels=class_names, fontsize=22, fontfamily='serif', fontname='Times New Roman')
plt.yticks(fontsize=22, fontfamily='serif', fontname='Times New Roman')
plt.ylim(0, 1)  # Definir o limite máximo do eixo y como 1
plt.grid(True)

# Salvar a figura
plt.savefig(f'Boxplot_accuracy_WithoutSpeed__{PosSensor}_{train_samples}pass.png', dpi=600, bbox_inches='tight')

# Mostrar o gráfico
plt.show()

print("--- Tempo total de execução: %.2f minutos ---" % ((time.time() - start_time) / 60))

