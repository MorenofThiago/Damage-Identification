# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:52:40 2024

@author: Java 5
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers, layers, initializers

plt.rcParams['font.family'] = 'Times New Roman'

start_time = time.time()

n_runs = 20
train_samples = 50
test_samples = 50

# Carregar os dados
PosSensor = 'TF'
Vagao = 'PrimVag'
DadosAll = loadmat(f'Data04-08_{PosSensor}_{Vagao}_Cut.mat')
dataBaseline = DadosAll['Baseline']               
dataCincoP = DadosAll['CincoP']                   
dataDezP = DadosAll['DezP']                        
dataVinteP = DadosAll['VinteP']  

speed = loadmat(f'Data04-08_velocidade.mat')
speed_baseline = speed['veloc_baseline']               
speed_cincoP = speed['veloc_cincoP']                   
speed_dezP = speed['veloc_dezP']                        
speed_vinteP = speed['veloc_vinteP']    

# Normalizacao dos dados
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

speed_baseline = normalize_data(speed_baseline)
speed_cincoP = normalize_data(speed_cincoP)
speed_dezP = normalize_data(speed_dezP)
speed_vinteP = normalize_data(speed_vinteP)

dataBaseline = normalize_data(dataBaseline)
dataCincoP = normalize_data(dataCincoP)
dataDezP = normalize_data(dataDezP)
dataVinteP = normalize_data(dataVinteP)

# Função para organizar dados com velocidade correspondente
def prepare_data_with_speed(data, speed, label):
    df = pd.DataFrame(data)
    df['label'] = label
    df['speed'] = speed.flatten()  # Garantir que a velocidade esteja no formato correto
    return df

# Preparar os dados com as velocidades reais associadas
df_baseline = prepare_data_with_speed(dataBaseline, speed_baseline, 0)
df_cincoP = prepare_data_with_speed(dataCincoP, speed_cincoP, 1)
df_dezP = prepare_data_with_speed(dataDezP, speed_dezP, 2)
df_vinteP = prepare_data_with_speed(dataVinteP, speed_vinteP, 3)

# Concatenar os dados em um único DataFrame
dadosRigidez = pd.concat([df_baseline, df_cincoP, df_dezP, df_vinteP], ignore_index=True)

# Embaralhar os dados mantendo a correspondência
dadosRigidez = dadosRigidez.sample(frac=1, random_state=42).reset_index(drop=True)

# Separar os dados em treino e teste
train_data = dadosRigidez.sample(n=train_samples * 4, random_state=42)
test_data = dadosRigidez.drop(train_data.index)

# Organizar entradas e saídas para treino e teste
x_train = train_data.drop(['label', 'speed'], axis=1).values.reshape(-1, 5830, 1)
y_train = tf.keras.utils.to_categorical(train_data['label'].values, num_classes=4)
speed_train = train_data['speed'].values

# Separar os dados de teste para cada cenário
test_data_baseline = test_data[test_data['label'] == 0].sample(n=test_samples, random_state=42)
test_data_cincoP = test_data[test_data['label'] == 1].sample(n=test_samples, random_state=42)
test_data_dezP = test_data[test_data['label'] == 2].sample(n=test_samples, random_state=42)
test_data_vinteP = test_data[test_data['label'] == 3].sample(n=test_samples, random_state=42)

# Concatenar os dados de teste de todos os cenários
test_data = pd.concat([test_data_baseline, test_data_cincoP, test_data_dezP, test_data_vinteP], ignore_index=True)

# Organizar entradas e saídas para teste
x_test = test_data.drop(['label', 'speed'], axis=1).values.reshape(-1, 5830, 1)
y_test = tf.keras.utils.to_categorical(test_data['label'].values, num_classes=4)
speed_test = test_data['speed'].values

speed_train = np.array(speed_train, dtype=float)
speed_test = np.array(speed_test, dtype=float)

def create_model(n_classes=4):
    global PosSensor
    
    initializer = tf.keras.initializers.GlorotNormal()
    #regularizer = regularizers.l2(0.01)
    
    signal_input = tf.keras.Input(shape=(5830, 1), name='signal_input')
    speed_input = tf.keras.Input(shape=(1,), name='speed_input')

    if PosSensor == 'TF':
        # Entrada para o sinal de aceleração
        signal_input = tf.keras.Input(shape=(5830, 1), name="signal_input")
        
        # Primeira camada Conv1D
        signal_x = tf.keras.layers.Conv1D(128, 5, activation='relu', kernel_initializer=initializer)(signal_input)
        signal_x = tf.keras.layers.BatchNormalization()(signal_x)
        signal_x = tf.keras.layers.MaxPooling1D(2)(signal_x)
    
        # Segunda camada Conv1D
        signal_x = tf.keras.layers.Conv1D(96, 4, activation='relu', kernel_initializer=initializer)(signal_x)
        signal_x = tf.keras.layers.BatchNormalization()(signal_x)
        signal_x = tf.keras.layers.MaxPooling1D(2)(signal_x)
    
        # Terceira camada Conv1D
        signal_x = tf.keras.layers.Conv1D(32, 5, activation='relu', kernel_initializer=initializer)(signal_x)
        signal_x = tf.keras.layers.MaxPooling1D(2)(signal_x)
    
        # Quarta camada Conv1D
        signal_x = tf.keras.layers.Conv1D(96, 3, activation='relu', kernel_initializer=initializer)(signal_x)
        signal_x = tf.keras.layers.BatchNormalization()(signal_x)
        signal_x = tf.keras.layers.MaxPooling1D(2)(signal_x)
    
        # Quinta camada Conv1D
        signal_x = tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_initializer=initializer)(signal_x)
        signal_x = tf.keras.layers.BatchNormalization()(signal_x)
        signal_x = tf.keras.layers.MaxPooling1D(2)(signal_x)
    
        # Flatten para converter em uma dimensão
        signal_x = tf.keras.layers.Flatten()(signal_x)
    
        # Dimensão da saída da última camada convolucional
        signal_x_dim = tf.keras.backend.int_shape(signal_x)[-1]
    
        # Entrada para os dados de velocidade
        speed_input = tf.keras.Input(shape=(1,), name="speed_input")
    
        # Processa os dados de velocidade para coincidir com a dimensão de saída do sinal de aceleração
        speed_x = tf.keras.layers.Dense(signal_x_dim, activation='relu', kernel_initializer=initializer)(speed_input)

        combined = tf.keras.layers.Concatenate(axis=-1)([signal_x, speed_x])
        x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer)(combined)
        #x = tf.keras.layers.Dropout(0.1)(x)
    
        # Saída final (softmax para classificação)
        output = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer=initializer)(x)

        model = tf.keras.Model(inputs=[signal_input, speed_input], outputs=output)
    
        # Compilando o modelo
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.000169),
                      metrics=['accuracy'])
    
        # Resumo do modelo
        model.summary()
        
    elif PosSensor == 'VG':
                   
        # Sub-rede para o sinal de aceleração
        signal_x = tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer)(signal_input)
        signal_x = tf.keras.layers.MaxPooling1D(2)(signal_x)
        signal_x = tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer)(signal_x)
        signal_x = tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer)(signal_x)
        signal_x = tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer)(signal_x)
        signal_x = tf.keras.layers.Conv1D(64, 5, activation='relu', kernel_initializer=initializer)(signal_x)
        signal_x = tf.keras.layers.Flatten()(signal_x)
        
        # Dimensão da saída da camada de convolução
        signal_x_dim = tf.keras.backend.int_shape(signal_x)[-1]
        
        # Sub-rede para a velocidade
        speed_x = tf.keras.layers.Dense(signal_x_dim, activation='relu', kernel_initializer=initializer)(speed_input)
        
        # Concatenando as duas saídas
        combined = tf.keras.layers.Concatenate(axis=-1)([signal_x, speed_x])
        
        # Camada densa
        x = tf.keras.layers.Dense(48, activation='relu', kernel_initializer=initializer)(combined)
        
        # Saída final
        output = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
        
        # Criando o modelo
        model = tf.keras.Model(inputs=[signal_input, speed_input], outputs=output)
        
        # Compilando o modelo
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015),
                      metrics=['accuracy'])
        
        # Resumo do modelo
        model.summary()
        
    return model



def train_and_evaluate_confusion_matrix(model, x_train, y_train, x_test, y_test, speed_train, speed_test, i):
    
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    #history = model.fit([x_train, speed_train], y_train, epochs=400, batch_size=50, validation_split=0.2, verbose=1, callbacks=[early_stopping])
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1, restore_best_weights=True)

    # Treinamento do modelo
    history = model.fit(
        [x_train, speed_train],   # Suas entradas (aceleração e velocidade)
        y_train,                  # Saída
        epochs=400,               # Número de épocas
        batch_size=10,            # Tamanho do batch
        validation_split=0.2,     # Fração de validação
        verbose=1,                # Nível de verbosidade
        callbacks=[early_stopping, reduce_lr]  # Callbacks (early stopping e redução de LR)
    )  
    
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.savefig(f'Loss_ConcOutput_{PosSensor}_n{train_samples}_it{i}.png', dpi=600, bbox_inches='tight')
    plt.show()

    ytestpred = model.predict([x_test, speed_test])
    ytestpred_classes = np.argmax(ytestpred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    conf_matrix = confusion_matrix(y_test_classes, ytestpred_classes)
    accuracy = accuracy_score(y_test_classes, ytestpred_classes)

    return conf_matrix, accuracy

best_overall_accuracy = 0
best_overall_conf_matrix = None
accuracies = {scenario: [] for scenario in ['Baseline', '5%', '10%', '20%']}

for i in range(n_runs):
    print(f"Execução {i+1}/{n_runs}")
    

    model = create_model(n_classes=4)
    best_conf_matrix, accuracy = train_and_evaluate_confusion_matrix(model, x_train, y_train, x_test, y_test, speed_train, speed_test, i)

    if accuracy > best_overall_accuracy:
        best_overall_accuracy = accuracy
        best_overall_conf_matrix = best_conf_matrix

    ytestpred = model.predict([x_test, speed_test])
    ytestpred_classes = np.argmax(ytestpred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    for j, scenario in enumerate(['Baseline', '5%', '10%', '20%']):
        scenario_mask = (y_test_classes == j)
        scenario_accuracy = accuracy_score(y_test_classes[scenario_mask], ytestpred_classes[scenario_mask])
        accuracies[scenario].append(scenario_accuracy)

# Plots
class_names = ['Baseline', 'DC1', 'DC2', 'DC3']

# Plotar matriz de confusão
# Normalizar matriz de confusão para obter porcentagens
conf_matrix_normalized = best_overall_conf_matrix.astype('float') / best_overall_conf_matrix.sum(axis=1)[:, np.newaxis]

# Plotar matriz de confusão normalizada
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', annot_kws={"size": 20}, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted', fontsize=22)
plt.ylabel('True', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(f'ConfusionMatrix_Speed_ConcOutput_{PosSensor}_n{train_samples}.png', dpi=600, bbox_inches='tight')
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
plt.savefig(f'Boxplot_Speed_ConcOutput_{PosSensor}_n{train_samples}.png', dpi=600, bbox_inches='tight')

# Mostrar o gráfico
plt.show()


print("--- Tempo total de execução: %.2f minutos ---" % ((time.time() - start_time) / 60))
