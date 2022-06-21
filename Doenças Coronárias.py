#!/usr/bin/env python
# coding: utf-8

# # Modelo de Classificação para Doenças Coronárias
# 
# ## Objetivo do projeto
# 
# O objetivo desse projeto é, criar um sistema inteligente para tornar mais ágil o processo de triagem dos pacientes que dão entrada no hospital e então fazer o encaminhamento para um especialista. O sistema deve mostrar se o paciente possui ou não doenças coronárias. O modelo de Machine Learning escolhido é um modelo de classificação otimizado pelo F1 Score.
# 
# O dataset é público e está disponível no Kaggle: https://www.kaggle.com/datasets/priyanka841/heart-disease-prediction-uci

# ### Dicionário das colunas (features):
# 
# - age: Idade (Anos)
# - sex: Sexo (1 = Masc e 0 = Fem)
# - cp: Nível de dor ( 1 a 4 )
# - trestbps: Pressão sanguínea em Repouso
# - chol: colesterol em mg/dl:
# - fbs: Fasting Blood Sugar (Teste diabético) > 120 mg/dl
# - restecg: Eletrocardiogramas em repouso (0,1 ou 2)
# - thalach: Ritmo cardíaco
# - exang: Exercício físico que gerou Angina
# - oldpeak: Depressão de ST induzida por exercício em relação ao repouso
# - slope: Tipo de inclinação do segmento ST de pico do exercício
# - ca: número de vasos sanguínios ressaltados (coloridos por fluoroscopia)
# - thal: Talassemia -> 3 = normal; 6 = fixed defect; 7 = reversable defect
# - target: (1 = doente; 0 = não doente)

# ## Importação das bibliotecas

# In[112]:


# Bibliotecas para visualização e tratamento dos dados
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# ## Visualização dos Dados

# In[113]:


# Visualização do datast completo
heart_df = pd.read_csv('heart.csv')
heart_df


# In[114]:


# Verificação das colunas e tipos de dados
heart_df.info()


# O dataset estudado não possui valores nulos e temos apenas uma coluna como tipo float.

# In[115]:


heart_df.isnull().sum()


# In[116]:


# Visualização dos percentis para verificação de outliers
heart_df.describe(percentiles=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).transpose()


# #### Verificando as colunas em relação ao "target":

# In[117]:


# Visualização gráfica por coluna (feature):
for i in heart_df.columns:
    sns.histplot(x=i, data=heart_df, hue='target', element='poly')
    plt.show()


# In[118]:


# Distruibuição da variáveis target
heart_df['target'].value_counts()


# #### Conclusões à partir dos gráficos acima em relação às doenças coronárias:
# 
# - Idade: em grande parte, a partir dos 40 anos é possível observar um pico bastante elevado de pessoas desenvolvendo doenças coronárias, decaindo gradativamente a partir dos 50 anos;
# - Sexo: mulheres tem uma maior tendência no desenvolvimento das doenças coronárias;
# - Nível de dor: pessoas que sentiram níveis moderados de dor possuem maior chance de ter doenças coronárias;
# - Pressão sanguínea: pessoas com pressão sanguínea entre 120mmHg e 140mmHg possuem uma maior tendência a possuir doenças coronárias;
# - Nível de colesterol: a partir de ~100mg/dl até ~280mg/dl de colesterol, existe uma maior tendência de ter doenças coronárias;
# - Ritmo cardíaco: é possível observar uma grande incidência à partir do aumento do ritmo cardíaco
# - Exercícios físicos: a tendência a desenvolver doenças coronárias está mais próxima do valor 0, ou seja, possivelmente pessoas que praticam pouca ou nenhuma atividade física;
# - Depressão de ST: valores próximos de 0 tem maior chance de possuir a doença;
# - Slope: valores = 2 possui maior risco de desenvolver a doença;
# - Vasos sanguíneos ressaltados: pessoas com 0 vasos ressaltados possuem maior tendência à ter a doença;
# - Talassemia (doença que resulta em anemia): pessoas com valores = 2 possuem maior chance de ter a doença.

# In[119]:


# Plotagem do gráfico heatmap para visualização da correlação:
plt.figure(figsize=(15,10))
sns.heatmap(heart_df.corr(), annot=True, cmap='YlGnBu')
plt.show()


# #### Conclusões à partir do gráfico heatmap:
# 
# Em relação à variável _target_, podemos observar uma correlação mais próxima de 1 em:
# - Target x Nível de dor (cp);
# - Target x Rítmo Cardíaco (thalach);
# - Target x Slope.

# In[120]:


# Plotagem dos gráficos boxplot para visualização de outliers:
for i in heart_df.columns:
    sns.boxplot(x='target', y=i, data=heart_df)
    plt.show()


# # Machine Learning
# 
# Nessa etapa, seguiremos alguns passos para a criação do modelo e, posteriormente, verificação do modelo.

# #### Preparação dos Dados
# 
# No processo de preparação dos dados, será feita a separação dos dados de treino e teste. Para isso, durante o processo, faremos a importação das bibliotecas necessárias.

# In[121]:


# Determinando os valores X e Y
X = heart_df.drop(columns=['target'])
Y = heart_df['target']


# In[122]:


# Importação da biblioteca necessária para separação dos dados de treino e teste
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
print(f"Shape X_train: {X_train.shape}")
print(f"Shape y_train: {y_train.shape}")
print(f"Shape X_test: {X_test.shape}")
print(f"Shape y_test: {y_test.shape}")


# In[123]:


# Normalização das variáveis
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Transformanda para dataframe para visualização
X_train = pd.DataFrame(X_train_scaled,columns = X_train.columns)
X_test = pd.DataFrame(X_test_scaled,columns = X_test.columns)


# ### Classificador Naive Bayes

# In[125]:


# Importação das bibliotecas utilizadas para os modelos de Classificação Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


# In[126]:


# Testar para diferentes configurações de Naive Bayes
# Gaussian
clf_GNB = GaussianNB()
clf_GNB.fit(X_train,y_train)
y_pred = clf_GNB.predict(X_test)

# Variáveis com as métricas do método Gaussian
accuracy_GNB = accuracy_score(y_test, y_pred)*100
precision_GNB = precision_score(y_test, y_pred)*100
recall_GNB = recall_score(y_test, y_pred)*100
f1_GNB = f1_score(y_test, y_pred)*100
print(f"Accuracy GuassianNB: {accuracy_GNB}%")
print(f"Precision GuassianNB: {precision_GNB}%")
print(f"Recall GuassianNB: {recall_GNB}%")
print(f"F1 Score GuassianNB: {f1_GNB}%")


# In[127]:


print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf_GNB, X_test, y_test, display_labels=["Yes", "No"], values_format='d')
plt.grid(False)
plt.show()


# In[128]:


# Bernoulli
clf_BNB = BernoulliNB()
clf_BNB.fit(X_train,y_train)
y_pred = clf_BNB.predict(X_test)

# Variáveis com as métricas do método Bernoulli
accuracy_BNB = accuracy_score(y_test, y_pred)*100
precision_BNB = precision_score(y_test, y_pred)*100
recall_BNB = recall_score(y_test, y_pred)*100
f1_BNB = f1_score(y_test, y_pred)*100
print(f"Accuracy BernoulliNB: {accuracy_BNB}%")
print(f"Precision BernoulliNB: {precision_BNB}%")
print(f"Recall BernoulliNB: {recall_BNB}%")
print(f"F1 Score BernoulliNB: {f1_BNB}%")


# In[129]:


print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf_BNB, X_test, y_test, display_labels=["Yes", "No"], values_format='d')
plt.grid(False)
plt.show()


# In[130]:


# Complement
clf_CNB = ComplementNB()
clf_CNB.fit(X_train,y_train)
y_pred = clf_CNB.predict(X_test)

# Variáveis com as métricas do método Complement
accuracy_CNB = accuracy_score(y_test, y_pred)*100
precision_CNB = precision_score(y_test, y_pred)*100
recall_CNB = recall_score(y_test, y_pred)*100
f1_CNB = f1_score(y_test, y_pred)*100
print(f"Accuracy ComplementNB: {accuracy_CNB}%")
print(f"Precision ComplementNB: {precision_CNB}%")
print(f"Recall ComplementNB: {recall_CNB}%")
print(f"F1 Score ComplementNB: {f1_CNB}%")


# In[131]:


print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf_CNB, X_test, y_test, display_labels=["Yes", "No"], values_format='d')
plt.grid(False)
plt.show()


# In[132]:


# Multinomial
clf_MNB = MultinomialNB()
clf_MNB.fit(X_train,y_train)
y_pred = clf_MNB.predict(X_test)

# Variáveis com as métricas do método Multinomial
accuracy_MNB = accuracy_score(y_test, y_pred)*100
precision_MNB = precision_score(y_test, y_pred)*100
recall_MNB = recall_score(y_test, y_pred)*100
f1_MNB = f1_score(y_test, y_pred)*100
print(f"Accuracy MultinomialNB: {accuracy_MNB}%")
print(f"Precision MultinomialNB: {precision_MNB}%")
print(f"Recall MultinomialNB: {recall_MNB}%")
print(f"F1 Score MultinomialNB: {f1_MNB}%")


# In[133]:


print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf_MNB, X_test, y_test, display_labels=["Yes", "No"], values_format='d')
plt.grid(False)
plt.show()


# In[134]:


# Criando variáveis das métricas Naive Bayes

clas_NB = []
clas_NB.append('Gaussian NB')
clas_NB.append('Bernoulli NB')
clas_NB.append('Complement NB')
clas_NB.append('Multinomial NB')

accuracy_NB = []
accuracy_NB.append(accuracy_GNB)
accuracy_NB.append(accuracy_BNB)
accuracy_NB.append(accuracy_CNB)
accuracy_NB.append(accuracy_MNB)

precision_NB = []
precision_NB.append(precision_GNB)
precision_NB.append(precision_BNB)
precision_NB.append(precision_CNB)
precision_NB.append(precision_MNB)

recall_NB = []
recall_NB.append(recall_GNB)
recall_NB.append(recall_BNB)
recall_NB.append(recall_CNB)
recall_NB.append(recall_MNB)

f1score_NB = []
f1score_NB.append(f1_GNB)
f1score_NB.append(f1_BNB)
f1score_NB.append(f1_CNB)
f1score_NB.append(f1_MNB)


# In[135]:


# Visualização das métricas Naive Bayes
metricas_NB = pd.DataFrame({'Classificador':clas_NB, 'Accuracy':accuracy_NB,'Precision':precision_NB,'Recall':recall_NB,'F1 Score':f1score_NB})
metricas_NB.sort_values(by='F1 Score', ascending=False)


# ### Classificador KNN - K Nearest Neighbors

# In[136]:


# Importação das bibliotecas utilizadas para o modelo de Classificação KNN  
from sklearn.neighbors import KNeighborsClassifier


# In[137]:


scores_list = []
K_neighbors = range(1, 20)


# In[138]:


for k in K_neighbors:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  scores_list.append(accuracy_score(y_test, y_pred))


# In[139]:


import matplotlib.pyplot as plt

plt.plot(K_neighbors, scores_list)
plt.xlabel("Valor de K")
plt.ylabel("Accuracy")


# Nota-se que o valor de k = 5 possui um valor maior de Acurácia.

# In[140]:


clf_KNN = KNeighborsClassifier(n_neighbors=5)
clf_KNN.fit(X_train,y_train)
y_pred = clf_KNN.predict(X_test)

# Variáveis das métricas de KNN
accuracy_KNN = accuracy_score(y_test, y_pred)*100
precision_KNN = precision_score(y_test, y_pred)*100
recall_KNN = recall_score(y_test, y_pred)*100
f1_KNN = f1_score(y_test, y_pred)*100
print(f"Accuracy KNN: {accuracy_KNN}%")
print(f"Precision KNN: {precision_KNN}%")
print(f"Recall KNN: {recall_KNN}%")
print(f"F1 Score KNN: {f1_KNN}%")


# In[141]:


print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf_KNN, X_test, y_test, display_labels=["Yes", "No"], values_format='d')
plt.grid(False)
plt.show()


# ### Classificador Regressão Logística

# In[142]:


# Importação das bibliotecas utilizadas para os modelos de Classificação Regressão Logística
from sklearn.linear_model import LogisticRegression


# In[143]:


clf_RL = LogisticRegression()
clf_RL.fit(X_train,y_train)
y_pred = clf_RL.predict(X_test)

# Variáveis das métricas de Regressão Logística
accuracy_RL = accuracy_score(y_test, y_pred)*100
precision_RL = precision_score(y_test, y_pred)*100
recall_RL = recall_score(y_test, y_pred)*100
f1_RL = f1_score(y_test, y_pred)*100
print(f"Accuracy KNN: {accuracy_RL}%")
print(f"Precision KNN: {precision_RL}%")
print(f"Recall KNN: {recall_RL}%")
print(f"F1 Score KNN: {f1_RL}%")


# In[144]:


print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf_RL, X_test, y_test, display_labels=["Yes", "No"], values_format='d')
plt.grid(False)
plt.show()


# ### Classificador Decision Tree

# In[145]:


# Importação das bibliotecas utilizadas para os modelos de Classificação Decision Tree
from sklearn.tree import DecisionTreeClassifier


# In[146]:


clf_DT = DecisionTreeClassifier(criterion="entropy")
clf_DT.fit(X_train,y_train)
y_pred = clf_DT.predict(X_test)

# Variáveis das métricas de Regressão Logística
accuracy_DT = accuracy_score(y_test, y_pred)*100
precision_DT = precision_score(y_test, y_pred)*100
recall_DT = recall_score(y_test, y_pred)*100
f1_DT = f1_score(y_test, y_pred)*100
print(f"Accuracy KNN: {accuracy_DT}%")
print(f"Precision KNN: {precision_DT}%")
print(f"Recall KNN: {recall_DT}%")
print(f"F1 Score KNN: {f1_DT}%")


# In[147]:


print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf_DT, X_test, y_test, display_labels=["Yes", "No"], values_format='d')
plt.grid(False)
plt.show()


# ### Classificador Random Forest

# In[148]:


# Importação das bibliotecas utilizadas para os modelos de Classificação Random Forest
from sklearn.ensemble import RandomForestClassifier


# In[149]:


clf_RF = RandomForestClassifier(criterion='entropy', n_estimators=150) 
clf_RF.fit(X_train,y_train)
y_pred = clf_RF.predict(X_test)

# Variáveis das métricas de Regressão Logística
accuracy_RF = accuracy_score(y_test, y_pred)*100
precision_RF = precision_score(y_test, y_pred)*100
recall_RF = recall_score(y_test, y_pred)*100
f1_RF = f1_score(y_test, y_pred)*100
print(f"Accuracy KNN: {accuracy_RF}%")
print(f"Precision KNN: {precision_RF}%")
print(f"Recall KNN: {recall_RF}%")
print(f"F1 Score KNN: {f1_RF}%")


# In[150]:


print(classification_report(y_test, y_pred))

plot_confusion_matrix(clf_RF, X_test, y_test, display_labels=["Yes", "No"], values_format='d')
plt.grid(False)
plt.show()


# ## Comparação das Métricas dos Classificadores
# 

# In[151]:


clas_final = []
clas_final.append('Gaussian NB')
clas_final.append('Bernoulli NB')
clas_final.append('Complement NB')
clas_final.append('Multinomial NB')
clas_final.append('KNN')
clas_final.append('Logistic Regression')
clas_final.append('Decision Tree')
clas_final.append('Random Forest')

accuracy_final = []
accuracy_final.append(accuracy_GNB)
accuracy_final.append(accuracy_BNB)
accuracy_final.append(accuracy_CNB)
accuracy_final.append(accuracy_MNB)
accuracy_final.append(accuracy_KNN)
accuracy_final.append(accuracy_RL)
accuracy_final.append(accuracy_DT)
accuracy_final.append(accuracy_RF)

precision_final = []
precision_final.append(precision_GNB)
precision_final.append(precision_BNB)
precision_final.append(precision_CNB)
precision_final.append(precision_MNB)
precision_final.append(precision_KNN)
precision_final.append(precision_RL)
precision_final.append(precision_DT)
precision_final.append(precision_RF)

recall_final = []
recall_final.append(recall_GNB)
recall_final.append(recall_BNB)
recall_final.append(recall_CNB)
recall_final.append(recall_MNB)
recall_final.append(recall_KNN)
recall_final.append(recall_RL)
recall_final.append(recall_DT)
recall_final.append(recall_RF)

f1score_final = []
f1score_final.append(f1_GNB)
f1score_final.append(f1_BNB)
f1score_final.append(f1_CNB)
f1score_final.append(f1_MNB)
f1score_final.append(f1_KNN)
f1score_final.append(f1_RL)
f1score_final.append(f1_DT)
f1score_final.append(f1_RF)


# In[155]:


# Visualização das métricas finais
metricas_finais = pd.DataFrame({'Classificador':clas_final, 'Accuracy':accuracy_final,'Precision':precision_final,'Recall':recall_final,'F1 Score':f1score_final})
metricas_finais.sort_values(by='F1 Score', ascending=False)


# #### Conclusão à partir da avaliação do F1 Score
# 
# Após a aplicação dos modelos de Classificação citados na tabela acima, pode-se observar que o modelo KNN possui o maior F1 Score seguido do modelo Gaussian Naive Bayes.
