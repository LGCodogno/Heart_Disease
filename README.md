<h1 align="center"> Heart Disease 🏥❤️ </h1>

<h3 align="center"> Projeto realizado durante a formação no curso Data Expert, proporcionado pela Escola DNC 📚 </h3>

<h2 align="left"> Modelo de Classificação para Doenças Coronárias </h2>

O objetivo desse projeto é, criar um sistema inteligente para tornar mais ágil o processo de triagem dos pacientes que dão entrada no hospital e então fazer o encaminhamento para um especialista. O sistema deve mostrar se o paciente possui ou não doenças coronárias. O modelo de Machine Learning escolhido é um modelo de classificação otimizado pelo F1 Score.

O dataset é público e está disponível no Kaggle: https://www.kaggle.com/datasets/priyanka841/heart-disease-prediction-uci

<h2 align="left"> Introdução </h2>

O dataset conta com 13 colunas onde fornece dados de 303 pacientes que serão utilizados para a análise com o objetivo de detectar possíveis doenças cardíacas. A coluna _target_ indica a presença ou não de doenças.

<details>
  <summary> Dicionário das colunas (features): </summary>
 
- age: Idade (Anos)
- sex: Sexo (1 = Masc e 0 = Fem)
- cp: Nível de dor ( 1 a 4 )
- trestbps: Pressão sanguínea em Repouso
- chol: colesterol em mg/dl:
- fbs: Fasting Blood Sugar (Teste diabético) > 120 mg/dl
- restecg: Eletrocardiogramas em repouso (0,1 ou 2)
- thalach: Ritmo cardíaco
- exang: Exercício físico que gerou Angina
- oldpeak: Depressão de ST induzida por exercício em relação ao repouso
- slope: Tipo de inclinação do segmento ST de pico do exercício
- ca: número de vasos sanguínios ressaltados (coloridos por fluoroscopia)
- thal: Talassemia -> 3 = normal; 6 = fixed defect; 7 = reversable defect
- target: (1 = doente; 0 = não doente)
</details>

<h2 align="left"> Análise Exploratória </h2>

Após a análise exploratória dos dados, foi realizada a correlação das colunas e vericado que em relação a variável _target_, podemos observar uma correlação mais próxima de 1 em:
- Target x Nível de dor (cp);
- Target x Rítmo Cardíaco (thalach);
- Target x Slope.

<h2 align="left"> Machine Learning </h2>

Para a fase de criação do modelo de Machine Learning, os dados foram separados em 80% treino e 20% teste. Após a separação, foram aplicados os seguintes modelos:
- Classificador Naive Bayes;
  * Gaussian;
  * Bernoulli;
  * Complement;
  * Multinomial.
- Classificador KNN - K Nearest Neighbors;
- Classificador Regressão Logística;
- Classificador Decision Tree;
- Classificador Random Forest.

Por fim, as métricas dos modelos foram organizadas numa nova tabela com foco no F1 Score:

--| Classificador | Accuracy | Precision | Recall | F1 Score
4 | KNN | 83.606557	81.818182	87.096774	84.375000
0	Gaussian NB	78.688525	76.470588	83.870968	80.000000
1	Bernoulli NB	77.049180	71.794872	90.322581	80.000000
2	Complement NB	75.409836	71.052632	87.096774	78.260870
7	Random Forest	75.409836	72.222222	83.870968	77.611940
5	Logistic Regression	73.770492	69.230769	87.096774	77.142857
6	Decision Tree	72.131148	68.421053	83.870968	75.362319
3	Multinomial NB	70.491803	65.853659	87.096774	75.000000
