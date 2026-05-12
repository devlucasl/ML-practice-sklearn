# %%
import pandas as pd

# %%
df = pd.read_excel('data/dados_frutas.xlsx')
df

# %%
from sklearn import tree

# %% 
arvore = tree.DecisionTreeClassifier(random_state=42)

# %%
# Variavel resposta ou foco é FRUTA 
# resposta = y
y = df['Fruta']

caracteristicas = ['Arredondada','Suculenta','Vermelha','Doce']
X = df[caracteristicas]

# %%
# y é uma serie
y

# %%
X

# %%
# ISSO AQUI É MACHINE LEARNING!!!!!!!!
# fit é AJUSTAR o modelo - vai criar a arvore
arvore.fit(X, y)

# %%
df

#%%
# 1 e 0 é binario (obv), corresponde se possui ou não as caracteristicas para a predição
print(f'a predição é: {arvore.predict([[0,0,0,1]])}')

# %% 
import matplotlib.pyplot as plt

# Plotando "grafico" da arvore usando pyplot
plt.figure(dpi=400  )

tree.plot_tree(arvore, 
               feature_names=caracteristicas, 
               class_names=arvore.classes_,
               filled=True)

#%%
# predict_proba mostra a probabilidade de ser
proba = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=arvore.classes_)