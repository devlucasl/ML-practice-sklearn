#%%
import pandas as pd

#%%
# Importando dados
df = pd.read_excel("data/dados_cerveja.xlsx")
df.head()

#%% 
# features = caracteristicas em inglês
features = ['temperatura','copo','espuma','cor']
# target = variavel resposta 
target = 'classe'

X = df[features]
y = df[target]

#%%
# Variaveis precisam ser todas numericas ('copo', 'espuma' e 'cor' são categoricas e estão como string)

# Transformando mud -> 1 | pint -> 2 
# Transformando sim -> 1 | não -> 0
# Transformando clara -> 0 | escura -> 1
X = X.replace({
    "mud":1, "pint": 2,
    "sim":1, "não":0,
    "clara":0, "escura":1,
})

#%% 
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)

#%%
import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features,
               class_names=model.classes_,
               filled=True)