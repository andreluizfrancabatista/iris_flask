from flask import Flask, request, render_template, url_for
import torch  
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')
    #return 'Hello, World!'

modelo = torch.jit.load('./models/model_scripted.pt')
modelo.eval()

@app.route('/predict', methods=['POST'])
def predict():
    sepal_w = float(request.form['sepal_w'])
    sepal_l = float(request.form['sepal_l'])
    petal_w = float(request.form['petal_w'])
    petal_l = float(request.form['petal_l'])
    data = np.array([sepal_w, sepal_l, petal_w, petal_l])
    data_tensor = torch.FloatTensor(data)
    y_pred = modelo(data_tensor)
    result = torch.round(y_pred, decimals=2).tolist()
    return render_template('results.html',  result=result)

#Testar o modelo - Tá funcionando!
#teste = np.array([[5.7,2.8,4.1,1.3]]) #criei o np.array
#testetensor = torch.FloatTensor(teste) #conversão para tensor
#testetensor /= 10 #conversão para valores entre 0 e 1
#y_pred = modelo(testetensor) #rodar o modelo com os dados reais
# Saída de teste: [setosa', 'versicolor', 'virginica']
#print(y_pred)