#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# In[23]:
app = Flask(__name__,static_url_path='/static')
# In[24]:


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predictDib',methods=['POST'])
def predictDib():
    model = pickle.load(open('model.pkl', 'rb'))

    dataset = pd.read_csv('diabetes.csv')
    dataset_X = dataset.iloc[:,[1, 2, 5, 6, 7]].values

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0,1))
    dataset_scaled = sc.fit_transform(dataset_X)

    arr = list(request.form.values())
    float_features = [float(x) for x in arr]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))
    # print(prediction)
    pred=''
    if prediction[0] == 1:
        pred = "You could have Diabetic problem,so please please, follow covid norms, get vaccinated and consult a Doctor."
    elif prediction[0] == 0:
        pred = "You are probably in safe zone but, please follow covid norms as much as possible "
    
    return render_template('index.html', prediction_text='{}'.format(pred))

@app.route('/predictHeart',methods=['POST'])
def predictHeart():
    
    model2 = pickle.load(open('model2.pkl', 'rb'))
    dataset2 = pd.read_csv('heart.csv')
    dataset2_X = dataset2.iloc[:,[2,3,4,5,7,8,9,11,12]].values
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0,1))
    dataset2_scaled = sc.fit_transform(dataset2_X)

    arr = list(request.form.values())
    float_features = [float(x) for x in arr]
    final_features = [np.array(float_features)]
    prediction1 = model2.predict(sc.transform(final_features))
    predh=''
    if prediction1[0] == 1:
        predh = "You could have Heart problems, so please please follow covid norms, get vaccinated and consult a Doctor."
    elif prediction1[0] == 0:
        predh = "You are probably in safe zone but, please follow covid norms as much as possible "
    
    return render_template('index.html', prediction_text='{}'.format(predh))
   


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




