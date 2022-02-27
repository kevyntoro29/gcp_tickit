from flask import Flask, render_template, request
import numpy as np
#import joblib
import tensorflow as tf
from pickle import load


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():    
    if request.method == "POST":        
        day = request.form.get('day')

    try:
            prediction = preprocessDataAndPredict(day)           

            return render_template('predict.html', prediction = prediction)        

    except ValueError:
            return "Please Enter valid values"        
            pass            
    pass
def preprocessDataAndPredict(day):    
    
    Xtest = np.load("Xtest.npy")
    Ypredic1 = np.load("Ypredic1.npy")
    sc = load(open('sc.pkl', 'rb'))

    last_predict = Xtest[-1][1:] 
    last_predict = np.hstack((last_predict,Ypredic1[-1])) # 9 ultimos mas 1 predict = 10
    last_predict = np.reshape(last_predict,(-1,1))
    lp = last_predict

    lp = np.reshape(lp,(1,-1))
    final_predict = []

    saved_model_path = r"C:/gcp_tickit/tf_save"
    another_strategy = tf.distribute.MirroredStrategy()
    with another_strategy.scope():
        load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        trained_model = tf.keras.models.load_model(saved_model_path, options=load_options)

    for i in range(int(day)):

        prediction_7 = trained_model.predict(lp)
        print('prediction:', prediction_7)
        final_predict.append(prediction_7)

        lp = lp[0][1:] 
        lp = np.append(lp,prediction_7) # 9 ultimos mas 1 predict = 10
        lp = np.reshape(lp,(1,-1))


    final_predict = np.reshape(final_predict,(-1,1))
    final_predict=sc.inverse_transform(final_predict)
    prediction = final_predict[-1]

    return prediction  
    pass

if __name__ == '__main__':
    app.run(debug=True)