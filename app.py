from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')
model = pickle.load(open("model_new.pkl","rb"))

@app.route("/predict", methods=["POST"])
def predict():
    # data dimasukkan dalam variabel dalam bentuk desimal
    float_features = [float(x) for x in request.form.values()]
    print(float_features)
    # data diubah menjadi bentuk (6,1) 6 baris 1 kolom
    features = np.asarray(float_features)
    print(features)
    # data diubah menjadi bentuk (1,6) 1 baris 6 kolom 
    new_features = features.reshape(1,-1)
    print(new_features)
    prediction = model.predict(new_features)
    #return render_template("index.html", prediction_text = "Your calories burn is {} Kilocalorie".format(str(prediction).strip("[]")))
    return render_template("index.html", prediction_text="Your calories burn is {:.2f} Kilocalorie".format(float(prediction)))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)