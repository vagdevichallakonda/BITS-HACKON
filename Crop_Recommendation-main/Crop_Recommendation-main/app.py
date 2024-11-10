import pickle
import numpy as np  # Make sure NumPy is imported
from flask import Flask, request, render_template, redirect, url_for

model = pickle.load(open(r'model.pkl', 'rb'))
sc = pickle.load(open(r'standscaler.pkl', 'rb'))
mx = pickle.load(open(r'minmaxscaler.pkl', 'rb'))


app = Flask(__name__)

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
    15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Home route
@app.route('/')
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=['POST'])
def predict():
    # Collect input data
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    # Prepare features for model prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    # Get predicted crop name
    crop = crop_dict.get(prediction[0], "Unknown")
    return redirect(url_for('crop_info', crop=crop))

# New route to display crop information
@app.route("/crop_info")
def crop_info():
    crop = request.args.get('crop', 'Unknown')
    return render_template('crop_info.html', crop=crop)

if __name__ == "__main__":
    app.run(debug=True)
