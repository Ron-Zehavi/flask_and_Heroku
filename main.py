from flask import Flask
from flask import request
import json
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/predict_single')
def predict_single():
    year = int(request.args.get('year'))
    km_driven = int(request.args.get('km_driven'))
    owner = int(request.args.get('owner'))
    name_Chevrolet = int(request.args.get('name_Chevrolet'))
    name_Ford = int(request.args.get('name_Ford'))
    name_Honda = int(request.args.get('name_Honda'))
    name_Hyundai = int(request.args.get('name_Hyundai'))
    name_Mahindra = int(request.args.get('name_Mahindra'))
    name_Maruti = int(request.args.get('name_Maruti'))
    name_Nissan = int(request.args.get('name_Nissan'))
    name_Other = int(request.args.get('name_Other'))
    name_Renault = int(request.args.get('name_Renault'))
    name_Skoda = int(request.args.get('name_Skoda'))
    name_Tata = int(request.args.get('name_Tata'))
    name_Toyota = int(request.args.get('name_Toyota'))
    name_Volkswagen = int(request.args.get('name_Volkswagen'))
    fuel_Diesel = int(request.args.get('fuel_Diesel'))
    fuel_Other = int(request.args.get('fuel_Other'))
    fuel_Petrol = int(request.args.get('fuel_Petrol'))
    seller_type_Dealer = int(request.args.get('seller_type_Dealer'))
    seller_type_Individual = int(request.args.get('seller_type_Individual'))
    transmission_Automatic = int(request.args.get('transmission_Automatic'))
    transmission_Manual = int(request.args.get('transmission_Manual'))

    # load model from file
    loaded_model = pickle.load(open('finalized_model.pickle', 'rb'))
    names = [year, km_driven, owner, name_Chevrolet, name_Ford,
             name_Honda, name_Hyundai, name_Mahindra, name_Maruti,
             name_Nissan, name_Other, name_Renault, name_Skoda, name_Tata,
             name_Toyota, name_Volkswagen, fuel_Diesel, fuel_Other,
             fuel_Petrol, seller_type_Dealer, seller_type_Individual,
             transmission_Automatic, transmission_Manual]
    y_pred = loaded_model.predict([names])
    return str(y_pred[0])


@app.route("/json", methods=["POST"])
def Multiple_prediction():
    req = request.get_json()
    loaded_model = pickle.load(open('finalized_model.pickle', 'rb'))
    df_req = pd.DataFrame.from_dict(req)
    y_pred = loaded_model.predict(df_req)
    y_pred = y_pred.tolist()
    response = json.dumps(y_pred)
    return str(response)


if __name__ == '__main__':
    app.run(debug=True)