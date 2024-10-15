from flask import Flask, request, render_template
import pandas as pd
import joblib

# Cargar el modelo
model = joblib.load('Aprendizaje Artificialmodel.joblib')

app = Flask(__name__)

# Define el orden de las características esperado por el modelo
feature_order = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 
    'sex_male', 'cp_atypical angina', 'cp_non-anginal pain', 
    'cp_typical angina', 'fbs_1', 'restecg_1', 'restecg_2', 
    'exang_1', 'slope_2', 'slope_3', 'ca_1.0', 'ca_2.0', 'ca_3.0', 
    'thal_6.0', 'thal_7.0'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Obtener los datos del formulario
        input_data = {
            'age': int(request.form['age']),
            'trestbps': int(request.form['trestbps']),
            'chol': int(request.form['chol']),
            'thalach': int(request.form['thalach']),
            'oldpeak': float(request.form['oldpeak']),
            'sex_male': int(request.form['sex']),
            'cp_atypical angina': int(request.form['cp'] == '1'),
            'cp_non-anginal pain': int(request.form['cp'] == '2'),
            'cp_typical angina': int(request.form['cp'] == '0'),
            'fbs_1': int(request.form['fbs']),
            'restecg_1': int(request.form['restecg'] == '1'),
            'restecg_2': int(request.form['restecg'] == '2'),
            'exang_1': int(request.form['exang']),
            'slope_2': int(request.form['slope'] == '1'),
            'slope_3': int(request.form['slope'] == '2'),
            'ca_1.0': int(request.form['ca'] == '1'),
            'ca_2.0': int(request.form['ca'] == '2'),
            'ca_3.0': int(request.form['ca'] == '3'),
            'thal_6.0': int(request.form['thal'] == '6'),
            'thal_7.0': int(request.form['thal'] == '7'),
        }

            # Convertir a DataFrame y alinear las columnas
        input_df = pd.DataFrame([input_data]).reindex(columns=feature_order, fill_value=0)
        
        # Hacer la predicción
        prediction = model.predict(input_df)[0]

        # Convertir la predicción a texto
        resultado = "Positivo" if prediction == 1 else "Negativo"
        
        # Renderizar el resultado
        return render_template('result.html', prediction=resultado)
    except Exception as e:
        return f"Ocurrió un error: {e}"
if __name__ == '__main__':
    app.run(debug=True)
