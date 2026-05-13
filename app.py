from flask import Flask, request, jsonify, render_template, redirect
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

app = Flask(__name__, template_folder='templates')

# ─────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────
logreg = joblib.load('models/logreg.pkl')
mlp = joblib.load('models/mlp.pkl')
scaler = joblib.load('models/scaler.pkl')

# ─────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [

    'age',
    'anaemia',
    'creatinine_phosphokinase',
    'diabetes',
    'ejection_fraction',
    'high_blood_pressure',
    'platelets',
    'serum_creatinine',
    'serum_sodium',
    'sex',
    'smoking',
    'time'
]

TARGET_NAME = 'DEATH_EVENT'

N_FEATURES = len(FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────
# EXTRAER X E Y
# ─────────────────────────────────────────────────────────────
def extraer_X_y(df):

    info = []

    # Limpiar columnas
    df.columns = [c.strip() for c in df.columns]

    # Buscar target
    target_col = None

    for c in df.columns:

        if c.lower() == TARGET_NAME.lower():

            target_col = c

            break

    # Si existe target
    if target_col:

        y_true = df[target_col].astype(int).values

        df_feat = df.drop(columns=[target_col])

        info.append(f"Target detectado: {target_col}")

    else:

        y_true = None

        df_feat = df.copy()

        info.append("Sin target → solo predicción")

    # Verificar columnas
    feat_lower = [c.lower() for c in df_feat.columns]

    expected_lower = [f.lower() for f in FEATURE_NAMES]

    # Caso ideal
    if set(feat_lower) == set(expected_lower):

        col_map = {
            c.lower(): c
            for c in df_feat.columns
        }

        ordered_cols = [
            col_map[f]
            for f in expected_lower
        ]

        X = df_feat[ordered_cols].values

        info.append(
            "✅ Columnas correctas detectadas y ordenadas"
        )

    # Fallback
    elif len(df_feat.columns) >= N_FEATURES:

        X = df_feat.iloc[:, :N_FEATURES].values

        info.append(
            "⚠️ Columnas no coinciden → usando primeras columnas"
        )

    else:

        raise ValueError(
            f"Se requieren {N_FEATURES} columnas "
            f"pero se encontraron {len(df_feat.columns)}"
        )

    # Convertir
    try:

        X = X.astype(float)

    except Exception as e:

        raise ValueError(
            f"Error convirtiendo a numérico: {e}"
        )

    return X, y_true, "\n".join(info)


# ─────────────────────────────────────────────────────────────
# SELECCIONAR MODELO
# ─────────────────────────────────────────────────────────────
def seleccionar_modelo(tipo):

    return mlp if tipo == 'mlp' else logreg


# ─────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():

    tab = request.args.get(
        'tab',
        'individual'
    )

    return render_template(

        'index.html',

        tab=tab,

        pred_individual=None,

        modelo='logreg'
    )


# ─────────────────────────────────────────────────────────────
# LIMPIAR
# ─────────────────────────────────────────────────────────────
@app.route('/clean')
def clean():

    return redirect('/')


# ─────────────────────────────────────────────────────────────
# PREDICCIÓN INDIVIDUAL
# ─────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():

    # =========================================================
    # JSON API
    # =========================================================
    if request.is_json:

        data = request.get_json()

        try:

            features = []

            for f in FEATURE_NAMES:

                if f not in data:

                    raise ValueError(
                        f'Falta el campo "{f}"'
                    )

                features.append(float(data[f]))

            X = np.array(features).reshape(1, -1)

            X_scaled = scaler.transform(X)

            model = seleccionar_modelo(
                data.get('model', 'logreg')
            )

            pred = int(model.predict(X_scaled)[0])

            # Probabilidad
            if hasattr(model, "predict_proba"):

                prob = float(
                    model.predict_proba(X_scaled)[0][pred]
                )

            else:

                prob = None

            return jsonify({

                'prediction': pred,

                'probability': (
                    round(prob, 4)
                    if prob is not None
                    else None
                ),

                'message': (
                    'Fallecerá'
                    if pred == 1
                    else 'Sobrevivirá'
                )
            })

        except Exception as e:

            return jsonify({
                'error': str(e)
            }), 400

    # =========================================================
    # HTML
    # =========================================================
    model_type = request.form.get(
        'modelo',
        'logreg'
    )

    form_data = {

        f: request.form.get(f, '')

        for f in FEATURE_NAMES
    }

    try:

        features = []

        for f in FEATURE_NAMES:

            value = request.form.get(f)

            if value is None or value.strip() == '':

                raise ValueError(
                    f'El campo "{f}" está vacío'
                )

            features.append(float(value))

        X = np.array(features).reshape(1, -1)

        X_scaled = scaler.transform(X)

        model = seleccionar_modelo(model_type)

        pred = int(model.predict(X_scaled)[0])

        mensaje = (

            '⚠️ Alto riesgo de fallecimiento'

            if pred == 1

            else '✅ Bajo riesgo / supervivencia'
        )

        return render_template(

            'index.html',

            pred_individual=pred,

            mensaje=mensaje,

            modelo=model_type,

            tab='individual',

            **form_data
        )

    except Exception as e:

        return render_template(

            'index.html',

            error=str(e),

            pred_individual=None,

            modelo=model_type,

            tab='individual',

            **form_data
        )


# ─────────────────────────────────────────────────────────────
# PREDICCIÓN POR LOTES
# ─────────────────────────────────────────────────────────────
@app.route('/batch', methods=['POST'])
def batch():

    file = request.files.get('file')

    if not file:

        return render_template(

            'index.html',

            error='No se subió archivo CSV',

            tab='lotes',

            modelo='logreg'
        )

    model_type = request.form.get(
        'modelo_batch',
        'logreg'
    )

    try:

        df = pd.read_csv(
            file,
            sep=None,
            engine='python'
        )

        model = seleccionar_modelo(model_type)

        X, y_true, info = extraer_X_y(df)

        X_scaled = scaler.transform(X)

        y_pred = model.predict(X_scaled).astype(int)

        # Si hay etiquetas reales
        if y_true is not None:

            acc = accuracy_score(
                y_true,
                y_pred
            )

            matriz = confusion_matrix(
                y_true,
                y_pred
            ).tolist()

            reporte = classification_report(

                y_true,

                y_pred,

                output_dict=True
            )

            info += f"\n\nAccuracy: {acc:.2%}"

        else:

            acc = None

            matriz = None

            reporte = None

            info += (
                f"\nTotal predicciones: {len(y_pred)}"
            )

        return render_template(

            'index.html',

            matriz=matriz,

            reporte=reporte,

            accuracy=(
                round(acc * 100, 2)
                if acc is not None
                else None
            ),

            predicciones=y_pred.tolist(),

            info_msg=info.replace("\n", "<br>"),

            modelo=model_type,

            tab='lotes'
        )

    except Exception as e:

        return render_template(

            'index.html',

            error=str(e),

            modelo=model_type,

            tab='lotes'
        )


# ─────────────────────────────────────────────────────────────
# API JSON BATCH
# ─────────────────────────────────────────────────────────────
@app.route('/predict-batch', methods=['POST'])
def predict_batch():

    file = request.files.get('file')

    model_type = request.form.get(
        'model',
        'logreg'
    )

    if not file:

        return jsonify({
            'error': 'No file provided'
        }), 400

    try:

        df = pd.read_csv(
            file,
            sep=None,
            engine='python'
        )

        model = seleccionar_modelo(model_type)

        X, y_true, info = extraer_X_y(df)

        X_scaled = scaler.transform(X)

        y_pred = model.predict(X_scaled).astype(int)

        result = {

            'predictions': y_pred.tolist(),

            'count': len(y_pred),

            'info': info
        }

        if y_true is not None:

            result['accuracy'] = round(

                accuracy_score(
                    y_true,
                    y_pred
                ),

                4
            )

            result['confusion_matrix'] = (

                confusion_matrix(
                    y_true,
                    y_pred
                ).tolist()
            )

            result['classification_report'] = (

                classification_report(
                    y_true,
                    y_pred,
                    output_dict=True
                )
            )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 400


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print(
        "Servidor corriendo en "
        "http://localhost:5000"
    )

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )