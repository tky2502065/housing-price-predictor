from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "housing_price_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
META_PATH = os.path.join(BASE_DIR, "model", "meta.pkl")

app = Flask(__name__)


model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
meta = joblib.load(META_PATH)


def predict_price(land_area, floor_area, building_age, station_minutes, ward, district,
                   structure=None, city_planning=None,
                   coverage_ratio=None, floor_area_ratio=None, road_width=None):

    structure = structure or meta["default_structure"]
    city_planning = city_planning or meta["default_city_planning"]
    coverage_ratio = meta["median_coverage_ratio"] if coverage_ratio is None else coverage_ratio
    floor_area_ratio = meta["median_floor_area_ratio"] if floor_area_ratio is None else floor_area_ratio
    road_width = meta["median_road_width"] if road_width is None else road_width

    district_avg_price = meta["district_avg"].get(district, meta["global_mean_price"])

    row = {col: 0.0 for col in meta["feature_cols"]}
    row.update({
        "land_area": land_area, "floor_area": floor_area,
        "building_age": building_age, "station_minutes": station_minutes,
        "coverage_ratio": coverage_ratio, "floor_area_ratio": floor_area_ratio,
        "road_width": road_width, "district_avg_price": district_avg_price,
    })
    for key in [f"ward_{ward}", f"structure_{structure}", f"cityplan_{city_planning}"]:
        if key in row:
            row[key] = 1

    x_cont = np.array([[row[c] for c in meta["continuous_cols"]]], dtype="float32")
    x_cont_scaled = scaler.transform(x_cont)
    x_dummy = np.array([[row[c] for c in meta["dummy_cols"]]], dtype="float32")
    x_scaled = np.hstack([x_cont_scaled, x_dummy])

    pred_log = model.predict(x_scaled, verbose=0)[0][0]
    return float(np.expm1(pred_log))


@app.route("/predict_form")
def predict_form():
    return render_template(
        "predict_form.html",
        ward_list=meta["ward_list"],
        district_by_ward=meta["district_by_ward"],
    )

@app.route("/predict", methods=["POST"])
def predict():
    land_area = float(request.form.get("land_area"))
    floor_area = float(request.form.get("floor_area"))
    building_age = float(request.form.get("building_age"))
    station_minutes = float(request.form.get("station_minutes"))
    ward = request.form.get("ward")
    district = request.form.get("district")

    warning = None
    if land_area > 300 or floor_area > 300:
        warning = "入力値が学習データの範囲(300㎡以下)を超えているため、予測精度が低い可能性があります。"

    price = predict_price(land_area, floor_area, building_age, station_minutes, ward, district)

    return render_template(
        "predict_result.html",
        price=price, warning=warning,
        land_area=land_area, floor_area=floor_area,
        building_age=building_age, station_minutes=station_minutes,
        ward=ward, district=district,
    )


if __name__ == "__main__":
    app.run(debug=True)