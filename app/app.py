from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route("/")
def route():
    table_category = pd.read_csv("app/data/table_category.csv")
    table_ward = pd.read_csv("app/data/table_ward.csv")
    table_line = pd.read_csv("app/data/table_line.csv")
    table_station = pd.read_csv("app/data/table_station.csv")
    table_floor_plan = pd.read_csv("app/data/table_floor_plan.csv")
    return render_template("index.html",
                            table_category=table_category,
                            table_ward=table_ward,
                            table_line=table_line,
                            table_station=table_station,
                            table_floor_plan=table_floor_plan
                            )

@app.route("/index", methods=["post"])
def index():

    # formの入力をdictで管理
    predict_data = {
        "distance_from_nearest_station": float(request.form["distance_from_nearest_station"]),
        "age": int(request.form["age"]),
        "total_number_of_floors": int(request.form["total_number_of_floors"]),
        "exclusive_area": float(request.form["exclusive_area"]),
        "category": int(request.form["category"]),
        "ward": int(request.form["ward"]),
        "line": int(request.form["line"]),
        "station": int(request.form["station"]),
        "floor_plan": int(request.form["floor_plan"]),
    }

    # LightGBM
    clf = pickle.load(open('app/data/trained_LGBM.pkl', 'rb'))
    before_predict=np.array(list(predict_data.values())).reshape(1,-1)
    pred = clf.predict(before_predict)[0]
    pred_yachin = round(10**pred, 1)

    # predict_dataの値を書き直す
    table_category = pd.read_csv("app/data/table_category.csv")
    table_ward = pd.read_csv("app/data/table_ward.csv")
    table_line = pd.read_csv("app/data/table_line.csv")
    table_station = pd.read_csv("app/data/table_station.csv")
    table_floor_plan = pd.read_csv("app/data/table_floor_plan.csv")
    for num, (key, item) in enumerate(predict_data.items()):
        if num < 4:
            continue
        exec(f"row_num = list(zip(*np.where(table_{key}['num'] == {item})))[0][0]")
        exec(f"predict_data['{key}'] = table_{key}.loc[row_num,'name']")

    # 日本語に直すための対応表
    eng_jap = {
         "distance_from_nearest_station": "最寄駅からの距離",
        "age": "築年数",
        "total_number_of_floors": "総階数",
        "exclusive_area": "専有面積",
        "category": "カテゴリ",
        "ward": "市区町村",
        "line": "路線",
        "station": "駅",
        "floor_plan": "間取り",
    }

    return render_template("result.html", predict_data=predict_data, pred_yachin=pred_yachin, eng_jap=eng_jap)
