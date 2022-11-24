from flask import Flask, render_template, request, current_app, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

import pickle
import numpy as np
import pandas as pd
from copy import copy

app = Flask(__name__)


@app.route("/")
def index():
    return redirect(url_for("predict"))

@app.route("/predict")
def predict():
    table_category = pd.read_csv("apps/data/table_category.csv")
    table_ward = pd.read_csv("apps/data/table_ward.csv")
    table_line = pd.read_csv("apps/data/table_line.csv")
    table_station = pd.read_csv("apps/data/table_station.csv")
    table_floor_plan = pd.read_csv("apps/data/table_floor_plan.csv")
    return render_template("predict.html",
                            table_category=table_category,
                            table_ward=table_ward,
                            table_line=table_line,
                            table_station=table_station,
                            table_floor_plan=table_floor_plan
                            )

@app.route("/predict_result", methods=["post"])
def predict_result():

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
    clf = pickle.load(open('apps/data/trained_LGBM.pkl', 'rb'))
    before_predict=np.array(list(predict_data.values())).reshape(1,-1)
    pred = clf.predict(before_predict)[0]
    pred_yachin = round(10**pred, 1)

    # predict_dataの値を書き直す
    table_category = pd.read_csv("apps/data/table_category.csv")
    table_ward = pd.read_csv("apps/data/table_ward.csv")
    table_line = pd.read_csv("apps/data/table_line.csv")
    table_station = pd.read_csv("apps/data/table_station.csv")
    table_floor_plan = pd.read_csv("apps/data/table_floor_plan.csv")
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

    return render_template("predict_result.html", predict_data=predict_data, pred_yachin=pred_yachin, eng_jap=eng_jap)


@app.route("/recommend")
def recommend():
    transform_floor_plan_area = pd.read_csv("./apps/data/transform_floor_plan_area.csv")
    land_price = pd.read_csv("./apps/data/land_price.csv")
    return render_template("recommend.html",
        transform_floor_plan_area=transform_floor_plan_area,
        land_price = land_price)

@app.route("/recommend_result", methods=["post"])
def recommend_result():
    # フォームデータを格納
    ideal_rent = {
        "家賃": float(request.form["rent"]),
        "区": request.form["ward"],
        "駅徒歩": int(request.form["time_to_station"]),
        "間取り": request.form["floor_plan"],
        "総階数": int(request.form["top_floor"]),
        "地上": int(request.form["top_floor"]),
        "階": int(request.form["rent_floor"]),
        "築年数": int(request.form["age"]),
    }
    print(ideal_rent)

    # 駅徒歩を変換
    ideal_rent['最寄駅からの距離'] = ideal_rent['駅徒歩']*100

    # 間取りを変換
    # その場で計算してもいいけどテーブルを用意してもいいかもね
    transform_floor_plan_area = pd.read_csv("./apps/data/transform_floor_plan_area.csv")
    condition = transform_floor_plan_area['間取り'] == ideal_rent['間取り']
    ideal_rent['面積'] = transform_floor_plan_area.loc[condition,'面積'].values[0]

    # 地価を変換
    # その場で計算してもいいけどテーブルを用意してもいいかもね
    land_price = pd.read_csv("./apps/data/land_price.csv")
    condition = land_price['区'] == ideal_rent['区']
    ideal_rent['地価'] = land_price.loc[condition,'地価'].values[0]

    # 家賃を変換
    ideal_rent['家賃'] = np.log10(ideal_rent['家賃'])

    # 整形
    df_ideal = pd.DataFrame(ideal_rent.values()).T
    df_ideal.columns = ideal_rent.keys()

    # 物件データ
    suumo_recomm = pd.read_csv("./apps/data/suumo_recomm.csv")
    # 変数
    features = [
        '最寄駅からの距離',
        '面積',
        '築年数',
        '地上',
        '階',
        '地価',
        '家賃',
    ]

    # 標準化
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    ss.fit(suumo_recomm[features])
    suumo_recomm_ss = copy(suumo_recomm)
    suumo_recomm_ss[features] = ss.transform(suumo_recomm[features])

    # 標準化を適用
    df_ideal_ss = copy(df_ideal)
    df_ideal_ss[features] = ss.transform(df_ideal[features])

    # コサイン類似度
    def cos_sim(v1,v2):
        dot = np.dot(v1,v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        sim = dot/(norm1*norm2)
        return sim

    # 物件の類似度を計算
    vec_sim = suumo_recomm_ss.apply(lambda x: cos_sim(x[features],df_ideal_ss.loc[0,features]),axis=1)

    # 一番類似している物件を3件
    num = vec_sim.nlargest(3).index.tolist()
    max_ = vec_sim.nlargest(3).values.tolist()

    # 変形したデータをもとに戻す
    # フォームデータ
    ideal_rent['家賃'] = (10**ideal_rent['家賃']).round(1)

    # 物件データ
    recomm_rent = ss.inverse_transform(suumo_recomm_ss.loc[num,features])
    recomm_rent = pd.DataFrame(recomm_rent,index=num, columns=features)

    # 面積→間取り
    recomm_rent['間取り'] = recomm_rent['面積'].map(lambda x: transform_floor_plan_area.loc[transform_floor_plan_area['面積'] == x,'間取り'].values[0])

    # 地価→区
    recomm_rent['区'] = recomm_rent['地価'].map(lambda x: land_price.loc[land_price['地価'] == x,'区'].values[0])

    # 最寄駅からの距離→駅徒歩
    recomm_rent['駅徒歩'] = (recomm_rent['最寄駅からの距離']/100).astype(int)

    # 家賃をもとに戻す
    recomm_rent['家賃'] = (10**(recomm_rent['家賃'])).round(1)

    # 不要な要素を消す
    del ideal_rent["最寄駅からの距離"],ideal_rent["地価"],ideal_rent["面積"],ideal_rent['地上']
    recomm_rent.drop(["最寄駅からの距離","地価","面積"], axis=1, inplace=True)

    # 名前を変える
    recomm_rent.rename(columns={"地上":"総階数"}, inplace=True)

    # 順番調整
    recomm_rent = recomm_rent[ideal_rent.keys()]

    # 型変換
    recomm_rent["築年数"] = recomm_rent["築年数"].astype(int)
    recomm_rent["総階数"] = recomm_rent["総階数"].astype(int)
    recomm_rent["階"] = recomm_rent["階"].astype(int)

    # 類似度を与える
    recomm_rent['類似度'] = np.array(max_).reshape(-1,1)
    recomm_rent["類似度"] = recomm_rent["類似度"].round(3)

    # urlを与える
    recomm_rent['url'] = suumo_recomm_ss.loc[num,'url']

    return render_template("recommend_result.html",
        ideal_rent=ideal_rent,recomm_rent=recomm_rent)
