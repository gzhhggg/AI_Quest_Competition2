# AI_Quest_Competition_2

タイトル：「PBL_04 工数予測（印刷業）AI課題」  

## ストーリー  
ABC印刷株式会社では手作業で印刷物をつくるために必要な機械の稼働時間を予想している。  
それにより、予想結果誤差が大きかったり納期遅れが発生している。  
データセットを用いて印刷工数の予測が出来るAIモデルを構築・分析して結果を提出する。

## 課題内容  
<img src="https://user-images.githubusercontent.com/93046615/164146211-dd059486-ed97-4790-8d5c-85494a0ae75e.png" width="900px">

## 学習用

- base_train.csv
  - 行数: 16913
  - 列数: 80
  - 主なカラム
    - 受注番号: 受注時に振り分けられる番号
    - 受注日: 受注した日にち
    - 受注数量: 受注した数量
- processing_train.csv
  - 行数: 22095
  - 列数: 67
  - 主なカラム
    - 受注番号: 受注時に振り分けられる番号
    - 号機名: 作業を行う機械
    - 数量1~4: 実際に作成する生産物の数量
    - 数量項目名1~4: 対応する生産物の数量の単位
- actual_train.csv
  - 行数: 110953
  - 列数: 37
  - 主なカラム
    - 受注番号: 受注時に振り分けられる番号
    - 号機名: 作業を行う機械
    - 作業日: 実際に作業を行う日にち
    - 所要時間: 作業が始まってから終わるまでの時間(分)。所要時間=作業時間+合計時間+残業時間となる。
    - 作業時間: 機械が生産を行うのにかかる時間(分)
    - 合計時間: 生産を行うための準備にかかる時間(分) ※作業日が2020/02/04から定義される。2020/02/03までは意味のないデータ。
    - 残業時間: 問い合わせ対応などの生産に直接関わらない時間(分)

## 評価用

- base_test.csv
  - 行数: 5073
  - 列数: 80
  - 主なカラム
    - 受注番号: 受注時に振り分けられる番号
    - 受注日: 受注した日にち
    - 受注数量: 受注した数量
- processing_test.csv
  - 行数: 5983
  - 列数: 67
  - 主なカラム
    - 受注番号: 受注時に振り分けられる番号
    - 号機名: 作業を行う機械
    - 数量1~4: 実際に作成する生産物の数量
    - 数量項目名1~4: 対応する生産物の数量の単位
- actual_test.csv
  - 行数: 5983
  - 列数: 4
  - 主なカラム
    - index: 応募用サンプルファイルと紐づくインデックス
    - 受注番号: 受注時に振り分けられる番号
    - 号機名: 作業を行う機械
    - 作業日: 実際に作業を行う日にち

# 検討
目的変数は正味作業時間と付帯作業時間の為、最低2つのモデルを作成する必要がある   
正味作業時間は相関が強い特徴量があるが、付帯作業時間にはどの特徴量も相関なし  
学習用のデータ、評価用のデータには欠損値がとても多い、必要なカラムを絞って学習モデルを作成する  
processing.csvとatual.csvはほとんどが欠損値  
グルアーと印刷機でデータの内容が異なるため作成するモデルは下記の４つとする
* グルアー正味作業時間
* グルアー付帯作業時間
* 印刷機（2, 4, 6, 7, 8号機)正味作業時間
* 印刷機（2, 4, 6, 7, 8号機)付帯作業時間


# 作成モデルについて  
submition.ipynb  
### 1.ライブラリのimport  
```bash
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import japanize_matplotlib
sns.set(font="IPAexGothic")
%matplotlib inline
# 前処理
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# LightGBM
import optuna.integration.lightgbm as lgb

# 評価指標
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
```
### 2.学習データ評価データの前処理  
欠損値が多いカラムの削除、値がマイナスのデータの０埋めを行う  
2020/4/20から付帯時間の定義が変わっているので同じ算出方法にそろえる  
```bash
def preprocessing_train(actual_train,processing_train,base_train):
    train_merged = pd.merge(processing_train,actual_train,on =["受注番号","号機名"])
    train_merged = pd.merge(train_merged,base_train,on =["受注番号"])
    train_merged["作業日"] = pd.to_datetime(train_merged["作業日"])
    train_merged = train_merged[train_merged["号機名"].isin(["グルアー","2号機","4号機","6号機","7号機","8号機"])]
    train_merged["付帯時間"] = train_merged["合計時間"].where(train_merged["作業日"] >= "2020-02-04",train_merged["所要時間"] - (train_merged["作業時間"] + train_merged["残業時間"]))
    train_merged = train_merged[(train_merged['付帯時間']>0)&(train_merged['作業時間']>0)] 
    train_df = train_merged[["受注番号","号機名","数量1","合計数量","予備数量","仕上数量","連量","加工数量","製品仕様コード_x","カテゴリ名1","カテゴリ名2","表色数","裏色数","流用受注番号","展開寸法幅","展開寸法長さ","作業時間","付帯時間"]]
    return train_df
```
### 3.外れ値の削除  
欠損値が多いため、中心値から95％以上の外れ値を削除する
```bash
work_time_quantile = df_train["作業時間"].quantile(0.95)
hutai_time_quantile = df_train["付帯時間"].quantile(0.95)
df_train = df_train.query('作業時間 < @work_time_quantile')
df_train = df_train.query("付帯時間 < @hutai_time_quantile")
df_train[["作業時間","付帯時間"]].describe()
```
Befor ⇒　After  
 <img src="https://user-images.githubusercontent.com/93046615/164148015-86f26926-7fbe-45fd-94bb-e77abaf93a6b.png" width="300px"> <img src="https://user-images.githubusercontent.com/93046615/164148134-3f0963a3-de51-43e2-8331-d60bca8964b0.png" width="300px">

### 4.グルアーと 印刷機（2, 4, 6, 7, 8号機)に分けるて欠損値穴埋め（中央値・0埋め)
グルアーと印刷機でデータを分割してそれぞれ異なる前処理を行う  
```bash
gruar = df[df["号機名"] == "グルアー"]
print = df[df["号機名"] != "グルアー"]
```
### 5.カテゴリ変数をダミー変数に変換
```bash
gruar = pd.get_dummies(gruar)
```
### 6.目的変数と説明変数に分ける  
モデルは4つ作成するため、8個のXYを作成  
```bash
x_gruar_work = train_gruar.drop(["作業時間","付帯時間"],axis=1)
x_gruar_hutai = train_gruar.drop(["付帯時間"],axis=1)
y_gruar_work = train_gruar["作業時間"]
y_gruar_hutai = train_gruar["付帯時間"]

x_print_work = train_print.drop(["作業時間","付帯時間"],axis=1)
x_print_hutai = train_print.drop(["付帯時間"],axis=1)
y_print_work = train_print["作業時間"]
y_print_hutai = train_print["付帯時間"]
```
### 7.学習データと評価データの作成
train_test_splitでデータの分割（8:2)
```bash
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=TEST_SIZE,random_state=RANDOM_STATE)
x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=TEST_SIZE,random_state=RANDOM_STATE)
```
### 8.パラメータチューニング  
最適なパラメータを調べる  
```bash
    params = {
        'objective': 'mean_squared_error',
        'metric': 'mae',
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    best_params, history = {}, []

    # LightGBM学習
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=[lgb_train, lgb_eval],
                    early_stopping_rounds=100
                )

    best_params = gbm.params
```
### 9.評価
評価して値を取得
評価指標はR2 MAE MSE RMSE を取得
今回の課題は慢性的に誤差を減らしたいためRMSEを重要視する  
<img src="https://user-images.githubusercontent.com/93046615/164150947-fb9f108a-e8f2-42d7-9fe4-8a6860d8bce3.png" width="600px">
```bash
    # 評価
    def calculate_scores(true, pred):
        """全ての評価指標を計算する

        Parameters
        ----------
        true (np.array)       : 実測値
        pred (np.array)       : 予測値

        Returns
        -------
        scores (pd.DataFrame) : 各評価指標を纏めた結果

        """
        scores = {}
        scores = pd.DataFrame({'R2': r2_score(true, pred),
                            'MAE': mean_absolute_error(true, pred),
                            'MSE': mean_squared_error(true, pred),
                            'RMSE': np.sqrt(mean_squared_error(true, pred))},
                            index = ['scores'])
        return scores
```
### 10.importanceの確認
importanceを確認して不要な特徴量がないか調べる  
<img src="https://user-images.githubusercontent.com/93046615/164149893-ee5cb1cc-2cd3-4cac-b475-22ce4c6165cf.png" width="200px"><img src="https://user-images.githubusercontent.com/93046615/164149910-d827f719-dfe3-4d5e-a5a8-22ba6e08d155.png" width="200px">

### 11.テストデータで評価
```bash
pred_gruar_hutai = gbm_gruar_hutai.predict(test_gruar)
pred_print_hutai = gbm_print_hutai.predict(test_print)
df_gruar = pd.DataFrame({"合計時間":pred_gruar_work,"付帯時間":pred_gruar_hutai})
df_print = pd.DataFrame({"合計時間":pred_print_work,"付帯時間":pred_print_hutai})
```
<img src="https://user-images.githubusercontent.com/93046615/164150140-dfc23beb-f1e2-48da-939f-9042db3ef9e8.png" width="200px">

### 12.提出用データの出力
```bash
submit.to_csv("submit.csv",index=None,header=False)
```
# Note  
本課題は、課題内容の理解に苦しんだ  
課題内容を理解するために印刷して下線を引きながら何度も読みかえしたことを覚えている  
また、学習データ評価データどちらも欠損値・異常値が多すぎる・・・あまりにも汚い    

コンペ優勝者も私と同様４つのモデルを作成していたが前処理の手法が全くことなっていた  
今回の一番の収穫は前処理の大事さを知れたことだ。  
パラメータチューニングやアンサンブル学習なんかよりも前処理が命  
とにかく前処理！！ひたすら前処理！！  

コンペの結果は42位/326  
プレゼン資料を含めた総合順位では16位になった  

# Author

* 作成者 KeiichiAdachi
* 所属 Japan/Aichi
* E-mail keiichimonoo@gmail.com
 
# License
なし  
