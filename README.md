# GAN_zoo
### 環境
pytorch 1.9.0
## 注意
各モデルの学習には学習途中の画像を保存するためのlogsディレクトリ，及び学習途中のパラメータを保存するためのparamsディレクトリが作成されます．
すでに同名のディレクトリが存在している場合，内容物が上書きされるため注意してください．
## 使用方法
### SAGAN
```python
from SAGAN import SAGAN
from gan_modules import DataLoader 

sagan = SAGAN(
    n_dims=128,                     #潜在変数の次元数
    n_dis=2,                        #Generatorの更新1回に対して何回Discriminatorを更新するか
    max_resolution=128,             #生成したい画像の解像度
    g_lr = 1e-5,                    #Generatorの学習率
    d_lr = 3e-5,                    #Discriminatorの学習率
    g_betas=(0,0.999),              #Generaotrで使用するAdamの移動平均で使用するパラメータ
    d_betas=(0,0.999),              #Discriminatorで使用するAdamの移動平均で使用するパラメータ
    initial_layer="linear",         #Generatorの入力層で使用する層　全結合層:linear 転置畳み込み層:tconv
    upsampling_mode="nearest",      #Generatorの中間層でアップサンプリングする際の方法 Nearest upsampling:nearest 転置畳み込み層:tconv
    downsampling_mode="pooling",    #Discriminatorの中間層で使用するダウンサンプリング層 Average pooling:pooling 畳み込み層:conv
    loss="hinge",                   #使用する損失関数 Hinge loss:hinge Wasserstein loss with gradient penalty: wasserstein
    )
dataset = DataLoader("path of your data set folder",resolution=128) #resolution -> 生成したい画像の解像度

#学習
#is_tensorboard -> tensorboardを使用するかどうか 
#image_num -> 1epoch中に何枚の生成画像を出力するか
sagan.fit(dataset,epochs=30,batch_size=32,is_tensorboard=True,image_num=10)

#画像生成
generated_image = sagan.generate(image_num=16) 
#return torch tensor (generated_image.shape->(image_num,3,max_resolutions,max_resolutions))

```

### PGGAN

```python
from PGGAN import PGGAN
from gan_modules import DataLoader

pggan = PGGAN(
    n_dims=512,             #潜在変数の次元数
    n_dis=1,                #Generatorの更新1回に対して何回Discriminatorを更新するか
    max_resolution=256,     #生成したい画像の解像度
    g_lr=1e-3,              #Generatorの学習率
    d_lr=2e-3,              #Discriminatorの学習率
    d_betas=(0,0.99),       #Generaotrで使用するAdamの移動平均で使用するパラメータ
    g_betas=(0,0.99),       #Discriminatorで使用するAdamの移動平均で使用するパラメータ
    negative_slope=0.2,     #中間層で使用するLeaky ReLuの係数
    is_spectral_norm=False, #Equalized Learning Rateの代わりにSpectral Normalizationを使用するかどうか
    is_moving_average=True, #Generatorに対して指数移動平均を適用するかどうか
    loss="wasserstein"      #使用する損失関数 Hinge loss:hinge Wasserstein loss with gradient penalty: wasserstein
)
dataset = DataLoader("path of your data set folder",resolutions=256) #resolutions -> 生成したい画像の解像度

#学習
#is_tensorboard -> tensorboardを使用するかどうか 
#image_num -> 1epoch中に何枚の生成画像を出力するか
#epochs,batch_size -> int型が渡された場合，すべての解像度で同じ値を使用します
#またlist型のオブジェクトを渡すことで各解像度ごとに値を設定できます
#この際，最大解像度に至るまでの解像度の段階とlistの長さが一致しない場合，それ以降はリストの最後の要素を使用します
#pggan.fit(dataset,epochs=5,batch_size=16,is_tensorboad=False,image_num=100) int型で渡した場合
#pggan.fit(dataset,epochs=[1,2],batch_size=[2,3],is_tensorboad=False,image_num=100) listの長さと解像度の段階が一致しない場合
pggan.fit(dataset,epochs=[1,2,3,4,5,6,7],batch_size=[2,3,5,7,11,13,17],is_tensorboad=False,image_num=100) #list型を渡した場合

#画像生成
generated_image = pggan.generate(image_num=16) 
#return torch tensor (generated_image.shape->(image_num,3,max_resolutions,max_resolutions))
```
### LightWeight GAN
Under development
