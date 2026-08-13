# DenseNet121 + 單 token 輕量 ViT 重現

[![Paper](https://img.shields.io/badge/paper-IEEE_Access-00629B)](https://doi.org/10.1109/ACCESS.2026.3707422)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![Data](https://img.shields.io/badge/data-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![Reproduction](https://img.shields.io/badge/accuracy_gap-±1.2pp-2ea44f)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

重現的論文:
> Hasanah, Liu, Azmi, "A Lightweight Single-Token CNN-Transformer Architecture
> for Robust Multi-Crop Plant Disease Classification", IEEE Access, 2026.
> 沒有找到官方程式碼;這是從論文的公式 (1)-(22) 和 Figure 1-4 獨立重新實作
> 的 clean-room 版本。

## 重現結果

| 資料集 | 類別數 | 論文 Accuracy | 重現 Accuracy |
|---|---|---|---|
| Corn leaf | 4 | 97.50% | 96.30% |
| Tomato leaf † | 10 | 99.60% | 99.30% |
| BananaLSD | 4 | 98.94% | 98.40% |
| MangoLeafBD | 8 | 99.88% | **100.00%** ‡ |
| Groundnut leaf | 6 | 99.66% | 99.47% |

五個資料集全部都落在論文結果的 1.2 個百分點以內——是這次做的所有重現裡
整體對得最準的一次。可訓練參數量:7,666,244,對比論文宣稱的約 8.37M
(低 8%)。† 見下方 Tomato 的落差說明——這裡評估用的資料集比論文引用的
規模更小。

‡ **MangoLeafBD 的 100% 不代表模型真的完美,查證後是資料切分洩漏
(data leakage) 造成的假象**:原始 Kaggle 圖片檔名是拍攝時間戳記,實際
檢查發現同一類別裡 **85.3% 的 test 圖片,跟某張 train 圖片是 5 秒內拍的**
(同一片葉子連拍好幾張,隨機切分時被拆到 train/test 兩邊)——模型「測試」
時看到的圖片其實跟訓練圖片幾乎是同一張。這不只是這次重現的問題:
`data_prep/make_splits.py` 引用的論文協定原文是「在 image 層級做切分」,
如果論文原始用的 Kaggle 資料集也是同樣連拍性質的圖片,論文自己的 99.88%
(800 張錯 1 張左右)很可能也受同樣的洩漏影響,只是程度剛好比我們的
100% 略低——兩者都不能當作模型真實泛化能力的可靠指標。要修正的話得先按
拍攝時間/檔名分組再切分(同一組連拍全部分到同一邊),但這樣切出來的
train/test 就不是論文原文描述的協定了,屬於「已知偏差」而非單純的落差。

## 資料集 — 全部 5 個都拿到了(透過 Kaggle,用使用者自己的 API token)

| 論文的資料集 | 用的 Kaggle 來源 | 論文的數量 | 我們的切分 |
|---|---|---|---|
| Corn leaf(4 類) | `smaranjitghose/corn-or-maize-leaf-disease-dataset` | 3348/840 | 3351/837(自己做的 80/20 切分,池化總數 4188——完全對上) |
| Tomato leaf(10 類) | `kaustubhb999/tomatoleaf` | 12805/3207 | 8000/2000(自己對 10,000 張平衡過的 `train/` 資料夾做 80/20 切分;這個 Kaggle 資料集現在的圖片數比論文引用的少——見下方說明) |
| BananaLSD(4 類) | `shifatearman/bananalsd`,只用 `OriginalSet` | 749/188 | 749/188(自己做的 80/20 切分——完全對上) |
| MangoLeafBD(8 類) | `aryashah2k/mango-leaf-disease-dataset` | 3200/800 | 3200/800(自己做的 80/20 切分——完全對上) |
| Groundnut leaf(6 類) | `warcoder/groundnut-plant-leaf-data`、`Groundnut_Leaf_dataset` | 8287/2074 | 8288/2072(把資料集自己不均勻的 7910/2451 train/test 池化,重新切 80/20——總數 10,361 跟論文完全對上) |

**Tomato 的落差**:論文直接引用 `kaustubhb999/tomatoleaf`(參考文獻
[37]),這裡用的也是同一個資料集——但 Kaggle 上目前的版本只有 10,584 張圖
(10,000 張在完全平衡的 `train/` 裡 + 一個只涵蓋 10 類中 6 類、共 584 張的
不完整 `val/`),遠低於論文引用的 16,012 張。Kaggle 上的資料集,擁有者
發布後隨時可能更新/替換,沒有版本保證,所以這很可能是同一個資料集在論文
寫成之後縮水了,而不是抓錯資料集。這裡用的是乾淨、完全平衡的 `train/`
資料夾(每類 1000 張),忽略不完整的 `val/`,再自己做 80/20 切分。

## 架構(`src/model.py`)

對應論文 Section III-C 的四個模組:`DenseNetBackbone`(DenseNet121 的
`.features`,只微調 denseblock3/4,更前面的層都凍結)、`SingleTokenViT`
(一個 CNN 產生的 token + 一個 class token,一層 transformer encoder,
embed dim 64,2 個 head)、`AttentionPooling`(對這兩個 token 再做一次
獨立的 attention,mean pool 後線性擴展到 128 維)、`FusionClassifier`
(把 1024+128 拼接 -> 512 -> num_classes,dropout 0.3)。

這篇論文寫得異常精確(全篇都有明確的公式和張量形狀),但有兩處參數量的
內部矛盾,都詳細記錄在 `model.py` 的 module docstring 裡:

1. **Patch embedding**:公式 (4) 描述的是在 1024-channel 特徵圖上做一個
   字面意義的 7×7/stride-7 conv,光這一層就要花費約 3.2M 參數——是論文
   自己宣稱「整個 ViT 模組只有約 0.15M」(Section 5)的 20 倍以上。這裡
   改成 global-average-pool + Linear(1024,64)(約 65.6K 參數),讓整個
   ViT 模組落在約 99K,更接近論文宣稱的預算。
2. **Fusion classifier**:公式 (22) 明確給出的 W1 形狀(512×1152)光是
   這一層就要花費約 590K 參數,已經是論文宣稱「fusion classifier 約
   0.23M」(Section 5)的 2.5 倍。這裡保留公式明確給出的形狀(比一個四捨
   五入的總結數字更權威),而不是把它縮小去湊那個數字。

最終結果:**總參數量 7,666,244**(可訓練 6,236,612,denseblock1/2
凍結),對比論文宣稱的 8.37M——差距約 8%,因為上面兩處修正剛好往相反
方向拉。

## 訓練(`src/train.py`、`src/run_all.py`)

完全照 Section III-B/E、Table 3 的規格:AdamW,固定 lr=1e-5,batch_size=32,
35 epochs,沒有驗證集切分、沒有 early stopping(論文對每個資料集都是跑
固定 epoch 數的單一一次)。前處理:resize 到 224×224,每個 channel 用
mean=0.5/std=0.5 normalize(對應到 [-1,1] 區間)。只在訓練集做資料增強:
隨機水平翻轉 + 隨機旋轉(論文只說「random horizontal flipping and random
rotation」,沒給角度數值——這裡用 20°)。

`run_all.py` 會依序(由小到大:banana、mango、corn、groundnut、tomato)
跑完全部 5 個資料集,避免同時跑造成 GPU 記憶體搶用。在 RTX 4070 Ti 上估計
總共要 10-12 小時。

## 已知偏差總結

- Patch embedding 和 fusion classifier 的參數量,選擇對齊論文宣稱的總參數
  量(8.37M)——詳見 `model.py`。
- Tomato 資料集現在 Kaggle 上的圖片數比論文引用的少(見上方表格)——用的
  是目前實際能拿到的版本。
- 全程沒有用驗證集/early stopping(對應論文明確說的「固定 epoch 數跑一次」
  協定,不是我們自己加的限制)。
- 旋轉增強的角度(20°)是自己選的;論文只說「random rotation」沒給數值。

## 環境需求

Python 3.10+、PyTorch + torchvision(建議 CUDA 版)、scikit-learn、
matplotlib、`kaggle`(只有 `data_prep/` 下載資料時需要,要用你自己的 API
憑證——不包含在 repo 裡)。

## 授權

本 repo 程式碼採用 MIT 授權。不包含任何資料集圖片——Kaggle 來源見上方
資料集表格,各自有其原始授權。
