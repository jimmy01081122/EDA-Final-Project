# TSP 各類演算法 Python 實作與分析

這是一個用於研究與比較多種經典演算法在解決旅行推銷員問題（Traveling Salesman Problem, TSP）上效能的專案。專案內包含了五種不同演算法的 Python 實現，並附有對其原理和在標準 TSPLIB 測資上表現的分析。

## 專案特色

* **五種經典演算法**：涵蓋了精確演算法、啟發式、元啟發式及近似演算法。
* **簡易上手**：每種演算法均為單一 Python 腳本，依賴少，易於執行與測試。
* **標準化輸入**：所有腳本均設計為可直接讀取標準的 `.tsp` 格式檔案。
* **效能分析**：附帶的 PDF 文件詳細記錄了各演算法在不同測資下的 CPU 時間、記憶體用量、求解成本以及與官方最佳解的誤差百分比。

## 包含的演算法

本專案實作了以下五種演算法：

1.  **Held-Karp 動態規劃 (Dynamic Programming, DP)**
    * 一種**精確演算法**，保證能找到最優解。
    * 由於其 `O(n² * 2^n)` 的複雜度，僅適用於 `n < 25` 的小型問題。

2.  **模擬退火 (Simulated Annealing, SA)**
    * 一種**元啟發式演算法**，模擬固體退火過程。
    * 透過「溫度」參數控制，允許在搜索初期跳出局部最優解，平衡了廣度與深度搜索。

3.  **蟻群最佳化 (Ant Colony Optimization, ACO)**
    * 一種**元啟發式演算法**，模擬螞蟻覓食時透過費洛蒙尋找最短路徑的集體智慧。
    * 實作了最經典的「螞蟻系統 (Ant System)」模型。

4.  **遺傳演算法 (Genetic Algorithm, GA)**
    * 一種**元啟發式演算法**，借鑑生物界的「物競天擇」與遺傳機制。
    * 透過選擇、交叉、突變等操作，迭代演化出更優的解族群。

5.  **最小生成樹近似演算法 (MST-based Approximation)**
    * 一種**近似演算法**，利用最小生成樹（MST）來快速建構一條路徑。
    * 對於滿足三角不等式的度量 TSP，能保證其解的成本不超過最優解的 2 倍。

## 如何使用

### 1. 環境準備

首先，您需要安裝 Python 以及本專案所需的函式庫。

```bash
pip install tsplib95 numpy psutil
```

* `tsplib95`: 用於解析 `.tsp` 格式的標準函式庫。
* `numpy`: 用於高效的數值與矩陣運算。
* `psutil`: 用於精確測量 CPU 與記憶體使用量。

### 2. 準備測資

從 [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/) 等來源下載您想測試的 `.tsp` 檔案，並將其放置於與 Python 腳本相同的目錄下。

### 3. 執行腳本

打開終端機，使用 `python` 指令執行對應的演算法腳本。大部分腳本的設計是直接在程式碼中修改要執行的 `TSP_FILE` 變數。

**範例 (以蟻群演算法為例):**

假設您已下載 `ch130.tsp`，您可以修改 `aco_tsp.py` (假設檔案名稱如此) 中的檔案路徑變數，然後執行：

```bash
python aco_tsp.py
```

程式將會輸出求解的路徑、總成本、執行時間以及與已知最佳解的比較。

## 檔案結構說明 (推測)

* `dp.py`: Held-Karp 動態規劃的實作。
* `sa.py`: 模擬退火演算法的實作。
* `ant.py`: 蟻群最佳化演算法的實作。
* `ga.py`: 遺傳演算法的實作。
* `mst.py`: 最小生成樹近似演算法的實作。
* `*.pdf`: 各演算法的詳細實驗數據與原理分析報告。
* `*.tsp`: 從 TSPLIB 下載的測試案例檔案。
* `*.opt.tour`: (若有) 對應測試案例的官方最佳路徑檔案，用於計算誤差。
