import os
import time
import numpy as np
import re
import pandas as pd

# ローカル通信はプロキシを通さない設定
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
# もしHTTP_PROXYなどを設定している場合は、それがChromeDriverの通信に干渉しないようにします
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

from bs4 import BeautifulSoup
import json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# 1. ChromeDriverのパスを指定 (フルパスで記載してください)
# 例: C:\\chromedriver\\chromedriver.exe
CHROMEDRIVER_PATH = r"C:\chromedriver-win64\chromedriver.exe"

# 2. Chromeブラウザ本体のパスを指定 (通常は以下です)
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def get_driver():
    options = Options()
    options.binary_location = CHROME_PATH  # Chromeの場所を明示
    options.add_argument("--headless")

    service = Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def extract_hall_number_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    columns = ["serial_no", "ita_no", "sub_no", "axes_origin", "hall_symbol", 
               "sch_symbol", "hm_short", "hm_full", "pg_no", "pg_hm", "pg_sch", "lc_no", "lc_hm"]
    
    table_data = []
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        rowspan_cells = {}
        
        for row in rows:
            # 取得したセルをすべて抽出
            cells = row.find_all('td')
            
            # --- デバッグ: なぜ抜けるかを確認 ---
            # if len(cells) < 5: continue
            # ---------------------------------
            
            current_row = []
            cell_idx = 0
            
            # 13列分のデータを強制的に埋める（セルが足りない場合は None）
            for col_idx in range(13):
                if rowspan_cells.get(col_idx, [0, None])[0] > 0:
                    current_row.append(rowspan_cells[col_idx][1])
                    rowspan_cells[col_idx][0] -= 1
                else:
                    if cell_idx < len(cells):
                        cell = cells[cell_idx]
                        # ここでセパレーターをスペースに指定して結合を解く
                        text = cell.get_text(separator=' ', strip=True)
                        current_row.append(text)
                        
                        rowspan = int(cell.get('rowspan', 1))
                        if rowspan > 1:
                            rowspan_cells[col_idx] = [rowspan - 1, text]
                        cell_idx += 1
                    else:
                        current_row.append(None)
            
            # 修正ポイント: 最初の列が数字（serial_no）であれば強制的に追加
            if current_row[0] and str(current_row[0]).strip().isdigit():
                table_data.append(current_row)
            else:
                # 数字でない行（ヘッダーや解説行）をスキップしたことをログ出力可能
                pass

    df = pd.DataFrame(table_data, columns=columns)
    
    # 1. 数値型の整数化（Nullable Integer型を使用）
    int_cols = ["serial_no", "ita_no", "sub_no", "pg_no", "lc_no"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    

        
    df = df.sort_values('serial_no').reset_index(drop=True)

    systems = [
        (1, 2, "triclinic", "-1", 1),
        (3, 107, "monoclinic", "2/m", 2),
        (108, 348, "orthorhombic", "mmm", 2),
        (349, 429, "tetragonal", None, None),
        (430, 461, "trigonal", None, None),
        (462, 488, "hexagonal", None, None),
        (489, 530, "cubic", None, None),
    ]

    df['crystal_system'] = None
    
    for start, end, sys_name, lc_name, lc_no in systems:
        mask = (df['serial_no'] >= start) & (df['serial_no'] <= end)
        df.loc[mask, 'crystal_system'] = sys_name
        if lc_name is not None:
            df.loc[mask, 'lc_hm'] = lc_name
        if lc_no is not None:
            df.loc[mask, 'lc_no'] = lc_no
        
    # 2. HM記号の表記整形
    def clean_hm(narrow=True):
        def action(text):
            if not isinstance(text, str): return text

            if '=' in text: text = text.split('=')[0]

            if not text.startswith("-"):
                separated = re.sub(r'(?<!^)(?=[A-Z])', '$', text)
            
                # 分離した結果をリスト化
                parts = separated.split('$')
                if len(parts) > 1:
                    print(parts)
            
                # 2. 複数の表記がある場合は、最後の要素（最新表記）を採用
                # 例: ["Abm2", "Aem2"] -> "Aem2"
                text = parts[-1]
            
            # バックスラッシュを削除（エスケープされた \ も含めて削除）
            text = text.replace('\\', '') 
            text = text.replace('^', '')
            text = text.replace('_', '')
            text = re.sub(r'\\bar\{(\d)\}', r'-\1', text)

            if narrow:
                text = text.replace(' ', '')
            
            # 以下、既存の処理
            text = re.sub(r'(\d)¯', r'-\1', text)
            
            return text.strip()
        return action

    for col in ['hm_short', 'hm_full', 'pg_hm', 'lc_hm', 'sch_symbol', 'pg_sch']:
        df[col] = df[col].apply(clean_hm(True))
    for col in ['hall_symbol']:
        df[col] = df[col].apply(clean_hm(False))

    return df


driver = get_driver()
# 情報を保持するリスト
hall_number_dict = {}
try:

    # ページのURL
    url = f"https://yseto.net/sg/sg1"
    driver.get(url)
    print(url)
    
    # 読み込みを確実にするための待機（必要に応じて調整してください）
    time.sleep(1) 
    
    df = extract_hall_number_data(driver.page_source)

    output_file = os.path.join(os.path.dirname(__file__), "hall_number_table.json")
    data_list = df.to_dict(orient='records')

    # 3. ファイルに書き出し
    with open(output_file, 'w', encoding='utf-8') as f:
        # json.dump の出力結果を一度文字列として生成し、replaceをかける
        json_str = json.dumps(data_list, indent=4, ensure_ascii=False)
        
        # ここで強制的にエスケープされたスラッシュを元に戻す
        final_json = json_str.replace('\\/', '/')
        
        f.write(final_json)
    
except Exception as e:
    print(f"エラーが発生しました: {e}")
finally:       
    driver.quit()
    
