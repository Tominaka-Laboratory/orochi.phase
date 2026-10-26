import os
import time
import numpy as np
import re


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


def extract_subgroup_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    subgroups_div = soup.find('div', id='subgroups')
    
    extracted_data = []
    
    if not subgroups_div:
        return extracted_data

    # すべての tr を取得し、空行やヘッダーっぽいものをスキップする
    rows = subgroups_div.find_all('tr')
    
    for row in rows:
        cells = row.find_all('td')
        # tdが複数ある行を探す（データ行は通常 5〜6列あります）
        if len(cells) >= 3:
            # セルからテキストを取得
            row_texts = [cell.get_text(strip=True) for cell in cells]
            
            # 行列情報が含まれる列を探す (「a,」や「b,」が含まれる列)
            # スペースグループ 1 の場合、4番目か5番目のインデックスにデータがあることが多い
            for text in row_texts:
                if 'a' in text and (',' in text or ';' in text):
                    extracted_data.append({
                        "raw_text": text,
                        "row_data": row_texts
                    })
                    break
    return extracted_data

def process_extracted_data(extracted_list):
    processed_database = []
    
    for entry in extracted_list:
        trans_str = entry['raw_text'].replace(' ','')
        group_info = entry['row_data'][0] 
        
        # 1. Index抽出 ([数字])
        index_match = re.search(r'\[(\d+)\]', group_info)
        index = int(index_match.group(1)) if index_match else None
        
        # 2. Subgroup番号抽出 (カッコ内の最後の数字)
        subgroup_no = None
        match = re.search(r'\((.*?)\)', group_info)
        if match:
            # カンマで区切って最後の要素を取得
            parts = match.group(1).split(',')
            subgroup_no = parts[-1].strip()
        
        # 3. Name抽出 (カッコより前の部分)
        clean_name = re.sub(r'\[.*?\]', '', group_info)
        clean_name = re.sub(r'\(.*?\)', '', clean_name).strip()
        
        # 行列とシフトの計算
        matrix, shift = parse_transformation(trans_str)
        
        processed_database.append({
            "subgroup_name": clean_name,
            "index": index,
            "subgroup_no": subgroup_no,
            "transformation_str": trans_str,
            "matrix": matrix,
            "origin_shift": shift
        })
        
    return processed_database

def parse_transformation(trans_str):
    """
    'a-b, a+b, 2c; 0, 0, 1/2' のような文字列を変換
    """
    parts = trans_str.split(';')
    matrix_part = parts[0].strip()
    shift_part = parts[1].strip() if len(parts) > 1 else "0,0,0"

    # 行列 P の構築
    matrix = np.zeros((3, 3))
    axes = matrix_part.split(',')
    for col_idx, axis_expr in enumerate(axes):
        # 変数 a, b, c それぞれの係数を取得
        for row_idx, var in enumerate(['a', 'b', 'c']):
            # パターン: 符号、数字、分数、変数の順に検索
            pattern = rf'([+-]?\s*\d*(?:/\d+)?)\s*{var}'
            match = re.search(pattern, axis_expr)
            if match:
                coeff_str = match.group(1).replace(' ', '')
                if coeff_str in ['', '+']: coeff = 1.0
                elif coeff_str == '-': coeff = -1.0
                elif '/' in coeff_str:
                    n, d = coeff_str.split('/')
                    coeff = float(n) / float(d)
                else: coeff = float(coeff_str)
                matrix[row_idx, col_idx] = coeff
    
    # 原点シフト s の構築
    shift = []
    for s_val in shift_part.split(','):
        s_val = s_val.strip().replace(' ', '')
        if '/' in s_val:
            n, d = s_val.split('/')
            shift.append(float(n) / float(d))
        else:
            shift.append(float(s_val))
            
    return matrix.tolist(), shift


driver = get_driver()
# 情報を保持するリスト
subgroup_dict = {}
try:
    for sgn in range(1, 231):
        # ページのURL
        url = f"https://symmdb.iucr.org/group/{sgn}/subgroups"
        driver.get(url)
        print(url)
        
        # 読み込みを確実にするための待機（必要に応じて調整してください）
        time.sleep(1) 
        
        data = extract_subgroup_data(driver.page_source)
        json_data = process_extracted_data(data)
        subgroup_dict[str(sgn)] = json_data

    output_file = os.path.join(os.path.dirname(__file__), "subgroup_table.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(subgroup_dict, f, indent=4, ensure_ascii=False)
except Exception as e:
    print(f"エラーが発生しました: {e}")
finally:       
    driver.quit()
    
