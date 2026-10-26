import sqlite3
import spglib
import numpy as np
import re


def create_integrated_sg_db(db_path, space_group_list, hall_number_list):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # テーブル作成
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS space_groups (
            hall_number INTEGER PRIMARY KEY,
            hall_symbol TEXT,
            crystal_system TEXT,
            table_no INTEGER,
            hm_short TEXT,
            hm_full TEXT,
            setting_no TEXT, 
            origin INTEGER,
            axes TEXT,
            sch_symbol TEXT,
            pg_no INTEGER,
            pg_hm TEXT,
            pg_sch TEXT,
            lc_no INTEGER,
            lc_hm TEXT,
            ext_system TEXT,
            ext_a TEXT,
            ext_b TEXT,
            ext_c TEXT,
            rotation TEXT,
            translation TEXT
        )
    ''')

    # 
    fnmap = {}
    for row in space_group_list:
         full_notation = str(row[2])
         ext_system = str(row[3])
         ext_a = str(row[4])
         ext_b = str(row[5])
         ext_c = str(row[6])
         if full_notation not in fnmap:
             fnmap[full_notation] = {
                 'ext_system': ext_system,
                 'ext_a': ext_a,
                 'ext_b': ext_b,
                 'ext_c': ext_c
             }
         else:
              continue
         

    # 2. データの挿入
    for i, row in enumerate(hall_number_list):
        hm_full = row["hm_full"]
        ext_system = fnmap[hm_full]["ext_system"] if hm_full in fnmap else ""
        ext_a = fnmap[hm_full]["ext_a"] if hm_full in fnmap else ""
        ext_b = fnmap[hm_full]["ext_b"] if hm_full in fnmap else ""
        ext_c = fnmap[hm_full]["ext_c"] if hm_full in fnmap else ""
        axes_origin = row["axes_origin"]
        origin, axes = parse_axes_origin(axes_origin)
        
        syms = spglib.get_symmetry_from_database(row["serial_no"])
        rotation = np.array(syms['rotations']).tolist()
        translation = np.array(syms['translations']).tolist()


        d = {
            "hall_number": row["serial_no"],
            "hall_symbol": row["hall_symbol"],
            "crystal_system": row["crystal_system"],
            "table_no": row["ita_no"],
            "hm_short": row["hm_short"],
            "hm_full": row["hm_full"],
            "setting_no": row["sub_no"],
            "origin": origin,
            "axes": axes,
            "sch_symbol": row["sch_symbol"],
            "pg_no": row["pg_no"],
            "pg_hm": row["pg_hm"],
            "pg_sch": row["pg_sch"],
            "lc_no": row["lc_no"],
            "lc_hm": row["lc_hm"],
            "ext_system": ext_system,
            "ext_a": ext_a,
            "ext_b": ext_b,
            "ext_c": ext_c,
            "rotation": str(rotation),
            "translation": str(translation)         
        }
        for key, val in d.items():
            if val is None:
                d[key] = ""

        columns = ", ".join(d.keys())
        placeholders = ", ".join([":" + k for k in d.keys()])

        cursor.execute(f'''
                INSERT INTO space_groups ({columns}) 
                VALUES ({placeholders})
            ''', d)
        
    conn.commit()
    conn.close()
    print(f"Integrated database created: {db_path}")

def parse_axes_origin(txt):
    if not txt or not isinstance(txt, str):
        return 1, "a,b,c"  # デフォルト値を返す

    # 1. 前後の余白を消し、特殊なバー記号を標準のマイナスに置換
    # ¯ (オーバーライン) を - に変える
    txt = txt.replace('¯', '-').strip()

    # 2. 先頭の数字 (Origin) を取り出す。なければ None
    # ^(\d)? : 行頭に数字があればグループ1へ（なければ空）
    # \s*    : 数字の後のスペースを飛ばす
    # (.*)   : 残りすべてをグループ2（軸部分）
    match = re.match(r'^(\d)?\s*(.*)$', txt)
    
    origin_choice = 1  # デフォルトは設定1
    axes_part = ""

    if match:
        # 数字が取得できれば上書き
        if match.group(1):
            origin_choice = int(match.group(1))
        axes_part = match.group(2).strip()
    else:
        axes_part = txt # マッチしない場合は全体を軸とする

    # 3. 軸部分の整形 (b a -c -> b,a,-c)
    # アルファベットの「後ろ」にマイナスが残っている場合の最終補正
    # (例: "c-" を "-c" に直す)
    axes_part = re.sub(r'([a-c])\s*-', r'-\1', axes_part)
    
    # スペースやセミコロンをカンマに統一
    axes_part = re.sub(r'[\s;]+', ',', axes_part)
    # 連続したカンマを1つにする
    axes_part = re.sub(r',+', ',', axes_part).strip(',')

    # 軸が空（数字だけ入力された場合など）はデフォルトに戻す
    if not axes_part:
        axes_part = "a,b,c"

    return origin_choice, axes_part

import numpy as np
import sympy

def create_integrated_pg_db(db_path, point_group_dict):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # テーブル作成
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS point_groups (
            id INTEGER PRIMARY KEY,
            hm_symbol TEXT,
            scm_symbol TEXT,
            operations TEXT,
            rotations TEXT
        )
        ''')
         
    _s_to_hm = {
        "C1": "1",    "Ci": "-1",   "C2": "2",    "Cs": "m",    "C2h": "2/m",
        "D2": "222",  "C2v": "mm2", "D2h": "mmm", "C4": "4",    "S4": "-4",
        "C4h": "4/m", "D4": "422",  "C4v": "4mm", "D2d": "-42m", "D4h": "4/mmm",
        "C3": "3",    "C3i": "-3",  "S6": "-3",   "D3": "32",   "C3v": "3m",  "D3d": "-3m",
        "C6": "6",    "C3h": "-6",  "C6h": "6/m", "D6": "622",  "C6v": "6mm", "D3h": "-6m2", "D6h": "6/mmm",
        "T": "23",    "Th": "m-3",  "O": "432",   "Td": "-43m", "Oh": "m-3m"
    }

    hm_to_s = {v: k for k, v in _s_to_hm.items()}

    # 2. データの挿入
    for hm_symbol, operations in point_group_dict.items():
        x, y, z = sympy.symbols("x y z")
        basis = [x, y, z]
        matrices = []
        for op in operations:
            # linear_eq_to_matrixで回転行列 A と 定数ベクトル b を取得
            # 点群操作なので b は必ず 0 になる
            A, b = sympy.linear_eq_to_matrix(op, basis)
            matrices.append(np.array(A.tolist(), dtype=np.float64))
        matrices = np.array(matrices).reshape(-1, 3, 3).tolist()

        d = {
            "hm_symbol": hm_symbol,
            "scm_symbol": hm_to_s[hm_symbol],
            "operations": str(operations),
            "rotations": str(matrices)      
        }
        for key, val in d.items():
            if val is None:
                d[key] = ""

        columns = ", ".join(d.keys())
        placeholders = ", ".join([":" + k for k in d.keys()])

        cursor.execute(f'''
                INSERT INTO point_groups ({columns}) 
                VALUES ({placeholders})
            ''', d)
        
    conn.commit()
    conn.close()
    print(f"Integrated database created: {db_path}")


from database.spacegroup import space_group_table
import os
import json
hall_file = os.path.join(os.path.dirname(__file__), "hall_number_table.json")
with open(hall_file, 'r', encoding='utf-8') as f:
    hall_number_list = json.load(f)


# 実行例（space_group_table は spacegroup.txt から読み込んだnp.arrayやlist）
save_file = os.path.join(os.path.dirname(__file__), "spacegroup.db")
create_integrated_sg_db(save_file, space_group_table, hall_number_list)

from database.point_group import point_group_dictionary
create_integrated_pg_db(save_file, point_group_dictionary)


