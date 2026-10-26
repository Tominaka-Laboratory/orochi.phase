import sqlite3
import json
import sympy

# SymPy変数の定義（ソース[1]に基づき、文字列化のために必要）
x, y, z = sympy.symbols("x y z")

# --- データの読み込み ---
# ここでは source から提供された wyckoff_dictionary を使用すると仮定します
# 実際の実行時は、wyckoff.txt をインポートするか exec() で展開してください。

from wyckoff import wyckoff_dictionary 

def migrate_to_sqlite(db_path, data_dict):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # テーブルの作成
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wyckoff_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            space_group TEXT,
            setting TEXT,
            multiplicity INTEGER,
            wyckoff_letter TEXT,
            point_symmetry TEXT,
            coordinates TEXT
        )
    ''')

    # 高速化のためのインデックス作成
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sg_setting ON wyckoff_positions(space_group, setting)')

    for sg, settings in data_dict.items():
        for setting_key, redundant_list in settings.items():
            # 冗長な階層 [[ [Site1], [Site2] ]] を Siteリストへ展開 [2]
            # ソースによりネストの深さが異なる可能性があるため、慎重に取り出し
            try:
                # 三重リストの場合: redundant_list がワイコフ位置のリスト
                sites = redundant_list[0]
            except (IndexError, TypeError):
                # 二重リストの場合: redundant_list がリスト
                sites = redundant_list

            for site in sites:
                multiplicity, letter, symmetry, coords = site
                
                # 座標(SymPyオブジェクトを含む)を文字列/JSON形式に変換
                # 検索時に使いやすいよう、str(coords) または json化
                coords_str = str(coords)

                cursor.execute('''
                    INSERT INTO wyckoff_positions 
                    (space_group, setting, multiplicity, wyckoff_letter, point_symmetry, coordinates)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (sg, setting_key, multiplicity, letter, symmetry, coords_str))

    conn.commit()
    conn.close()
    print(f"Migration completed: {db_path}")

migrate_to_sqlite("wyckoff.db", wyckoff_dictionary)