import sqlite3
import json

def create_element_database(db_path="element_data.db"):
    # データベースに接続（ファイルがなければ作成される）
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. elements テーブル: 元素の基本情報
    # typical_valences は JSON文字列として保存
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS elements (
            atomic_number INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            atomic_mass REAL,
            typical_valences TEXT
        )
    ''')

    # 2. ion_radii テーブル: 半径情報
    # coordination_number が NULL の場合は典型的な半径として扱う
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ion_radii (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            atomic_number INTEGER,
            symbol TEXT NOT NULL,
            charge INTEGER,
            coordination_number INTEGER,
            radius REAL,
            is_shannon BOOLEAN,
            FOREIGN KEY (atomic_number) REFERENCES elements (atomic_number)
        )
    ''')

    # 3. bond_valence_params テーブル: BVM計算用パラメータ
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bond_valence_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            atom_a TEXT,
            valence_a INTEGER,
            atom_b TEXT,
            valence_b INTEGER,
            r0 REAL,
            b REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS element_groups (
        group_id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_name TEXT UNIQUE NOT NULL
    )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS element_group_mapping (
            atomic_number INTEGER,
            group_id INTEGER,
            PRIMARY KEY (atomic_number, group_id),
            FOREIGN KEY (atomic_number) REFERENCES elements (atomic_number),
            FOREIGN KEY (group_id) REFERENCES element_groups (group_id)
    )
    ''')
                   

    # 高速検索のためのインデックス作成
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON elements(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_radius_lookup ON ion_radii(atomic_number, charge, coordination_number)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bvm_lookup ON bond_valence_params(atom_a, atom_b)')

    conn.commit()
    return conn, cursor

def populate_elements(cursor, number_symbol, number_mass, typical_valence):
    """elementsテーブルにデータを流し込む例"""
    for num_str, symbol in number_symbol.items():
        num = int(num_str)
        mass = number_mass.get(num_str)
        valences = typical_valence.get(num_str, [])
        
        cursor.execute('''
            INSERT OR REPLACE INTO elements (atomic_number, symbol, atomic_mass, typical_valences)
            VALUES (?, ?, ?, ?)
        ''', (num, symbol, mass, json.dumps(valences)))

def populate_radii(cursor, number_radius, shannon_radii, number_symbol):
    sym_to_num = {v: int(k) for k, v in number_symbol.items()}

    # 1. 基本半径 (number_radius) の登録
    for num_str, charges in number_radius.items():
        atomic_num = int(num_str)
        symbol = number_symbol[num_str]
        
        for charge_str, rad in charges.items():
            cursor.execute('''
                INSERT INTO ion_radii (atomic_number, symbol, charge, coordination_number, radius, is_shannon)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (atomic_num, symbol, int(charge_str), None, float(rad), False))

    # 2. 詳細なシャノン半径 (shannon_radii) の登録
    for symbol, charges in shannon_radii.items():
        if symbol not in sym_to_num:
            continue
        atomic_num = sym_to_num[symbol]
        
        for charge_str, cns in charges.items():
            charge = int(charge_str)
            for cn_str, values in cns.items():
                cn = int(cn_str)
                
                # valuesの例: [ShannonRadius, MLRadius, StdDev] または [[LowSpin, HighSpin], MLRadius, StdDev]
                # ShannonRadius に相当する要素を抽出する
                
                radius_value = 0.0
                
                # 最初の要素(values[0])がリストかどうかで判断
                if isinstance(values[0], list):
                    # スピン状態のリスト: [LowSpin, HighSpin]
                    # HighSpin (index 1) を優先して採用する例
                    spin_data = values[0]
                    radius_value = float(spin_data[1]) if len(spin_data) > 1 and spin_data[1] is not None else 0.0
                else:
                    # 通常の数値
                    radius_value = float(values[0]) if values[0] is not None else 0.0
                
                # もし Shannon Radius が 0 なら ML Radius (values[1]) で補完
                if radius_value == 0.0 and len(values) > 1:
                    radius_value = float(values[1]) if values[1] is not None else 0.0

                radius_value = np.round(radius_value/100,5)
                cursor.execute('''
                    INSERT INTO ion_radii (atomic_number, symbol, charge, coordination_number, radius, is_shannon)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (atomic_num, symbol, charge, cn, radius_value, True))

def populate_bvm(cursor, bond_valence_table):
    """bond_valence.txtの情報をインポート"""
    for row in bond_valence_table:
        cursor.execute('''
            INSERT INTO bond_valence_params (atom_a, valence_a, atom_b, valence_b, r0, b)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (row[0], int(row[1]), row[2], int(row[3]), float(row[4]), float(row[5])))

def populate_element_types(cursor, element_types):
    for group_name, atomic_numbers in element_types.items():
        # 1. グループ名の登録（すでにあれば無視）
        cursor.execute('INSERT OR IGNORE INTO element_groups (group_name) VALUES (?)', (group_name,))
        
        # 2. 登録したグループの ID を取得
        cursor.execute('SELECT group_id FROM element_groups WHERE group_name = ?', (group_name,))
        result = cursor.fetchone()
        
        if result:
            # 【修正点】タプルからインデックス0の要素（IDの数値）を取り出す
            group_id = result[0] 

            # 3. 各原子番号との紐付けを登録
            for num in atomic_numbers:
                cursor.execute('''
                    INSERT OR IGNORE INTO element_group_mapping (atomic_number, group_id)
                    VALUES (?, ?)
                ''', (int(num), group_id))

# 使用例
from atomic_numbers import *
from bond_valence import *
from ions import *
from mass import *
from radius import *
from valence import *
from empirical import *

import os
db_path=os.path.join(os.path.dirname(__file__), "element_data.db")

conn, cursor = create_element_database(db_path)
populate_elements(cursor, number_symbol, number_mass, typical_valence)
populate_radii(cursor, number_radius, shannon_radii, number_symbol)
populate_bvm(cursor, bond_valence_table)
populate_element_types(cursor, element_types)
conn.commit()
conn.close()