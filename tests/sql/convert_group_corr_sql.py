import sqlite3
import json
import numpy as np

def create_unified_correlation_db(db_path, subgroup_table, hall_number_list):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    sub_map = {}
    par_map = {}

    for row in hall_number_list:
        hm_full = row["hm_full"]
        if hm_full not in sub_map:
            sub_map[hm_full] = {"sub_hall_number": row["serial_no"], 
                                 "sub_hm_short": row["hm_short"], 
                                 "sub_hm_full": row["hm_full"], 
                                 "sub_table_no": row["ita_no"], 
                                 "sub_setting_no": row["sub_no"]}

        if row["sub_no"] == 1:
            table_no = row["ita_no"]
            if table_no not in par_map:
                par_map[table_no] = {"parent_hall_number": row["serial_no"], 
                                     "parent_hm_short": row["hm_short"], 
                                     "parent_hm_full": row["hm_full"], 
                                     "parent_table_no": row["ita_no"], 
                                     "parent_setting_no": row["sub_no"]}

    # 2. テーブル作成

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS group_correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_hall_number INTEGER,
            parent_table_no INTEGER,
            parent_hm_short TEXT,
            parent_hm_full TEXT,
            parent_setting_no INTEGER,
            sub_hall_number INTEGER,
            sub_table_no INTEGER,
            sub_hm_short TEXT,
            sub_hm_full TEXT,
            sub_setting_no INTEGER,
            index_val INTEGER,
            transformation_str TEXT,
            matrix_p2s TEXT,
            shift_p2s TEXT,
            matrix_s2p TEXT,
            shift_s2p TEXT
        )
    ''')

    # 3. データの計算と挿入
    for p_no_str, subgroups in subgroup_table.items():
        p_no = int(p_no_str)
        p_info = par_map.get(p_no)

        for item in subgroups:
            hm_full = item["subgroup_name"].replace(" ", "")
            if hm_full not in sub_map:
                 s_info = {
                            "sub_hall_number": None, 
                            "sub_hm_short": hm_full, 
                            "sub_hm_full": hm_full, 
                            "sub_table_no": item["subgroup_no"], 
                            "sub_setting_no": None}
            else:
                s_info = sub_map.get(hm_full)
                        
            # 行列計算 (P -> S)
            M = np.array(item["matrix"])
            S = np.array(item["origin_shift"])
            
            # 逆行列計算 (S -> P)
            try:
                M_inv = np.linalg.inv(M)
                S_inv = -M_inv.dot(S)
            except np.linalg.LinAlgError:
                continue

            conv_info = {"index_val": item["index"], 
                         "transformation_str": item["transformation_str"], 
                         "matrix_p2s": json.dumps(M.tolist()), 
                         "shift_p2s": json.dumps(S.tolist()), 
                         "matrix_s2p": json.dumps(M_inv.tolist()), 
                         "shift_s2p": json.dumps(S_inv.tolist())
                         }

            keys = list(p_info.keys()) + list(s_info.keys()) + list(conv_info.keys())
            keys_str = ", ".join(keys)
            values = list(p_info.values()) + list(s_info.values()) + list(conv_info.values())
            vals_str = ", ".join(["?"]*(len(keys)))

            cursor.execute(f'''
                INSERT INTO group_correlations 
                ({keys_str}) 
                VALUES ({vals_str})
            ''', values)

    conn.commit()
    conn.close()
    print("Unified group correlation table created.")

import os
subgroup_file = os.path.join(os.path.dirname(__file__), "subgroup_table.json")
with open(subgroup_file, "r") as f:
    subgroup_table = json.load(f)
hall_number_file = os.path.join(os.path.dirname(__file__), "hall_number_table.json")
with open(hall_number_file, "r") as f:
    hall_number_list = json.load(f)

create_unified_correlation_db("group_correlations.db", subgroup_table, hall_number_list)