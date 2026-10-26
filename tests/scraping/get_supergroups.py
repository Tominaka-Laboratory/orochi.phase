import json
import numpy as np
import os
from orochi.phase.database.info.subgroups import subgroup_table


def generate_supergroup_table():
    """
    subgroup_table (Parent -> Subgroup) の関係を反転させ、
    supergroup_table (Subgroup -> Supergroup) を作成してJSONで保存する。
    """
    supergroup_dict = {}

    # 1. subgroup_table を走査して反転関係を構築
    for parent_no, subs in subgroup_table.items():
        for entry in subs:
            # 変換後の空間群番号 (Subgroup No.)
            child_no = str(entry["subgroup_no"])
            
            if child_no not in supergroup_dict:
                supergroup_dict[child_no] = []
            
            # 行列 P を取得して逆行列 P^-1 を計算
            matrix_p = np.array(entry["matrix"])
            try:
                # 逆行列 (Subgroup から Supergroup への格子変換行列)
                matrix_inv = np.linalg.inv(matrix_p).tolist()
            except np.linalg.LinAlgError:
                # 万が一、特異行列（行列式0）の場合は単位行列を代入
                matrix_inv = np.eye(3).tolist()

            # 原点シフト s を取得して逆方向のシフトを計算 (-P^-1 * s)
            origin_shift = np.array(entry["origin_shift"])
            # 逆方向の座標変換において、新しい原点への補正値を計算
            # 浮動小数点の誤差を考慮しつつ変換
            origin_shift_inv = (-np.dot(np.linalg.inv(matrix_p), origin_shift) % 1.0).tolist()

            # 超群としてのエントリを作成
            super_entry = {
                "supergroup_no": parent_no,
                "supergroup_name": "Standard Symbol or Lookup Required", # 必要に応じてDatabaseから取得
                "matrix_inv": matrix_inv,
                "origin_shift_inv": origin_shift_inv,
                "index": entry["index"],
                "parent_transformation_str": entry["transformation_str"]
            }
            
            # リストに追加（1つの空間群に複数の超群候補があるため）
            supergroup_dict[child_no].append(super_entry)

    # 2. JSONファイルとして保存
    output_file = os.path.join(os.path.dirname(__file__), "supergroup_table.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(supergroup_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Successfully generated: {output_file}")

if __name__ == "__main__":
    generate_supergroup_table()