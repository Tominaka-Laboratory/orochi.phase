#  Source: International Tables for Crystallography (2006). Vol. A, ch. 3.1, pp. 44-54

import os
import sqlite3
import numpy as np
import pandas as pd
import json

space_group_table = {
    "hall_number": int,
    "hall_symbol": str,
    "crystal_system": str,
    "table_no": int,
    "hm_short": str,
    "hm_full": str,
    "setting_no": int,
    "axes_origin": str,
    "sch_symbol": str,
    "pg_no": int,
    "pg_hm": str,
    "pg_sch": str,
    "lc_no": int,
    "lc_hm": str,
    "ext_system": str,
    "ext_a": str,
    "ext_b": str,
    "ext_c": str}

group_correlations_table = {
    "parent_hall_number": int,
    "parent_table_no": int,
    "parent_hm_symbol": str,
    "parent_full_notation": str,
    "sub_hall_number": int,
    "sub_table_no": int,
    "sub_hm_symbol": str,
    "sub_full_notation": str,
    "index_val": int,
    "transformation_str": str,
    "matrix_p2s": json.loads,
    "shift_p2s": json.loads,
    "matrix_s2p": json.loads,
    "shift_s2p": json.loads}

point_group_table = {
    "hm_symbol": str,
    "scm_symbol": str,
    "operations": str
}

wyckoff_table = {
    "space_group": str,
    "setting": str,
    "multiplicity": int,
    "wyckoff_letter": str,
    "point_symmetry": str,
    "coordinates": str
}



class Database(object):
    """
    Database class for searching and retrieving space group information.

    This class provides tools to map space group symbols or numbers to their
    corresponding table entries and Hall numbers.

    Attributes
    ----------
    table : list of list
        The master space group table containing numbers, symbols, and settings.
    hall_table : dict
        Mapping of Hall numbers (str) to international space group numbers (int).
    """

    def __init__(self):
        """
        Initialize the space group database and its mapping tables.
        """

        dir = os.path.dirname(__file__)
        self.sg_table = os.path.join(dir, "info", "symmetry_groups.db")
        self.gc_table = os.path.join(dir, "info", "group_correlations.db")
        self.wy_table = os.path.join(dir, "info", "wyckoff.db")

    ## General Function
    def get(self, table_name:str="sg", targets:list=None, **kwargs):

        if table_name == "sg":
            table = self.sg_table
            sql_table = "space_groups"
            form = space_group_table
        elif table_name == "gc":
            table = self.gc_table
            sql_table = "group_correlations"
            form = group_correlations_table
        elif table_name == "wy":
            table = self.wy_table
            sql_table = "wyckoff_positions"
            form = wyckoff_table
        elif table_name == "pg":
            table = self.sg_table
            sql_table = "point_groups"
            form = point_group_table
        else:
            raise ValueError(f"Invalid target: {targets}")

        keys = []
        vals = []
        for key, val in kwargs.items():
            if val is None:
                continue
            keys.append(key)
            vals.append(str(val))
        
        conn = sqlite3.connect(table)
        target = ",".join(targets) if targets else "*"
        where_query = " AND ".join([f"{key} = ?" for key in keys])
        query = f'SELECT {target} FROM {sql_table} WHERE {where_query}'
        df = pd.read_sql_query(query, conn, params=vals)
        conn.close()

        if df.empty:
            return None
        else:
            for col in df.columns:
                if col in form:
                    df[col] = df[col].apply(lambda x: form[col](x) if pd.notna(x) else None)
            return df

    ## Spacegroups          
    def get_symbol(self, number=None, hm_full=None):
        """
        Get the Short HM symbol from the space group number.

        Parameters
        ----------
        number : int
            International space group number (1-230).

        Returns
        -------
        str or None
            Short Hermann-Mauguin symbol.
        """
        symbol = self.get("sg", 
                          targets=["hm_short"], 
                          table_no=number, 
                          hm_full=hm_full)
        if symbol is not None:
            res = symbol["hm_short"]
            if res.size == 1:
                return res[0]
            else:
                return res
        else:
            return None


    def get_number(self, symbol):
        """
        Get the international space group number from a symbol.

        Parameters
        ----------
        symbol : str
            Space group symbol (Short or Full notation).

        Returns
        -------
        int or None
            International space group number.
        """

        number = self.get("sg", 
                          targets=["table_no"], 
                          hm_short=symbol.replace(" ", ""))
        if number is not None:
            res = number["table_no"]
            if res.size == 1:
                return res[0]
            else:
                return res
        else:
            return None

    def get_full_symbol(self, 
                         space_group:str=None, 
                         number:int=None,
                         setting:int=1):
        """
        Standardize a space group input into a Short Hermann-Mauguin symbol.

        Parameters
        ----------
        space_group : str or int
            The space group number (1-230) or symbol (e.g., 'P21/c', 'P 1 21/c 1').

        Returns
        -------
        str or None
            The standard Short HM symbol if found, otherwise None.
        """

        symbol = self.get("sg", 
                          targets=["hm_full"], 
                          hm_short=space_group.replace(" ", ""), 
                          table_no=number,
                          setting_no=setting)
        if symbol is not None:
            res = symbol["hm_full"]
            if res.size == 1:
                return res[0]
            else:
                return res
        else:
            return None

    def get_hall_number(self,
                        hm_short:str=None, 
                        hm_full:str=None,
                        table_no:int=None, 
                        setting:int=1):
        """
        Determine the Hall number (1-530) based on space group and setting.

        This method is essential for retrieving exact symmetry operations from 
        spglib, as it distinguishes between different settings (e.g., b-axis 
        vs c-axis unique) of the same space group number [1, 2].

        Parameters
        ----------
        space_group : str or int
            International space group number or symbol.
        setting : int, optional
            The choice or setting index (1-based), by default 1.

        Returns
        -------
        int or None
            The unique Hall number if a match is found, otherwise None.
        """
        if hm_short is not None:
            hm_short=hm_short.replace(" ", "")
        if hm_full is not None:
            hm_full=hm_full.replace(" ", "")

        hall_num = self.get("sg", 
                        targets=["hall_number"], 
                        hm_short=hm_short, 
                        hm_full=hm_full,
                        table_no=table_no,
                        setting_no=setting)
        if hall_num is not None:
            hall_num = hall_num["hall_number"].values
            if hall_num.size == 1:
                return hall_num[0]
            else:
                return hall_num
        else:
            return None

    def get_hm_info(self, hall_number):
        res = self.get("sg", 
                          targets=["hm_short", "hm_full", "table_no", "setting_no"], 
                          hall_number=hall_number)
        if res is not None:
            return res
        else:
            return None

    def get_crystal_system(self, hall_number):
        res = self.get("sg", 
                          targets=["crystal_system"], 
                          hall_number=hall_number)
        if res is not None:
            return res.iloc[0]["crystal_system"]
        else:
            return None

    def get_symmetry_operations(self, hall_number=None, hm_full=None, setting=None):
        res = self.get("sg", 
                          targets=["rotation", "translation"], 
                          hall_number=hall_number, 
                          hm_full=hm_full,
                          setting_no=setting)
        if res is not None:
            rot = json.loads(res.iloc[0]["rotation"])
            tra = json.loads(res.iloc[0]["translation"])
            return {"rotation": np.array(rot).reshape(-1,3,3),
                    "translation":np.array(tra).reshape(-1,3)
                    }
        else:
            return None


    ## Subgroups
    def get_maximal_subgroups(self, number):
        """
        Returns a list of dictionaries containing transformation data
        to maximal subgroups for a given parent space group number.
        """
        targets = ["sub_hall_number", "sub_table_no", "sub_hm_symbol",
                   "sub_full_notation", "index_val", "transformation_str",
                   "matrix_p2s", "shift_p2s"]

        return self.get("gc", 
                        targets=targets, 
                        parent_table_no=number)

    def get_subgroup_candidates(self, parent_hall_number):
        """
        Given a parent hall number, returns a DataFrame containing candidate
        subgroup hall numbers and their corresponding transformation strings.
        """

        # 1. Retrieve parent space group information
        res_parent = self.get("sg", 
                            targets=["hm_short", "hm_full", "table_no", "setting_no"], 
                            hall_number=parent_hall_number)

        if res_parent is None or res_parent.empty:
            return None

        # Extract parent information from iloc[0]
        parent_table_no = res_parent.iloc[0]["table_no"]
        parent_setting_no = res_parent.iloc[0]["setting_no"]

        # Unwrap .value attribute if present
        if hasattr(parent_table_no, "value"):
            parent_table_no = parent_table_no.value
        if hasattr(parent_setting_no, "value"):
            parent_setting_no = parent_setting_no.value

        # 2. Retrieve candidate subgroups from parent
        res_gc = self.get("gc", 
                        targets=["sub_table_no", "sub_setting_no", "transformation_str"],
                        parent_table_no=parent_table_no,
                        parent_setting_no=parent_setting_no)

        # Check for None or empty result
        if res_gc is None or res_gc.empty:
            return None

        res_gc["sub_setting_no"] = res_gc["sub_setting_no"].fillna(1).astype(int)
        # 3. Process subgroup candidates
        candidate_records = []

        for _, row in res_gc.iterrows():
            sub_table_no = row["sub_table_no"]
            sub_setting_no = row["sub_setting_no"]
            transformation_str = row["transformation_str"]

            if hasattr(sub_table_no, "value"):
                sub_table_no = sub_table_no.value
            if hasattr(sub_setting_no, "value"):
                sub_setting_no = sub_setting_no.value
            if hasattr(transformation_str, "value"):
                transformation_str = transformation_str.value

            # Retrieve corresponding sub_hall_number from sg table
            res_sg = self.get("sg", 
                            targets=["hall_number"], 
                            table_no=int(sub_table_no),
                            setting_no=int(sub_setting_no))

            if res_sg is None or res_sg.empty:
                continue

            # Collect hall_number entries returned from sg table
            for _, sg_row in res_sg.iterrows():
                sub_hall = sg_row["hall_number"]
                if hasattr(sub_hall, "value"):
                    sub_hall = sub_hall.value

                candidate_records.append({
                    "sub_hall_number": sub_hall,
                    "transformation_str": transformation_str,
                    "sub_table_no": sub_table_no,
                    "sub_setting_no": sub_setting_no
                })

        if not candidate_records:
            return None

        return pd.DataFrame(candidate_records)

    def get_subgroup_conversions(self, 
                                 parent_hall_number, 
                                 sub_hall_number, 
                                 transformation_str):
        """
        Returns conversion matrix (matrix_p2s, shift_p2s)
        """

        # 1. Retrieve parent space group information
        res_parent = self.get("sg", 
                              targets=["hm_short", "hm_full", "table_no", "setting_no"], 
                              hall_number=parent_hall_number)

        if res_parent is None or res_parent.empty:
            return None

        parent_table_no = res_parent.iloc[0]["table_no"]
        parent_setting_no = res_parent.iloc[0]["setting_no"]

        if hasattr(parent_table_no, "value"):
            parent_table_no = parent_table_no.value
        if hasattr(parent_setting_no, "value"):
            parent_setting_no = parent_setting_no.value

        # 2. Retrieve child space group information
        res_sub = self.get("sg", 
                           targets=["hm_short", "hm_full", "table_no", "setting_no"], 
                           hall_number=sub_hall_number)

        if res_sub is None or res_sub.empty:
            return None

        sub_table_no = res_sub.iloc[0]["table_no"]
        sub_setting_no = res_sub.iloc[0]["setting_no"]

        if hasattr(sub_table_no, "value"):
            sub_table_no = sub_table_no.value
        if hasattr(sub_setting_no, "value"):
            sub_setting_no = sub_setting_no.value

        # 3. Query gc table (excluding setting_no from filter criteria)
        res_gc = self.get("gc", 
                        targets=["matrix_p2s", "shift_p2s"],
                        parent_table_no=int(parent_table_no),
                        sub_table_no=int(sub_table_no),
                        transformation_str=transformation_str
                        )

        if res_gc is None or res_gc.empty:
            return None

        return res_gc



    ## Supergroups
    def get_minimal_supergroups(self, number):
        """
        Returns a list of dictionaries containing transformation data
        to minimal supergroups for a given parent space group number.
        """
        targets = ["parent_hall_number", "parent_table_no", "parent_hm_symbol",
                   "parent_full_notation", "index_val", "transformation_str",
                   "matrix_s2p", "shift_s2p"]
        return self.get("gc", 
                        targets=targets, 
                        sub_table_no=number)

    def get_supergroup_candidates(self, sub_hall_number):
        """
        Returns a list of supergroup numbers for a given subgroup number.
        """
        return self.get("gc", 
                        targets=["parent_hall_number", "parent_hm_full", "transformation_str"], 
                        sub_hall_number=sub_hall_number)

    def get_supergroup_conversions(self, 
                                 parent_hall_number, 
                                 sub_hall_number, transformation_str):
        """
        Returns conversion matrix
        """
        return self.get("gc",
                        targets=["matrix_s2p", "shift_s2p"],
                        parent_hall_number=int(parent_hall_number),
                        sub_hall_number=int(sub_hall_number),
                        transformation_str=transformation_str)

    ## Point groups
    def get_point_group_rotations(self, hm_symbol=None, sch_symbol=None):
        rotations = self.get("pg", 
                        targets=["rotations"], 
                        hm_symbol=hm_symbol,
                        scm_symbol=sch_symbol)

        if rotations is not None:
            return rotations.iloc[0]["rotations"]
        else:
            return None


    ## Wyckoff
    def get_wyckoff_positions(self, hm_full=None, setting=1, hall_number=None):
        if hall_number is not None:
            df = self.get_hm_info(hall_number=hall_number).iloc[0]
            hm_full = df["hm_full"]
            setting = df["setting_no"]

        info = self.get("wy",
                        targets=["multiplicity", "wyckoff_letter", "point_symmetry", "coordinates"],
                        space_group=hm_full,
                        setting=setting)

        if info is not None:
            return info.to_dict(orient="records")
        else:
            return None

    def get_site_constraints(self, hm_full=None, setting=1, hall_number=None, wyckoff_letter=None):
        if hall_number is not None:
            df = self.get_hm_info(hall_number=hall_number).iloc[0]
            hm_full = df["hm_full"]
            setting = df["setting_no"]

        info = self.get("wy",
                        targets=["coordinates"],
                        space_group=hm_full,
                        setting=setting,
                        wyckoff_letter=wyckoff_letter)

        if info is not None:
            return info.iloc[0]["coordinates"]
        else:
            return None 

    def get_site_symmetry_symbol(self, hm_full=None, setting=1, hall_number=None, wyckoff_letter=None):
        if hall_number is not None:
            df = self.get_hm_info(hall_number=hall_number).iloc[0]
            hm_full = df["hm_full"]
            setting = df["setting_no"]

        info = self.get("wy",
                        targets=["point_symmetry"],
                        space_group=hm_full,
                        setting=setting,
                        wyckoff_letter=wyckoff_letter)

        if info is not None:
            return info.iloc[0]["point_symmetry"]
        else:
            return None 
        
    def get_equiv_multiplicity_wyckoff_letters(self, hm_full=None, setting=1, hall_number=None, wyckoff_letter=None):
        if hall_number is not None:
            df = self.get_hm_info(hall_number=hall_number).iloc[0]
            hm_full = df["hm_full"]
            setting = df["setting_no"]

        info = self.get("wy",
                        targets=["multiplicity"],
                        space_group=hm_full,
                        setting=setting,
                        wyckoff_letter=wyckoff_letter)

        if info is not None:
            multiplicity = info.iloc[0]["multiplicity"]
        else:
            multiplicity = None
            return None
        
        info = self.get("wy",
                        targets=["wyckoff_letter"],
                        space_group=hm_full,
                        setting=setting,
                        multiplicity=int(multiplicity))

        if info is not None:
            wyckoff_letters = info["wyckoff_letter"]
        else:
            wyckoff_letters = None

        return wyckoff_letters

        

