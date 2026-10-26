import re
import os
import sqlite3
import pandas as pd
import json

elements = {"atomic_number": int, "symbol": str, "atomic_mass": float, "typical_valences": json.loads}
ionic_radii = {"atomic_number": int, "charge": int, "coordination_number": int, "radius": float, "is_shannon": bool}
bond_valence_params = {"atom_a": str, "valence_a": int, "atom_b": str, "valence_b": int, "r0": float, "b": float}
element_group_mapping = {"atomic_number": int, "group_id": int}
element_groups = {"group_id": int, "group_name": str}


class AtomInformation:
    def __init__(self):
        dir = os.path.dirname(__file__)
        self.table = os.path.join(dir, "info", "element_data.db")

    ## General Function
    def get(self, table_name:str, targets:list=None, **kwargs):

        if table_name == "elements":
            form = elements
        elif table_name == "ion_radii":
            form = ionic_radii
        elif table_name == "bond_valence_params":
            form = bond_valence_params
        elif table_name == "element_group_mapping":
            form = element_group_mapping
        elif table_name == "element_groups":
            form = element_groups
        else:
            return None
        
        keys = []
        vals = []
        for key, val in kwargs.items():
            if val is None:
                continue
            keys.append(key)
            vals.append(str(val))
        
        conn = sqlite3.connect(self.table)
        target = ",".join(targets) if targets else "*"
        where_query = " AND ".join([f"{key} = ?" for key in keys])
        query = f'SELECT {target} FROM {table_name} WHERE {where_query}'
        df = pd.read_sql_query(query, conn, params=vals)
        conn.close()

        if df.empty:
            return None
        else:
            for col in df.columns:
                if col in form:
                    df[col] = df[col].apply(lambda x: form[col](x) if pd.notna(x) else None)
            return df

    def parse_symbol(self, symbol):
        symbol_str = str(symbol).strip()
        pattern = r'^([A-Z][a-z]?)(\d*)([\+\-]?)$'
        match = re.match(pattern, symbol_str)
        if match:
            symbol = match.group(1)
            charge_str = match.group(2)
            sign = match.group(3)
            if sign:
                charge_num = int(charge_str) if charge_str else 1
                final_charge = charge_num if sign == "+" else -charge_num
            else:
                final_charge = 0
        else:
            final_charge = 0

        return symbol, final_charge

    def get_atomic_number(self, symbol=None):
        atomic_number = self.get("elements", 
                                    targets=["atomic_number"], 
                                    symbol=symbol)
        if atomic_number is None:
            return None
        atomic_number = atomic_number.iloc[0]["atomic_number"]      
        return atomic_number


    def get_symbol(self, atomic_number=None):
        symbol = self.get("elements", 
                          targets=["symbol"], 
                          atomic_number=atomic_number)
        if symbol is None:
            return None
        symbol = symbol.iloc[0]["symbol"]
        return symbol

    def get_atomic_mass(self, atomic_number=None, symbol=None):
        mass = self.get("elements", 
                        targets=["atomic_mass"], 
                        atomic_number=atomic_number, 
                        symbol=symbol)
        if mass is None:
            return None
        mass = mass.iloc[0]["atomic_mass"]
        return mass

    def get_typical_valences(self, atomic_number=None, symbol=None):
        valences = self.get("elements", 
                            targets=["typical_valences"], 
                            atomic_number=atomic_number, 
                            symbol=symbol)
        if valences is None:
            return [0]
        valences = valences.iloc[0]["typical_valences"]
        return valences

    def get_shannon_radius(self, atomic_number=None, symbol=None, charge=None, coordination_number=None):
        radius = self.get("ion_radii", 
                          targets=["radius"], 
                          atomic_number=atomic_number, 
                          symbol=symbol, 
                          charge=charge, 
                          is_shannon=1, 
                          coordination_number=coordination_number)
        if radius is None:
            return None
        radius = radius.iloc[0]["radius"]
        return radius

    def get_radius(self, atomic_number=None, symbol=None, charge=None):
        if charge is None:
            charge = 0
        charge = int(charge)
        radius = self.get("ion_radii", 
                          targets=["radius"], 
                          atomic_number=atomic_number, 
                          symbol=symbol, 
                          charge=charge, 
                          is_shannon=0)
        if radius is None and charge != 0:
            radius = self.get("ion_radii",
                              targets=["radius"],
                              atomic_number=atomic_number,
                              symbol=symbol,
                              charge=charge,
                              is_shannon=1)
        if radius is None:
            radius = self.get("ion_radii",
                              targets=["radius"],
                              atomic_number=atomic_number,
                              symbol=symbol,
                              charge=0,
                              is_shannon=0)
        if radius is not None:
            radius = radius.iloc[0]["radius"]

        return radius

    def get_element_type(self, atomic_number=None, symbol=None):
        if atomic_number is None:
            atomic_number = self.get_atomic_number(symbol)
            
        group_id = self.get("element_group_mapping", 
                            targets=["group_id"], 
                            atomic_number=atomic_number)
        if group_id is None:
            return None
        group_id = group_id.iloc[0]["group_id"]
        group_name = self.get("element_groups", 
                              targets=["group_name"], 
                              group_id=group_id).iloc[0]["group_name"]
        return group_name

    def get_typical_coordination_number(self, atomic_number=None, symbol=None, charge=None) -> list:
        if charge is not None and int(charge) != 0:
            df = self.get("ion_radii",
                          targets=["coordination_number"],
                          atomic_number=atomic_number,
                          symbol=symbol,
                          charge=int(charge),
                          is_shannon=1)
            if df is not None and not df.empty:
                cns = sorted(df["coordination_number"].dropna().astype(int).unique().tolist())
                if cns:
                    return cns
        df = self.get("ion_radii",
                      targets=["coordination_number"],
                      atomic_number=atomic_number,
                      symbol=symbol,
                      is_shannon=1)
        if df is not None and not df.empty:
            cns = sorted(df["coordination_number"].dropna().astype(int).unique().tolist())
            if cns:
                return cns
        return [0]





