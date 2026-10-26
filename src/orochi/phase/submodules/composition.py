import re
import numpy as np


def set_symbol_occ_adp_from_string(
    self, string
):  # maybe useful for non-cif type input (Ce 1 O 2)
    def read_composition(string):
        string = re.sub(r"(\w)([A-Z])", r"\1 \2", string)
        list = string.split()
        symbols = []
        occ = []
        isatom = True

        def read_atom(item):
            if item.find("+") != -1:
                return [item], False, 1

            elif item.find("-") != -1:
                return [item], False, 1

            elif re.sub(r"[\D]+", "", item) == "":
                return [item], False, 1

            else:
                sym = [_f for _f in re.split(r"[\d]+", item) if _f]
                isatom = True
                oc = [_f for _f in re.split(str(sym), item) if _f]
                return sym, isatom, oc

        def read_ratio(item):
            if re.sub(r"[\d,.]+", "", item) == "":
                return 1, True, [item]

            else:
                return 1, False, [1.0]

        for item in list:
            if isatom:
                sym, isatom, oc = read_atom(item)
                if isatom:
                    symbols += sym
                    occ += oc
                else:
                    symbols += sym

            else:
                sym, isatom, oc = read_ratio(item)
                if isatom:
                    occ += oc
                else:
                    occ += oc

                    sym, isatom, oc = read_atom(item)
                    if isatom:
                        symbols += sym
                        occ += oc
                    else:
                        symbols += sym

        return symbols, occ
    

    symbols, occ = read_composition(string)
    self["symbol"] = np.array(symbols)
    self["label"] = np.array(
        [symbol + "(" + str(id) + ")" for id, symbol in enumerate(symbols)]
    )

    self["occ"] = np.array(occ, dtype=float)
    self["xyz"] = np.zeros((np.size(self["occ"]), 3))
    self["adp"] = np.ones_like(self["occ"]) * 0.01
    self["space_group"] = None

def set_composition_from_txt(self):
    txt = self["composition"]

    if txt == []:
        return None
    
    composition_dictionary = text_to_dictionary(txt)
    symbols = list(composition_dictionary.keys())
    occ = list(composition_dictionary.values())

    self["symbol"] = np.array(symbols)
    self["label"] = np.array(
        [symbol + "(" + str(id) + ")" for id, symbol in enumerate(symbols)]
    )

    self["occ"] = np.array(occ, dtype=float)
    self["xyz"] = np.zeros((np.size(self["occ"]), 3))
    self["adp"] = np.ones_like(self["occ"]) * 0.01
    self["space_group"] = None


def text_to_dictionary(txt):
    # Dictionary to store parsed element-count pairs
    dictionary = {}
    
    # Regular expression pattern:
    # ([A-Z][a-z]?) : Element symbol (1 uppercase letter + optional lowercase letter)
    # ([0-9,.]*)    : Following numeric stoichiometry (optional)
    pattern = r'([A-Z][a-z]?)([0-9,.]*)'
    
    # Extract all element and stoichiometry pairs
    matches = re.findall(pattern, txt)
    
    for atom, number_str in matches:
        # Default to 1.0 when stoichiometry is omitted (e.g., "Ru" in "RuO2")
        if number_str == "":
            number = 1.0
        else:
            number = float(number_str.replace(',', ''))
        
        if atom in dictionary:
            dictionary[atom] += number
        else:
            dictionary[atom] = number
            
    return dictionary