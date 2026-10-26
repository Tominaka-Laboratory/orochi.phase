import numpy as np
import re, os, datetime
from ...database.symmetries import Database 

def cif_to_phase(cif_file, phase):
    db = Database()

    def make_cif_dictionary(lines):
        in_loop = False
        is_keyword = False
        is_read = False
        is_separate = False
        is_continue = False

        keyword = ""

        cif_dictionary = {}
        loop_list = []
        continue_list = []
        separate_block = []

        continue_count = 1

        for m, line in enumerate(lines):
            if is_continue:
                if continue_count > 1:
                    continue_count -= 1
                    continue
                elif continue_count == 1:
                    line = line.strip(";")
                    is_continue = False

            line = line.strip()
            blocks = list(filter(None, line.split()))
            is_keyword = False

            # find the majors=====================================
            if len(blocks) == 0:
                continue

            elif line.strip()[0] == "_":
                is_keyword = True
                if is_read:
                    in_loop = False
                    is_read = False

            elif line.strip()[0] == ";":
                is_continue = True
                continue_list += [line.strip(";").strip()]

                while True:
                    li = lines[m + continue_count].strip()
                    if len(li) == 0:
                        continue_list += [li]
                        continue_count += 1
                        continue
                    if li[0] == ";":
                        break
                    else:
                        continue_list += [li]
                        continue_count += 1

                blocks = [" ".join(continue_list)]
                continue_list = []

            elif line.strip()[0] == "#":
                if is_read:
                    in_loop = False
                    is_read = False
                    loop_list = []
                continue

            elif blocks[0][:5] == "loop_":
                in_loop = True
                is_read = False
                loop_list = []
                continue

            # join the terms with "'"=====================================
            correct_blocks = []
            cb = []
            joint = False
            for b in blocks:
                b = b.strip()
                if "'" == b[0]:
                    cb += [b.strip("'")]
                    joint = True
                    continue

                elif "'" == b[-1]:
                    cb += [b.strip("'")]
                    correct_blocks += [" ".join(cb)]
                    cb = []
                    joint = False
                    continue

                elif joint:
                    cb += [b]
                else:
                    correct_blocks += [b]

            blocks = correct_blocks

            # sort out the line=====================================
            if is_keyword and (len(blocks) != 1):
                keyword = blocks[0].strip()
                val = blocks[1:]
                cif_dictionary[keyword] = val
                in_loop = False
                is_read = False
                loop_list = []
                continue

            elif not is_keyword and not in_loop:
                continue

            elif is_keyword and in_loop:
                keyword = blocks[0].strip()
                cif_dictionary[keyword] = []
                loop_list += [keyword]
                continue

            elif not is_keyword and in_loop:

                if is_separate:
                    separate_block += blocks
                    blocks = list(filter(None, separate_block))

                if len(loop_list) == len(blocks):
                    for n, keyword in enumerate(loop_list):
                        cif_dictionary[keyword] += [blocks[n]]
                    is_read = True
                    is_separate = False
                    separate_block = []

                else:
                    if is_read:
                        in_loop = False
                    else:
                        if is_separate:
                            continue
                        else:
                            is_separate = True
                            separate_block += list(filter(None, blocks))

        return cif_dictionary

    def separate_value_std(input):
       
        if isinstance(input, (list, np.ndarray)):
            v = []
            s = []

            for inp in input:
                val = list(filter(None, re.sub("[(,)]", " ", inp).split()))
                if re.sub("[.-]", "", val[0]).isdecimal():
                    value = float(val[0])
                else:
                    value = 0.0

                if len(val) == 2:
                    if re.sub("[.]", "", val[1]).isdecimal():
                        std = float(val[1])
                        unit = value / int(str(value).replace(".", ""))
                        std *= unit

                    else:
                        std = np.nan
                else:
                    std = np.nan
                v += [value]
                s += [std]
            return np.array(v), np.array(s)

        else:
            val = list(filter(None, re.sub("[(,)]", " ", input).split()))
            if re.sub("[.]", "", val[0]).isdecimal():
                value = float(val[0])
            else:
                value = 0.0

            if len(val) == 2:
                if re.sub("[.]", "", val[1]).isdecimal():
                    std = float(val[1])
                    unit = value / int(str(value).replace(".", ""))
                    std *= unit
                else:
                    std = np.nan
            else:
                std = np.nan
            return value, std

    def apply_cell_information(phase, cif_dictionary):
        read_dic = {
            "_cell_length_a": 0,
            "_cell_length_b": 1,
            "_cell_length_c": 2,
            "_cell_angle_alpha": 3,
            "_cell_angle_beta": 4,
            "_cell_angle_gamma": 5,
        }

        for key, id in list(read_dic.items()):
            if key in cif_dictionary:
                val = cif_dictionary[key]
                val, std = separate_value_std(val)
                phase["lattice","value",id] = val
                phase["lattice","std",id] = std

        return phase

    def apply_space_group_information(phase, cif_dictionary):
        is_spg = False
        is_spn = False

        if "_space_group_name_H-M_alt" in list(cif_dictionary.keys()):
            space_group = cif_dictionary["_space_group_name_H-M_alt"][0]
            is_spg = True

        elif "_symmetry_space_group_name_H-M" in list(cif_dictionary.keys()):
            space_group = cif_dictionary["_symmetry_space_group_name_H-M"][0]
            is_spg = True

        if "_space_group_IT_number" in list(cif_dictionary.keys()):
            space_group_number = int(
                cif_dictionary["_space_group_IT_number"][0]
            )
            is_spn = True

        elif "_symmetry_Int_Tables_number" in list(cif_dictionary.keys()):
            space_group_number = int(
                cif_dictionary["_symmetry_Int_Tables_number"][0]
            )
            is_spn = True

        if is_spg:
            if len(space_group.split(":")) > 1:
                space_group = space_group.split(":")[0]

            if space_group.isdecimal():
                phase["space_group","number"] = int(space_group)
            else:
                space_group = space_group[0] + space_group[1:].replace(
                    "M", "m"
                ).replace("N", "n").replace("A", "a").replace(
                    "B", "b"
                ).replace(
                    "C", "c"
                ).replace(
                    "D", "d"
                )
                phase["space_group"] = space_group

        if is_spn:
            phase["space_group","number"] = int(space_group_number)

        return phase

    def apply_atom_sites(phase, cif_dictionary):
        # read symbols: this is required
        if "_atom_site_type_symbol" in list(cif_dictionary.keys()):

            cif_symbols = np.array(cif_dictionary["_atom_site_type_symbol"])
            symbols = np.unique(cif_symbols)
            charges = np.zeros(cif_symbols.size)

            for sym in symbols:
                charge_str = re.sub("[a-z,A-Z]", "", sym)
                symbol_str = sym.strip(charge_str)

                use = np.where(cif_symbols == sym)
                cif_symbols[use] = symbol_str

                if len(charge_str) == 0:
                    continue

                if "-" in charge_str:
                    v = float(charge_str.strip("-")) * -1.0
                    charges[use] = v
                else:
                    v = float(charge_str.strip("+"))
                    charges[use] = v

            phase["charge"] = charges
            phase["symbol"] = cif_symbols

        if "_atom_site_label" in list(cif_dictionary.keys()):
            phase["label"] = cif_dictionary["_atom_site_label"]
        else:
            phase["label"] = phase["symbol"]

        # read fractional coordinates: this might be written in other ways
        phase.par["xyz"]["value"] = np.zeros((phase["symbol"].size, 3))
        phase.par["xyz"]["std"] = np.zeros((phase["symbol"].size, 3))

        if "_atom_site_fract_x" in list(cif_dictionary.keys()):
            val = cif_dictionary["_atom_site_fract_x"]
            val, std = separate_value_std(val)
            phase["xyz","value",:, 0] = val
            phase["xyz","std",:, 0] = std

        if "_atom_site_fract_y" in list(cif_dictionary.keys()):
            val = cif_dictionary["_atom_site_fract_y"]
            val, std = separate_value_std(val)
            phase["xyz","value",:, 1] = val
            phase["xyz","std",:, 1] = std

        if "_atom_site_fract_z" in list(cif_dictionary.keys()):
            val = cif_dictionary["_atom_site_fract_z"]
            val, std = separate_value_std(val)
            phase["xyz","value",:, 2] = val
            phase["xyz","std",:, 2] = std

        phase["xyz"] -= np.floor(phase["xyz"])

        # read adps:
        if "_atom_site_U_iso_or_equiv" in list(cif_dictionary.keys()):
            val = cif_dictionary["_atom_site_U_iso_or_equiv"]
            val, std = separate_value_std(val)
            phase["adp","value"] = val
            phase["adp","std"] = std

        elif "_atom_site_B_iso_or_equiv" in list(cif_dictionary.keys()):
            val = cif_dictionary["_atom_site_B_iso_or_equiv"]
            val, std = separate_value_std(val)
            phase["adp","value"] = val / (8.0 * np.pi**2.0)
            phase["adp","std"] = std / (8.0 * np.pi**2.0)

        else:
            phase["adp","value"] = (
                np.ones_like(phase["symbol"]).astype(float) * 0.01
            )

        # read occupancy
        if "_atom_site_occupancy" in list(cif_dictionary.keys()):
            val = cif_dictionary["_atom_site_occupancy"]
            val, std = separate_value_std(val)
            phase["occ","value"] = val
            phase["occ","std"] = std
        else:
            phase["occ","value"] = np.ones_like(
                phase["symbol"]
            ).astype(float)

        if "_atom_site_symmetry_multiplicity" in list(cif_dictionary.keys()):
            val = cif_dictionary["_atom_site_symmetry_multiplicity"]
            phase["multiplicity","value"] = val

        return phase

    ld = open(cif_file, mode="r")
    lines = ld.readlines()
    ld.close()
    cif_dictionary = make_cif_dictionary(lines)

    
    phase = apply_cell_information(phase, cif_dictionary)
    phase = apply_space_group_information(phase, cif_dictionary)
    phase = apply_atom_sites(phase, cif_dictionary)
    try:
        phase.set_wyckoff_letters()
    except Exception:
        pass

    phase.par["xyz"]["min"] = phase["xyz"] - 0.05
    phase.par["xyz"]["max"] = phase["xyz"] + 0.05
    phase.par["xyz"]["fit"] = np.zeros_like(phase["xyz"])

    phase.par["occ"]["min"] = phase["occ"] - 0.05
    phase.par["occ"]["max"] = phase["occ"] + 0.05
    phase.par["occ"]["fit"] = np.zeros_like(phase["occ"])

    phase.par["adp"]["min"] = np.ones_like(phase["adp"]) * 0.0001
    phase.par["adp"]["max"] = np.ones_like(phase["adp"]) * 0.1
    phase.par["adp"]["fit"] = np.zeros_like(phase["adp"])

    phase.par["lattice"]["min"] = phase["lattice"] - np.array([0.01, 0.01, 0.01, 1, 1, 1])
    phase.par["lattice"]["max"] = phase["lattice"] + np.array([0.01, 0.01, 0.01, 1, 1, 1])
    phase.par["lattice"]["fit"] = np.zeros_like(phase["lattice"])

    return phase

def phase_to_cif(phase, file, save_type="raw"):
    def value_and_std(input, input2):
        if np.size(input) == np.size(input2):
            same_size = True
        else:
            same_size = False

        if isinstance(input, (list, np.ndarray)):
            v = []

            for id, value in enumerate(input):
                if same_size:
                    if input2[id] in [None, np.nan]:
                        v += [format(value, "1.5f")]

                    else:
                        res = format(input2[id], "1.1e").split("e")

                        if len(res) == 2:
                            std, ord = res
                            if float(std) > 1.9:
                                std = str(int(round(float(std))))
                                ord = int(ord)
                            else:
                                std = str(int(round(float(std) * 10.0)))
                                ord = int(ord) - 1
                            val = format(value, "." + str(abs(ord)) + "f")
                            v += [f"{val}({std})"]
                        else:
                            v += [format(value, "1.5f")]

                else:
                    v += [format(value, "1.5f")]
            return v

        else:
            value = input

            if same_size:
                if isinstance(input2, (list, np.ndarray)):
                    std = input2[0]
                else:
                    std = input2

                if std in [None, np.nan]:
                    return format(value, "1.5f")
                else:
                    res = format(std, "1.1e").split("e")
                    if len(res) == 2:
                        std, ord = res

                        if float(std) > 1.9:
                            std = str(int(round(float(std))))
                            ord = int(ord)

                        else:
                            std = str(int(round(float(std) * 10.0)))
                            ord = int(ord) - 1

                        return (
                            format(value, "." + str(abs(ord)) + "f")
                            + "("
                            + std
                            + ")"
                        )
                    else:
                        return format(value, "1.5f")
            else:
                return format(value, "1.5f")

    dic = {}

    if save_type == "raw":
        dic["_symmetry_Int_Tables_number"] = str(
            phase["space_group", "number"]
        )
        dic["_symmetry_space_group_name_H-M"] = phase["space_group"]
        dic["_symmetry_cell_setting"] = str(phase["crystal_system"])

        dic["_cell_length_a"] = value_and_std(
            phase["lattice"][0], phase["lattice","std"][0]
        )
        dic["_cell_length_b"] = value_and_std(
            phase["lattice"][1], phase["lattice","std"][1]
        )
        dic["_cell_length_c"] = value_and_std(
            phase["lattice"][2], phase["lattice","std"][2]
        )
        dic["_cell_angle_alpha"] = value_and_std(
            phase["lattice"][3], phase["lattice","std"][3]
        )
        dic["_cell_angle_beta"] = value_and_std(
            phase["lattice"][4], phase["lattice","std"][4]
        )
        dic["_cell_angle_gamma"] = value_and_std(
            phase["lattice"][5], phase["lattice","std"][5]
        )

        dic["_atom_site_type_symbol"] = phase["symbol"]
        dic["_atom_site_label"] = phase["label"]


        dic["_atom_site_fract_x"] = value_and_std(
            phase["xyz"][:, 0], phase["xyz","std"][:, 0]
        )

        dic["_atom_site_fract_y"] = value_and_std(
            phase["xyz"][:, 1], phase["xyz","std"][:, 1]
        )
        dic["_atom_site_fract_z"] = value_and_std(
            phase["xyz"][:, 2], phase["xyz","std"][:, 2]
        )

        dic["_atom_site_U_iso_or_equiv"] = value_and_std(
            phase["adp"], phase["adp","std"]
        )
        dic["_atom_site_occupancy"] = value_and_std(
            phase["occ"], phase["occ","std"]
        )

    elif save_type == "lattice_sites":
        dic["_symmetry_Int_Tables_number"] = "1"
        dic["_symmetry_space_group_name_H-M"] = "P1"
        dic["_symmetry_cell_setting"] = "cartesian"

        if phase.lattice_sites.get("lattice_error") == None:
            dic["_cell_length_a"] = value_and_std(
                phase.lattice_sites["lattice"][0], None
            )
            dic["_cell_length_b"] = value_and_std(
                phase.lattice_sites["lattice"][1], None
            )
            dic["_cell_length_c"] = value_and_std(
                phase.lattice_sites["lattice"][2], None
            )
            dic["_cell_angle_alpha"] = value_and_std(
                phase.lattice_sites["lattice"][3], None
            )
            dic["_cell_angle_beta"] = value_and_std(
                phase.lattice_sites["lattice"][4], None
            )
            dic["_cell_angle_gamma"] = value_and_std(
                phase.lattice_sites["lattice"][5], None
            )
        else:
            dic["_cell_length_a"] = value_and_std(
                phase.lattice_sites["lattice"][0],
                phase.lattice_sites["lattice_error"][0],
            )
            dic["_cell_length_b"] = value_and_std(
                phase.lattice_sites["lattice"][1],
                phase.lattice_sites["lattice_error"][1],
            )
            dic["_cell_length_c"] = value_and_std(
                phase.lattice_sites["lattice"][2],
                phase.lattice_sites["lattice_error"][2],
            )
            dic["_cell_angle_alpha"] = value_and_std(
                phase.lattice_sites["lattice"][3],
                phase.lattice_sites["lattice_error"][3],
            )
            dic["_cell_angle_beta"] = value_and_std(
                phase.lattice_sites["lattice"][4],
                phase.lattice_sites["lattice_error"][4],
            )
            dic["_cell_angle_gamma"] = value_and_std(
                phase.lattice_sites["lattice"][5],
                phase.lattice_sites["lattice_error"][5],
            )

        dic["_atom_site_type_symbol"] = phase.lattice_sites["symbol"]
        dic["_atom_site_label"] = phase.lattice_sites["label"]

        if phase.lattice_sites.get("xyz_error") == None:
            dic["_atom_site_fract_x"] = value_and_std(
                phase.lattice_sites["xyz"][:, 0], None
            )
            dic["_atom_site_fract_y"] = value_and_std(
                phase.lattice_sites["xyz"][:, 1], None
            )
            dic["_atom_site_fract_z"] = value_and_std(
                phase.lattice_sites["xyz"][:, 2], None
            )
        else:
            dic["_atom_site_fract_x"] = value_and_std(
                phase.lattice_sites["xyz"][:, 0],
                phase.lattice_sites["xyz_error"][:, 0],
            )
            dic["_atom_site_fract_y"] = value_and_std(
                phase.lattice_sites["xyz"][:, 1],
                phase.lattice_sites["xyz_error"][:, 1],
            )
            dic["_atom_site_fract_z"] = value_and_std(
                phase.lattice_sites["xyz"][:, 2],
                phase.lattice_sites["xyz_error"][:, 2],
            )

        if phase.lattice_sites.get("adp_error") == None:
            dic["_atom_site_U_iso_or_equiv"] = value_and_std(
                phase.lattice_sites["adp"], None
            )
        else:
            dic["_atom_site_U_iso_or_equiv"] = value_and_std(
                phase.lattice_sites["adp"], phase.lattice_sites["adp_error"]
            )

        if phase.lattice_sites.get("occ_error") == None:
            dic["_atom_site_occupancy"] = value_and_std(
                phase.lattice_sites["occ"], None
            )
        else:
            dic["_atom_site_occupancy"] = value_and_std(
                phase.lattice_sites["occ"], phase.lattice_sites["occ_error"]
            )

    elif save_type == "model":
        dic["_symmetry_Int_Tables_number"] = "1"
        dic["_symmetry_space_group_name_H-M"] = "P1"
        dic["_symmetry_cell_setting"] = "cartesian"

        if phase.model.get("lattice_error") == None:
            dic["_cell_length_a"] = value_and_std(
                phase.model["lattice"][0], None
            )
            dic["_cell_length_b"] = value_and_std(
                phase.model["lattice"][1], None
            )
            dic["_cell_length_c"] = value_and_std(
                phase.model["lattice"][2], None
            )
            dic["_cell_angle_alpha"] = value_and_std(
                phase.model["lattice"][3], None
            )
            dic["_cell_angle_beta"] = value_and_std(
                phase.model["lattice"][4], None
            )
            dic["_cell_angle_gamma"] = value_and_std(
                phase.model["lattice"][5], None
            )
        else:
            dic["_cell_length_a"] = value_and_std(
                phase.model["lattice"][0],
                phase.lattice_sites["all_error"][0],
            )
            dic["_cell_length_b"] = value_and_std(
                phase.model["lattice"][1],
                phase.lattice_sites["all_error"][1],
            )
            dic["_cell_length_c"] = value_and_std(
                phase.model["lattice"][2],
                phase.lattice_sites["all_error"][2],
            )
            dic["_cell_angle_alpha"] = value_and_std(
                phase.model["lattice"][3],
                phase.lattice_sites["all_error"][3],
            )
            dic["_cell_angle_beta"] = value_and_std(
                phase.model["lattice"][4],
                phase.lattice_sites["all_error"][4],
            )
            dic["_cell_angle_gamma"] = value_and_std(
                phase.model["lattice"][5],
                phase.lattice_sites["all_error"][5],
            )

        dic["_atom_site_type_symbol"] = phase.model["symbol"]
        dic["_atom_site_label"] = phase.model["label"]

        if phase.model.get("xyz_error") == None:
            dic["_atom_site_fract_x"] = value_and_std(
                phase.model["xyz"][:, 0], None
            )
            dic["_atom_site_fract_y"] = value_and_std(
                phase.model["xyz"][:, 1], None
            )
            dic["_atom_site_fract_z"] = value_and_std(
                phase.model["xyz"][:, 2], None
            )
        else:
            dic["_atom_site_fract_x"] = value_and_std(
                phase.model["xyz"][:, 0],
                phase.model["xyz_error"][:, 0],
            )
            dic["_atom_site_fract_y"] = value_and_std(
                phase.model["xyz"][:, 1],
                phase.model["xyz_error"][:, 1],
            )
            dic["_atom_site_fract_z"] = value_and_std(
                phase.model["xyz"][:, 2],
                phase.model["xyz_error"][:, 2],
            )

        if phase.model.get("adp_error") == None:
            dic["_atom_site_U_iso_or_equiv"] = value_and_std(
                phase.model["adp"], None
            )
        else:
            dic["_atom_site_U_iso_or_equiv"] = value_and_std(
                phase.model["adp"], phase.model["adp_error"]
            )

        if phase.model.get("occ_error") == None:
            dic["_atom_site_occupancy"] = value_and_std(
                phase.model["occ"], None
            )
        else:
            dic["_atom_site_occupancy"] = value_and_std(
                phase.model["occ"], phase.model["occ_error"]
            )

    delimiter = "  "
    f = open(file, "w")

    # header part
    # "_refine_ls_wR_factor_ref    "+'{:.4f}'.format(Rw),"_refine_ls_goodness_of_fit_ref    "+'{:.4f}'.format(GOF)+"\n"]
    f.write("data_" + os.path.basename(file))
    f.write("\n")
    f.write("_audit_creation_date" + delimiter + str(datetime.date.today()))
    f.write("\n")
    f.write("_audit_creation_method" + delimiter + "Orochi program")
    f.write("\n")

    # cell
    keys = [
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    ]
    for key in keys:
        f.write(key + delimiter + dic[key] + "\n")

    keys = [
        "_symmetry_space_group_name_H-M",
        "_symmetry_Int_Tables_number",
        "_symmetry_cell_setting",
    ]
    for key in keys:
        f.write(key + delimiter + dic[key] + "\n")

    # sites
    f.write("\n")
    f.write("loop_")
    f.write("\n")

    keys = [
        "_atom_site_type_symbol",
        "_atom_site_label",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_U_iso_or_equiv",
        "_atom_site_occupancy",
    ]
    for key in keys:
        f.write(key + "\n")

    for n, _ in enumerate(dic["_atom_site_type_symbol"]):
        for key in keys:
            f.write(dic[key][n] + delimiter)
        f.write("\n")

    f.close()

