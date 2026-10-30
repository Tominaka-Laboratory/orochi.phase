import numpy as np

par = {
    "file": {"value": None},
    # "viewer": {"value": None},
    "composition": {"value": [], "nominal": {}, "model": {}},
    "merge_distance": {"value": [0.1]},
    "symbol": {"value": np.array(["X1"])},
    "label": {"value": np.array(["X1"])},
    "xyz": {
        "value": np.array([[0.0, 0.0, 0.0]]),
        "min": np.array([[0.0, 0.0, 0.0]]),
        "max": np.array([[1.0, 1.0, 1.0]]),
        "fit": np.array([[False, False, False]]),
        "std": np.array([[0.0, 0.0, 0.0]]),
    },
    "occ": {
        "value": np.array([1.0]),
        "min": np.array([0.0]),
        "max": np.array([2.0]),
        "fit": np.array([False]),
        "std": np.array([0.0]),
    },
    "adp": {
        "value": np.array([0.01]),
        "min": np.array([0.001]),
        "max": np.array([0.3]),
        "fit": np.array([False]),
        "std": np.array([0.0]),
    },
    "group": {"value": np.array([0])},
    "group_coord": {"value": np.zeros(6), #center of mass x y z + rotation angles 
                    "min": np.zeros(6),
                    "max": np.array([1.0, 1.0, 1.0, 360.0, 360.0, 360.0]),
                    "fit": np.zeros(6, dtype=bool),
                    "std": np.zeros(6),
                    },
    "group_axes": {"value": np.eye(3)}, #roration axes for the groups
    "multiplicity": {"value": np.array([1])},
    "charge": {"value": np.array([0.0])},
    "radius": {"value": np.array([0.0])},
    "cn": {"value": []},
    "wyckoff_letters": {"value": np.array(["a"], dtype=str),
                        "fit": np.array([False]),
                        "candidates": [[]],
                        },
    # "site_symmetry_symbols": {
    #     "value": np.array(["m-3m"])
    # },
    # "site_symmetry_operations": {
    #     "value": np.array([["0","0","0"]])
    # },
    "density": {
        "value": 0.0,
        "min": 0.0,
        "max": 0.0,
        "fit": False,
    },
    "lattice": {
        "value": np.array(
            [4.0, 4.0, 4.0, 90.0, 90.0, 90.0]
        ),
        "min": np.array([0.1, 0.1, 0.1, 50, 50, 50]),
        "max": np.array(
            [50.0, 50.0, 50.0, 130.0, 130.0, 130.0]
        ),
        "fit": np.array(
            [False, False, False, False, False, False]
        ),
        "std": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "centroid": [True],
        "center": [0.5, 0.5, 0.5],
    },
    # "lattice_vector": {"value": np.ones((3, 3))},
    # "lattice_system": {"value": "triclinic"},
    "space_group": {
        "value": ["P1"], 
    },
    "hall_number": {
        "value": [1], #hall number
        "min": 1,
        "max": 530,
        "fit":[False],
        "opt_func":"suggest_subgroup_hall_number",
        "opt_func_params":{}
    },
    "symprec":{
        "value":[0.001],
    },
    "point_group":{"value":"1"},
    "modulation":{
        "value":[None],
        "function":None,
    },
    "translation_vector":{
        "value":[0.,0.,0.],
    },
    "rotation_matrix":{
        "value":np.eye(3),
    },
    "rotatoin_center":{
        "value":[0.,0.,0.],
    },

    "pbc": {"value": [True, True, True]},
    "map": {"value": np.array([0.0]), "xyz": np.array([0.0, 0.0, 0])},
    "material_type": {"value": [""]},
    "shape_type": {"value": "analytical", "member":["analytical", "model", "polygon", "sphere"]},  # or model, polygon, sphere
    "sphere": {
        "value": np.array([10.0, 10.0, 10.0]),
        "min": [0.1, 0.1, 0.1],
        "max": [100, 100, 100],
        "fit": np.array([False, False, False]),
        "std": np.array([0.0, 0.0, 0.0]),
        "origin": np.array([0, 0, 0]),
        "dimension": [3],
    },  # radii along a, b, c #221104 Tominaka (add error)
    "polygon": {
        "value": np.array(
            [6.0, 6.0, 6.0, 6.0, 6.0, 46.0]
        ),
        "min": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "max": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "fit": np.array(
            [False, False, False, False, False, False]
        ),
        "std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "origin": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "direction": [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        "max_length": [100.0],
    },
    "cylinder": {
        "value": np.array([6.0, 46.0]),
        "min": [1.0, 1.0],
        "max": [100.0, 100.0],
        "fit": np.array([False, False]),
        "std": [0.0, 0.0],
        "origin": [[0.0, 0.0, 0.0]],
        "direction": [[0.0, 0.0, 1.0]],
        "max_length": [100.0],
    },
    "surface": {
        "value": np.array([4.0, 4.0, 4.0]),
        "min": [1.0, 1.0, 1.0],
        "max": [
            100.0,
            100.0,
            100,
        ],
        "fit": np.array([False, False, False]),
        "std": [0.0, 0.0, 0.0],
        "substitute": [],
        "coordination": [],
    },
    "spherical_constraints":{
        "value":True
    },
    "symmetry_constraints":{
        "value":True
    },
    "lattice_constraints":{
        "value":True
    }
}

par_key_info = {  # key:{atribute:[type, index, shape]}
    "file": {"value": [str, None, 1]},
    # "viewer": {"value": [object, None, 1]},
    "composition": {
        "value": [str, None, 1],
        "nominal": [dict, None, 1],
        "model": [dict, None, 1],
    },
    "merge_distance": {"value": [float, None, 1]},
    "symbol": {"value": [str, None, (-1)]},
    "label": {"value": [str, None, (-1)]},
    "xyz": {
        "value": [float, None, (-1, 3)],
        "min": [float, None, (-1, 3)],
        "max": [float, None, (-1, 3)],
        "free": [int, None, (-1, 3)],
        "fit": [bool, None, (-1, 3)],
        "std": [float, None, (-1, 3)],
    },
    "adp": {
        "value": [float, None, (-1)],
        "min": [float, None, (-1)],
        "max": [float, None, (-1)],
        "fit": [bool, None, (-1)],
        "std": [float, None, (-1)],
    },
    "occ": {
        "value": [float, None, (-1)],
        "min": [float, None, (-1)],
        "max": [float, None, (-1)],
        "fit": [bool, None, (-1)],
        "std": [float, None, (-1)],
    },
    "group": {"value": [int, None, (-1)]},
    "group_coord": {"value": [float, None, (-1, 6)],
                    "min": [float, None, (-1, 6)],
                    "max": [float, None, (-1, 6)],
                    "fit": [bool, None, (-1, 6)],
                    "std": [float, None, (-1, 6)],
                    },
    "group_axes": {"value": [float, None, (-1, 3, 3)]},
    "multiplicity": {
        "value": [float, None, (-1)],
        "min": [float, None, (-1)],
        "max": [float, None, (-1)],
        "fit": [bool, None, (-1)],
        "std": [float, None, (-1)],
    },
    "charge": {
        "value": [float, None, (-1)],
        "min": [float, None, (-1)],
        "max": [float, None, (-1)],
        "fit": [bool, None, (-1)],
        "std": [float, None, (-1)],
    },
    "radius": {
        "value": [float, None, (-1)],
        "min": [float, None, (-1)],
        "max": [float, None, (-1)],
        "fit": [bool, None, (-1)],
        "std": [float, None, (-1)],
    },
    "cn": {"value": [list, None, (-1)]},
    "wyckoff_letters": {"value": [str, None, (-1)],
                        "fit": [bool, None, (-1)],
                        "candidates":[list, None, (-1)],
                        },
    # "site_symmetry_symbols": {"value": [str, None, (-1)]},
    # "site_symmetry_operations": {"value": [str, None, (-1,3)]},
    "density": {
        "value": [float, None, 1],
        "min": [float, None, 1],
        "max": [float, None, 1],
        "fit": [bool, None, 1],
    },
    "lattice": {
        "value": [float, None, (-1)],
        "min": [float, None, (-1)],
        "max": [float, None, (-1)],
        "fit": [bool, None, (-1)],
        "std": [float, None, (-1)],
        "centroid": [bool, None, (-1)],
        "center": [float, None, (-1)],
    },
    # "lattice_vector": {"value": [float, None, (3, 3)]},
    # "lattice_system": {"value": [str, None, 1]},
    "space_group": {"value": [str, None, 1]},
    "hall_number": {"value": [int, None, 1],
                    "min": [int, None, 1],
                    "max": [int, None, 1],
                    "fit": [bool, None, 1],
                    "opt_func": [str, None, 1],
                    "opt_func_params":[dict, None, 1]
                    },
    "symprec": {
        "value": [float, None, 1],
    },

    "modulation":{
        "value":None,
        "function":None,
    },
    "translation_vector":{
        "value":[float, None, 3],
    },
    "rotation_matrix":{
        "value":[float, None, (3,3)],
    },
    "rotatoin_center":{
        "value":[float, None, 3],
    },
    "pbc": {"value": [bool, None, (-1)]},
    "map": {
        "value": [float, None, (-1, -1, -1)],
        "xyz": [float, None, (-1, -1, -1, 3)],
    },
    "material_type": {"value": [str, None, 1]},
    "shape_type": {"value": [str, None, 1]},
    "sphere": {
        "value": [float, None, (-1)],
        "min": [float, None, (-1)],
        "max": [float, None, (-1)],
        "fit": [bool, None, (-1)],
        "std": [float, None, (-1)],
        "origin": [float, None, (-1, 3)],
        "dimension": [float],
    },
    "polygon": {
        "value": [float, None, (-1)],
        "min": [float, None, (-1)],
        "max": [float, None, (-1)],
        "fit": [bool, None, (-1)],
        "std": [float, None, (-1)],
        "origin": [float, None, (-1, 3)],
        "direction": [float, None, (-1, 3)],
        "max_length": [float],
    },
    "cylinder": {
        "value": [float, None, (-1)],
        "min": [float, None, (-1)],
        "max": [float, None, (-1)],
        "fit": [bool, None, (-1)],
        "std": [float, None, (-1)],
        "origin": [float, None, (-1, 3)],
        "direction": [float, None, (-1)],
        "max_length": [float],
    },
    "surface": {
        "value": [float, None, (-1)],
        "min": [float, None, (-1)],
        "max": [float, None, (-1)],
        "fit": [bool, None, (-1)],
        "std": [float, None, (-1)],
        "substitute": [str, None, (-1, 2)],
        "coordination": [float, None, (-1, 4)],
    },
    "fit": {"value": [str, None, 1]},
    "spherical_constraints":{
        "value":[bool, None, 1]
    },
    "symmetry_constraints":{
        "value":[bool, None, 1]
    },
    "lattice_constraints":{
        "value":[bool, None, 1]
    }
}

