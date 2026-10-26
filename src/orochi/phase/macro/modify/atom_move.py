from ..function import constraints
import numpy as np


def random_move_without_collision(
    phase, max_move_radius, fix_coordinates: list = [[0, 0]]
):
    sites = phase["xyz"]
    const = constraints.max_move_without_collision_to_youngers_constraints(
        max_move_radius=max_move_radius
    )(phase, "xyz")
    rng = np.random.default_rng()
    fix_coordinates = np.array(fix_coordinates).reshape(-1, 2)

    for id1, site in enumerate(sites):
        if id1 == 0:
            continue

        c = const(id1)
        for id2, coordinate in enumerate(site):
            if np.sum(
                np.where(
                    (fix_coordinates[:, 0] == id1)
                    & (fix_coordinates[:, 1] == id2),
                    1,
                    0,
                )
            ):
                continue

            min, max = c(id2)
            phase["xyz", "value", (id1, id2)] = (
                rng.random() * (max - min) + min
            )

    phase["xyz"] -= np.floor(phase["xyz"])
