import numpy as np

"""need maintainance
"""

def adjustment_based_on_scores(
    phase, key, score_function, adjust_function, report_name
):

    def decide_adjust_index(scores, score_dimension):

        if score_dimension == 0:  # one value object
            adjust_ids = []

        elif score_dimension == 1:  # 1d array
            mu = np.mean(scores)
            sigma = np.std(scores)
            adjust_ids = np.where((scores >= mu))[0]

        elif score_dimension == 2:  # 2D array
            adjust_ids = []

        return adjust_ids

    scores = score_function(phase)
    score_dimension = np.size(np.shape(scores))
    adjust_ids = decide_adjust_index(scores, score_dimension)

    params_generator = adjust_parameters_in_index_list(
        trial, phase, adjust_function, adjust_ids, key, score_dimension
    )

    for adjust_id in adjust_ids:
        is_change, fixed_params = next(params_generator)

        if is_change:
            study.enqueue_trial(
                params=fixed_params,
                user_attrs={
                    "report_name": report_name,
                    report_name: key + "_" + str(adjust_id),
                },
                skip_if_exists=True,
            )



def adjust_parameters_in_index_list(
    trial, module, adjust_function, adjust_ids, score_key, score_dimension
):

    def set_proposal_into_fixed_params(fixed_params, key, proposal) -> bool:
        def single_value(fixed_params, key, proposal):
            fixed_params[key] = proposal

            if trial.params[key] != proposal:
                return True
            else:
                return False

        is_same = True

        if isinstance(proposal, (list, np.ndarray)):
            for id2, prop_value in enumerate(proposal):
                name = key + "_" + str(id2)
                is_same *= single_value(fixed_params, name, prop_value)
        elif proposal == None:
            pass
        else:
            is_same *= single_value(fixed_params, key, proposal)

        return is_same

    for adjust_id in adjust_ids:
        fixed_params = {}

        for param_key, param_value in list(trial.params.items()):
            if param_key in fixed_params:
                continue

            unpacked = param_key.split("_")
            module_id = unpacked[0]
            unpacked_key = unpacked[1]
            param_dim = len(unpacked) - 2

            if unpacked_key == score_key:

                if param_dim == 0:
                    local_module = copy.deepcopy(module)
                    initial_param = trial.user_attrs["initial_params"][
                        param_key
                    ]
                    local_module.par.par[unpacked_key]["value"] = initial_param
                    function = adjust_function(local_module)
                    is_same = set_proposal_into_fixed_params(
                        fixed_params, param_key, proposal=function()
                    )
                    if is_same:
                        yield False, fixed_params
                    continue

                elif param_dim == 1:
                    if score_dimension == 1:
                        id1 = int(unpacked[2])
                        if id1 == adjust_id:

                            local_module = copy.deepcopy(module)
                            initial_param = trial.user_attrs["initial_params"][
                                param_key
                            ]
                            local_module.par.par[unpacked_key]["value"][
                                id1
                            ] = initial_param
                            function = adjust_function(local_module)
                            is_same = set_proposal_into_fixed_params(
                                fixed_params, param_key, proposal=function(id1)
                            )
                            if is_same:
                                yield False, fixed_params
                            continue
                        else:
                            fixed_params[param_key] = param_value

                elif param_dim == 2:
                    if score_dimension == 1:
                        id1 = int(unpacked[2])
                        if id1 == adjust_id:

                            local_module = copy.deepcopy(module)
                            common_key = "_".join(unpacked[:-1])
                            for initial_params_key in trial.user_attrs[
                                "initial_params"
                            ]:
                                if initial_params_key.startswith(common_key):
                                    id2 = initial_params_key.strip(
                                        common_key
                                    ).strip("_")
                                    initial_param = trial.user_attrs[
                                        "initial_params"
                                    ][initial_params_key]
                                    local_module.par.par[unpacked_key][
                                        "value"
                                    ][id1, int(id2)] = initial_param
                                else:
                                    continue

                            function = adjust_function(local_module)
                            is_same = set_proposal_into_fixed_params(
                                fixed_params,
                                common_key,
                                proposal=function(id1),
                            )
                            if is_same:
                                yield False, fixed_params
                            continue

                        else:
                            fixed_params[param_key] = param_value

                    if score_dimension == 2:
                        # this wont match the single value of adjust value. that should be in a list.
                        pass

                else:
                    # this will be made in the future
                    pass
            else:
                fixed_params[param_key] = param_value

        yield True, fixed_params
