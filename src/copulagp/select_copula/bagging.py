from copulagp.bvcopula import MixtureCopula
from copulagp.vine import CVine
from copulagp.bvcopula import Pair_CopulaGP_data
import torch


def bagged_copula(
    copula_data_list: list,
    n_estimators: int,
    X: torch.Tensor,
    device: torch.device = torch.device("cpu"),
):
    """
    Estimate the copula via pre-trained copula-GP object.
    Calculates average mixed copula via averaging mixing
    and theta params (copula variant wise)
    -----------------
    copula_data_list : List( Pair_CopulaGP_data )
        List of Copula-GP Data objects.
        Marginalizing along X gives a distribution over X.
    X : torch.Tensor on range [0,1]
        Tensor of marginalizing variable.
    device : torch.device
        Device ot marginalize on.
    """
    cop_datas = copula_data_list
    assert len(cop_datas) == n_estimators
    N = 0
    cop_indeces = dict()
    cop_combo_indeces = dict()
    cop_combinations = set()
    cop_counts = dict()
    rotations = []

    # Get Rotation and Index Information
    for i, cop_data in enumerate(cop_datas):
        for n, cop in enumerate(cop_data.bvcopulas):
            cop_combo = (cop[0], cop[1])
            if cop_combo not in cop_combinations:
                cop_combinations.add(cop_combo)
                cop_combo_indeces[cop_combo] = N
                cop_counts[N] = 0.0
                N = N + 1
                rotations.append(cop_combo[1])
            idx = cop_combo_indeces[cop_combo]
            cop_counts[idx] = cop_counts[idx] + 1.0
            cop_indeces[(i, n)] = idx

    # Marginalize
    cops = [
        cop_data.model_init(device).marginalize(torch.Tensor(X).to(device))
        for cop_data in cop_datas
    ]

    # Create Mixture as Average
    cop_list = [None for i in range(N)]
    thetas = torch.zeros((N, X.shape[0]))
    mixes = torch.zeros((N, X.shape[0]))

    for i, cop in enumerate(cops):
        for n, cop_type in enumerate(cop.copulas):
            idx = cop_indeces[(i, n)]
            cop_list[idx] = cop_type
            thetas[idx] = thetas[idx] + cop.theta[n] / cop_counts[idx]
            mixes[idx] = mixes[idx] + cop.mix[n] / len(cop_datas)

    print(cop_list)

    return MixtureCopula(
        theta=thetas,
        mix=mixes.clamp(0.001, 0.999),
        copulas=cop_list,
        rotations=rotations,
    )


def bagged_vine(
    vines_data: list, X: torch.Tensor, device: torch.device = torch.device("cpu")
):
    """
    Given a list of vine predictor model layer lists,
    get the mean vine via bagging each copula.
    ----------------
    vines_data: List( List( Pair_CopulaGP_data ) )
        list of vines to aggregate.
    X: torch.Tensor
        Predicting variable.
    device: torch.device
        device to marginalize / return on.
    """
    assert vines_data[0][0][0] is Pair_CopulaGP_data
    n_estimators = len(vines_data)
    dim = len(vines_data[0] + 1)
    bagged_copulas = [[[] for j in range(dim - 1 - i)] for i in range(dim - 1)]

    for models_i in vines_data:
        for l, layer in enumerate(models_i):
            for n, copula in enumerate(layer):
                bagged_copulas[l][n].append(copula)

    for l, layer in enumerate(bagged_copulas):
        for n, copula_data_list in enumerate(layer):
            bagged_copulas[l][n] = bagged_copula(
                copula_data_list, n_estimators, torch.Tensor(X), device=device
            )

    mean_vine = CVine(bagged_copulas, torch.Tensor(X).to(device), device=device)
    return mean_vine
