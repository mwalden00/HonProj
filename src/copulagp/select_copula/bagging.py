from copulagp.bvcopula import MixtureCopula
from copulagp.vine import CVine
from copulagp.bvcopula import Pair_CopulaGP_data
import numpy as np
import torch


def bagged_copula(
    copula_data_list: list,
    n_estimators: int,
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    R2_atol: int = 0.01,
    how: str = "AIC dynamic",
    rsample_size: int = 0,
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
        Tensor of parameterizing variable
    Y : torch.Tensor on range(0,1). Shape of (n samples) x (dimensions)
        Tensors of marginal values for sample. Used to get ecdf.
    device : torch.device
        Device ot marginalize on.
    how : str = "BIC dynamic" | "BIC static" | "AIC dynamic" | "AIC static" | "R2" | "mean"
        Method of bag weighting.
    rsample_size : int = -1
        Size of rsampling for log_prob. Defaults to size of X if size is leq. 0.
    R2_atol : int = 0.01
        Tolerance for deciding how deviated from the max R2 score found we wish to drop models.
    """
    if rsample_size <= 0:
        rsample_size = X.shape[0]
    methods = ["mean", "R2", "BIC dynamic", "BIC static", "AIC dynamic", "AIC static"]
    if how not in methods:
        raise ValueError("argument 'how' must be one of 'mean', 'R2', or 'BIC'")
    assert len(copula_data_list) == n_estimators and n_estimators > 1
    with torch.device(device=device):
        cop_datas = copula_data_list
        cops = [
            cop_data.model_init(device).marginalize(torch.Tensor(X).to(device))
            for cop_data in cop_datas
        ]

        buckets = torch.arange(X.shape[0]).chunk(20)

        # To get R2 terms, we first find empirical cdf along our points
        # and the empirical copula cdf along one axis of the copula.
        # We do this along the buckets.
        def ecdf(i):
            """Empirical CDF in bucket i."""
            vals = []
            for y2 in Y[1][buckets[i]]:
                vals.append(
                    len(Y[0][buckets[i]][Y[1][buckets[i]] < y2])
                    / (len(Y[0][buckets[i]]))
                )
            return vals

        def eccdf(cop, i):
            """Empirical Copula CDF in bucket i utilizing copula samples."""
            Y0_sample = cop.sample()[:, 0]
            vals = []
            for y2 in Y[1][buckets[i]]:
                vals.append(len(Y0_sample[Y[1][buckets[i]] < y2]) / (len(Y0_sample)))
            return vals

        # Get CCDFs. Involves marginalizing
        cop_ccdfs = [
            torch.vstack(
                [
                    torch.Tensor(
                        eccdf(cop_data.model_init(device).marginalize(X[buckets[i]]), i)
                    )
                    for i in range(20)
                ]
            )
            for cop_data in cop_datas
        ]

        # Lets collect the bucket copulas.
        import gc

        gc.collect()

        ecdfs = torch.vstack([torch.Tensor(ecdf(i)) for i in range(20)])
        R2s = torch.Tensor(
            [
                (
                    1
                    - (
                        ((ecdfs - ccdfs) ** 2) / ((ecdfs - 0.5) ** 2).clamp(0.001, 1)
                    ).sum(axis=1)
                ).mean()
                for ccdfs in cop_ccdfs
            ]
        )

        # If there is a single best fit, we return that.
        R2s = R2s[R2s >= (R2s.max() - R2_atol)]
        if len(R2s) == 1:
            print("Single best copula found.")
            return cops[np.argmax(R2s)]

        if how == "AIC dynamic":
            # We now weight the remaining copulas lineary via Bayesian Info Criterion.
            # AIC is defined as -2 * log likelihood + (# of params) * 2
            # For # params we have mixing params + theta params.
            # We get the overall mean, creating a static model.
            # We average cop.log_prob along axis 0 as it outputs log_prob of mixed copuls * weights.
            # This also means we aggregate dynamically in X
            # Log prob is mem. intensive, so we should cleanup.
            AICs = torch.vstack(
                [
                    -2 * cop.log_prob(Y.T) + (cop.mix.shape[0] + cop.theta.shape[0]) * 2
                    for cop in cops
                ]
            )
            # print("BICs: ", BICs)
            weights = torch.exp(-0.5 * AICs) / (torch.exp(-0.5 * AICs)).sum(axis=0)
            weights = weights / weights.sum(axis=0)
            assert torch.allclose(weights.sum(axis=0), torch.ones(weights.shape[1]))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif how == "AIC static":
            # We now weight the remaining copulas lineary via Bayesian Info Criterion.
            # AIC is defined as -2 * log likelihood + (# of params) * 2
            # For # params we have mixing params + theta params.
            # We get the overall mean, creating a static model.
            AICs = torch.vstack(
                [
                    -2 * cop.log_prob(Y.T).mean()
                    + (cop.mix.shape[0] + cop.theta.shape[0]) * 2
                    for cop in cops
                ]
            )
            # print("BICs: ", BICs)
            weights = torch.exp(-0.5 * AICs) / torch.exp(-0.5 * AICs).sum()
            assert torch.allclose(weights.sum(), torch.ones(1))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif how == "BIC dynamic":
            # We now weight the remaining copulas lineary via Bayesian Info Criterion.
            # BIC is defined as -2 * log likelihood + (# of params) * log(sample size)
            # We use the weights exp(-1/2 * BICs[i]) / sum(BICs)
            # For # params we have mixing params + theta params.
            # We average cop.log_prob along axis 0 as it outputs log_prob of mixed copuls * weights.
            # This also means we aggregate dynamically in X
            # Log prob is mem. intensive, so we should cleanup.
            BICs = torch.vstack(
                [
                    -2 * cop.log_prob(Y.T)
                    + (cop.mix.shape[0] + cop.theta.shape[0]) * np.log(len(X.shape))
                    for cop in cops
                ]
            )
            # print("BICs: ", BICs)
            weights = torch.exp(-0.5 * BICs) / (torch.exp(-0.5 * BICs)).sum(axis=0)
            weights = weights / weights.sum(axis=0)
            assert torch.allclose(weights.sum(axis=0), torch.ones(weights.shape[1]))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif how == "BIC static":
            # We now weight the remaining copulas lineary via Bayesian Info Criterion.
            # BIC is defined as -2 * log likelihood + (# of params) * log(sample size)
            # For # params we have mixing params + theta params.
            # We get the overall mean, creating a static model.
            BICs = torch.vstack(
                [
                    -2 * cop.log_prob(Y.T).mean()
                    + (cop.mix.shape[0] + cop.theta.shape[0]) * np.log(len(X.shape))
                    for cop in cops
                ]
            )
            # print("BICs: ", BICs)
            weights = torch.exp(-0.5 * BICs) / torch.exp(-0.5 * BICs).sum()
            assert torch.allclose(weights.sum(), torch.ones(1))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif how == "R2":
            # Simple weight by model fitness score. Static
            weights = R2s / R2s.sum()
        else:
            weights = torch.ones(len(cops)) / len(cops)

        assert len(cop_datas) == n_estimators
        N = 0
        cop_indeces = dict()
        cop_combo_indeces = dict()
        cop_combinations = set()
        cop_counts = dict()
        mix_total_weight = dict()
        rotations = []

        # Get rotation and index data
        for i, cop_data in enumerate(cop_datas):
            for n, cop in enumerate(cop_data.bvcopulas):
                cop_combo = (cop[0], cop[1])
                if cop_combo not in cop_combinations:
                    cop_combinations.add(cop_combo)
                    cop_combo_indeces[cop_combo] = N
                    cop_counts[N] = 0.0
                    mix_total_weight[N] = 0.0
                    N = N + 1
                    rotations.append(cop_combo[1])
                idx = cop_combo_indeces[cop_combo]
                cop_counts[idx] += 1.0
                cop_indeces[(i, n)] = idx
                mix_total_weight[idx] += weights[i]

        # Create mixture params as weighted average
        mix_list = [None for i in range(N)]
        thetas = torch.zeros((N, X.shape[0]))
        mixes = torch.zeros((N, X.shape[0]))

        for i, cop in enumerate(cops):
            for n, cop_type in enumerate(cop.copulas):
                idx = cop_indeces[(i, n)]
                mix_list[idx] = cop_type
                mixes[idx] += cop.mix[n] * weights[i]
                thetas[idx] += cop.theta[n] * weights[i] / mix_total_weight[idx]

        print(mix_list)

        return (
            MixtureCopula(
                theta=thetas,
                mix=mixes.clamp(0.001, 0.999),
                copulas=mix_list,
                rotations=rotations,
            ),
            weights,
        )


def bagged_vine(
    vines_data: list,
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    how: str = "BIC dynamic",
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
    n_estimators = len(vines_data)
    dim = len(vines_data[0]) + 1

    bagged_copulas = [[[] for j in range(dim - 1 - i)] for i in range(dim - 1)]

    for models_i in vines_data:
        for l, layer in enumerate(models_i):
            for n, copula in enumerate(layer):
                bagged_copulas[l][n].append(copula)

    layers = [Y]
    for l, layer in enumerate(bagged_copulas):
        next_layer = []
        for n, copula_data_list in enumerate(layer):
            # This is essentially the same thing as is done in copulagp.vine.log_prob
            bagged_copulas[l][n], _ = bagged_copula(
                copula_data_list,
                n_estimators,
                X,
                layers[-1][:, [n + 1, 0]].T,
                device=device,
                how=how,
            )
            # We get the CCDF of the bagged copula, and then feed those to the next bagger
            # This gets the current fit dependant probability.
            next_layer.append(
                bagged_copulas[l][n].ccdf(layers[-1][:, [n + 1, 0]]).float()
            )
        layers.appedn(torch.stack(next_layer, dim=-1))

    mean_vine = CVine(bagged_copulas, torch.Tensor(X).to(device), device=device)
    return mean_vine
