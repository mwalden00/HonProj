import numpy as np
import torch
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import copy
from copulagp.utils import Plot_Fit
from copulagp import vine as v
from copulagp.train import train_vine
from copulagp.bvcopula import MixtureCopula
from copulagp.synthetic_data import get_random_vine
from copulagp.select_copula import bagged_vine
import os
import gc
from test_helpers import parser, clean, num_vine_params


device = "cpu" if not torch.cuda.is_available() else "cuda:0"

if torch.cuda.is_available():
    device_list = [f"cuda:{n}" for n in range(torch.cuda.device_count())]
else:
    device_list = ["cpu"]


if __name__ == "__main__":
    args = parser().parse_args()
    if args.seed > 0:
        seed = args.seed
    else:
        seed = torch.seed()
    print("Seed: ", seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    mp.set_start_method("spawn")
    dim = args.dim
    max_el = args.max_el
    shuffle_bags = args.shuffle_bags
    linear_input = args.linear_input
    shuffle_sample = args.shuffle
    light = args.light

    with torch.device(device):

        with open("../models/results/pupil_traj_5_res_partial.pkl", "rb") as f:
            pupil_results = pkl.load(f)

        X = torch.Tensor(np.random.normal(0.5, 0.2, 10000)).clamp(0.001, 0.999)
        if linear_input:
            X = torch.linspace(0, 1, 10000)

        pupil_vine = get_random_vine(
            dim, torch.Tensor(X[-10000:]), device=device, max_el=max_el
        )
        print("True vine: ", [[cop.copulas for cop in l] for l in pupil_vine.layers])

        clean()
        if args.skip_true_ent == 1:
            ent = np.genfromtxt(f"./true_ent_{dim}.csv", delimiter=",")
        else:
            layers = copy.deepcopy(pupil_vine.layers)
            for l, layer in enumerate(layers):
                for n, cop in enumerate(layer):
                    layers[l][n] = MixtureCopula(
                        theta=cop.theta[:, -2000:],
                        mix=cop.mix[:, -2000:],
                        copulas=cop.copulas,
                        rotations=cop.rotations,
                    )
            pup_vine_ent = v.CVine(layers=layers, inputs=X[-2000:], device=device)
            ent = pup_vine_ent.entropy().detach().cpu().numpy()
            ent.tofile(f"./true_ent_{dim}.csv", sep=",")
        print(f"Test entropy extraction: {ent.mean()} +/- {2*np.std(ent)}")

        n_estimators = args.n_estimators
        assert 4000 % n_estimators == 0

        print(f"Getting {n_estimators} copulaGP estimators...")
        Y = pupil_vine.sample()
        Y_train = Y[:-2000].reshape(n_estimators, int(8000 / n_estimators), dim)

        if shuffle_sample:
            perm = torch.randperm(10000).cpu()
            X = X[perm]
            Y = Y[perm]

        X = X[-10000:]
        if shuffle_bags == 0:
            X_train = X[:-2000].reshape(n_estimators, int(8000 / n_estimators))
        else:
            perm = torch.randperm(8000).cpu()
            X_train = X[perm].reshape(n_estimators, int(8000 / n_estimators))

        for i in range(args.bagged_start, n_estimators):
            try:
                os.mkdir(f"../models/layers/pupil_vine/segments/seg_{i}/")
                os.mkdir(f"../models/results/pupil_segments/")
                os.mkdir(f"../data/segmented_pupil_copulas/")
            except:
                pass

            # print(f"\nSelecting Trial {i} with trajectory choices {choices}")
            # np.savetxt(f"./segmented_pupil/choices/choice_i.txt", choices)

            Y_chosen = Y_train[i]
            with open(f"../data/segmented_pupil_copulas/data_{i}_0.pkl", "wb") as f:
                pkl.dump(
                    dict(
                        [
                            ("X", X_train[i]),
                            ("Y", Y_chosen.cpu().numpy()),
                        ]
                    ),
                    f,
                )

            print(f"Training {i}-th Copula-GP Vine")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_vine(
                path_data=lambda x: f"../data/segmented_pupil_copulas/data_{i}_{x}.pkl",
                path_models=lambda x: f"../models/layers/pupil_vine/segments/seg_{i}/layer_{x}.pkl",
                path_final=f"../models/results/pupil_segments/pupil_{i}_res.pkl",
                path_logs=lambda a, b: f"./segmented_pupil/{a}/layer_{b}",
                exp=f"Vine on trial {i} Parametrized in Pupil Area",
                light=light == 1,
                device_list=device_list,
                start=args.bagged_start_layer,
            )

        print("\n\nGetting Bagged Vine...")

        vines2bag = []

        for i in range(n_estimators):
            with open(f"../models/results/pupil_segments/pupil_{i}_res.pkl", "rb") as f:
                vines2bag.append(pkl.load(f)["models"])

        BIC_dynamic_vine = bagged_vine(
            vines_data=vines2bag,
            X=torch.Tensor(X).to(device)[-2000:],
            Y=Y[-2000:],
            device=device,
        )

        BIC_static_vine = bagged_vine(
            vines_data=vines2bag,
            X=torch.Tensor(X).to(device)[-2000:],
            Y=Y[-2000:],
            device=device,
            how="BIC static",
        )

        R2_meaned_vine = bagged_vine(
            vines_data=vines2bag,
            X=torch.Tensor(X).to(device)[-2000:],
            Y=Y[-2000:],
            device=device,
            how="R2",
        )

        clean()
        print("Getting Bagged Vine Entropy...")
        if args.skip_ent_bagged == 1:
            ent_BIC_dynamic = np.genfromtxt(
                f"./{n_estimators}_BIC_dyn_pred_{dim}.csv", delimiter=","
            )
        else:
            ent_BIC_dynamic = BIC_dynamic_vine.entropy().detach().cpu().numpy()
            ent_BIC_dynamic.tofile(f"./{n_estimators}_BIC_dyn_pred_{dim}.csv", sep=",")
        print(f"Entropy: {ent_BIC_dynamic.mean()} +/- {np.std(ent_BIC_dynamic)}")

        if args.skip_ent_bagged == 1:
            ent_BIC_static = np.genfromtxt(
                f"./{n_estimators}_BIC_static_pred_{dim}.csv", delimiter=","
            )
        else:
            ent_BIC_static = BIC_static_vine.entropy().detach().cpu().numpy()
            ent_BIC_static.tofile(
                f"./{n_estimators}_BIC_static_pred_{dim}.csv", sep=","
            )
        print(f"Entropy: {ent_BIC_static.mean()} +/- {np.std(ent_BIC_static)}")

        if args.skip_ent_bagged == 1:
            ent_R2_mean = np.genfromtxt(
                f"./{n_estimators}_R2_pred_{dim}.csv", delimiter=","
            )
        else:
            ent_R2_mean = R2_meaned_vine.entropy().detach().cpu().numpy()
            ent_R2_mean.tofile(f"./{n_estimators}_R2_pred_{dim}.csv", sep=",")
        print(f"Entropy: {ent_R2_mean.mean()} +/- {np.std(ent_R2_mean)}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        X_train = X[-10000:-2000]
        Y_train = Y[-10000:-2000]
        print(X_train.shape)
        print(Y_train.shape)
        baseline_data = dict([("X", X_train), ("Y", Y_train.cpu().numpy())])
        with open("../data/segmented_pupil_copulas/baseline_data_0.pkl", "wb") as f:
            pkl.dump(baseline_data, f)

        print("Getting Baseline")
        train_vine(
            path_data=lambda x: f"../data/segmented_pupil_copulas/baseline_data_{x}.pkl",
            path_models=lambda x: f"../models/layers/pupil_vine/segments/baseline/layer_{x}.pkl",
            path_final=f"../models/results/pupil_segments/baseline_res.pkl",
            path_logs=lambda a, b: f"./segmented_pupil/{a}/layer_{b}",
            exp=f"Baseline Vine on {dim} dim random copula data Parametrized in Pupil Area",
            light=light == 1,
            start=args.skip_baseline,
            device_list=device_list,
        )

        with open("../models/results/pupil_segments/baseline_res.pkl", "rb") as f:
            baseline_results = pkl.load(f)
        baseline_model_data = copy.deepcopy(baseline_results["models"])
        print(baseline_model_data)

        print("Getting Entropies...")
        clean()
        for i, layer in enumerate(baseline_model_data):
            for j, cop_data in enumerate(layer):
                cop = cop_data.model_init(device).marginalize(torch.Tensor(X)[-2000:])
                baseline_model_data[i][j] = cop
        baseline_vine = v.CVine(
            baseline_model_data, torch.Tensor(X)[-2000:], device=device
        )
        if args.skip_ent_baseline:
            baseline_ent = np.genfromtxt(f"./baseline_{dim}.csv", delimiter=",")
        else:
            baseline_ent = baseline_vine.entropy().detach().cpu().numpy()
            baseline_ent.tofile(f"./baseline_{dim}.csv", sep=",")

        print(f"Baseline ent: {baseline_ent.mean()} +/- {2*np.std(baseline_ent)}")

        print("================================================================")
        print("================================================================")
        print("True: \t{:.6f} +/- {:.6f}".format(ent.mean(), 2 * np.std(ent[-2000:])))

        def pprint_vine_copula_test(name, model, pred_ent):
            print(
                "Test Ent. MAE {}: \t{:.6f}\t| Test. BIC: {}".format(
                    name,
                    np.abs(ent.mean() - pred_ent).mean(),
                    -2 * model.log_prob(X[-2000:]).mean()
                    + (
                        len(model.layers)
                        * (len(model.layers) + 1)
                        * num_vine_params(model)
                    )
                    / 2
                    * 2000,
                )
            )

        if dim == 2:
            print("Single copula test.")
        else:
            print("Vine copula test.")
        pprint_vine_copula_test("Baseline", baseline_vine, baseline_ent)
        pprint_vine_copula_test("R2 meaned", R2_meaned_vine, ent_R2_mean)
        pprint_vine_copula_test("BIC static", BIC_static_vine, ent_BIC_static)
        pprint_vine_copula_test("BIC dynamic", BIC_dynamic_vine, ent_BIC_dynamic)
