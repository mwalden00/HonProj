import numpy as np
import torch
import torch.multiprocessing as mp
import numpy as np
import pickle as pkl
import copy
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
    light = args.light

    with torch.device(device):

        with open("../data/pupil_vine_data_0.pkl", "rb") as f:
            pupil_data = pkl.load(f)

        n_estimators = args.n_estimators

        print(f"Getting {n_estimators} copulaGP estimators...")

        perm = torch.arange(10000).cpu()
        if args.shuffle == 1:
            perm = torch.randperm(10000).cpu()
        bagged_start_layer = args.bagged_start_layer
        X = [
            x.cpu().numpy()
            for x in torch.chunk(torch.Tensor(pupil_data["X"][perm]), n_estimators)
        ]

        Y = [
            y.cpu().numpy()
            for y in torch.chunk(torch.Tensor(pupil_data["Y"][perm]), n_estimators)
        ]

        for i in range(args.bagged_start, n_estimators):
            try:
                os.mkdir(f"../models/layers/pupil_vine/segments/seg_{i}/")
                os.mkdir(f"../models/results/pupil_segments/")
                os.mkdir(f"../data/segmented_pupil_copulas/")
            except:
                pass

            # print(f"\nSelecting Trial {i} with trajectory choices {choices}")
            # np.savetxt(f"./segmented_pupil/choices/choice_i.txt", choices)

            Y_i = Y[i]
            with open(
                f"../data/segmented_pupil_copulas/pupil_section_data_{i}_0.pkl", "wb"
            ) as f:
                pkl.dump(
                    dict(
                        [
                            ("X", X[i]),
                            ("Y", Y_i),
                        ]
                    ),
                    f,
                )

            print(f"Training {i}-th Copula-GP Vine")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_vine(
                path_data=lambda x: f"../data/segmented_pupil_copulas/pupil_section_data_{i}_{x}.pkl",
                path_models=lambda x: f"../models/layers/pupil_vine/segments/seg_{i}/layer_{x}.pkl",
                path_final=f"../models/results/pupil_segments/pupil_model_{i}_res.pkl",
                path_logs=lambda a, b: f"./segmented_pupil/{a}/layer_{b}",
                exp=f"Vine trained on bag {i} param. in pupil area",
                light=light == 1,
                device_list=device_list,
                start=bagged_start_layer,
            )
            bagged_start_layer = 0
        print("\n\nGetting Bagged Vine...")
        vines2bag = []

        for i in range(n_estimators):
            with open(
                f"../models/results/pupil_segments/pupil_model_{i}_res.pkl",
                "rb",
            ) as f:
                vines2bag.append(pkl.load(f)["models"])

        BIC_dynamic_vine = bagged_vine(
            vines_data=vines2bag,
            X=torch.Tensor(pupil_data["X"][:1500]).to(device),
            Y=torch.Tensor(pupil_data["Y"][:1500]).to(device),
            device=device,
            how="BIC dynamic",
        )

        with open("../models/vine2go.pkl", "wb") as f:
            pkl.dump(BIC_dynamic_vine, f)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        BIC_dynamic_entropy = BIC_dynamic_vine.entropy()

        BIC_dynamic_entropy.cpu().numpy().tofile(
            f"./bagged_pupil_entropy_{n_estimators}_estim_continuous.csv", sep=","
        )

        print("Mean cop entropy:", BIC_dynamic_entropy.cpu().numpy().mean())
