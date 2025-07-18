#!/usr/bin/env python
# coding: utf-8

"""First constraint: RMSE < 0.17 K"""

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__
from tqdm.auto import tqdm

pl.switch_backend("agg")

load_dotenv()
pl.style.use("../../../../../defaults.mplstyle")
ens_run = int(os.getenv("ENSEMBLE_RUN")) #starts at 1
orig_const = os.getenv("CONSTRAINT_SET")
ens_constraint_set = f"{orig_const}r{ens_run}"
cutoff_env = os.getenv("CUTOFF_YEAR")
if cutoff_env:
    cutoff_year = int(cutoff_env)
    constraint_set = f"{ens_constraint_set}cut{cutoff_year}"
else:
    cutoff_year = None
    constraint_set = os.getenv("CONSTRAINT_SET")
print("Doing RMSE constraint...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
#constraint_set = os.getenv("CONSTRAINT_SET")+"cut"+cutoff_env
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")

assert fair_v == __version__


temp_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{orig_const}r1/prior_runs/"
    "temperature_1850-2101.npy", mmap_mode='r'
)

#df_gmst = pd.read_csv("../../../../../data/forcing/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.rebased_1850-1900.csv")
df_gmst = pd.read_csv("../../../../../data/cmip-thorne/ts_ESM1-2-LR_ssp245.csv")
gmst = df_gmst.iloc[:,ens_run]
gmst = gmst - np.mean(gmst[:51]) #rebase to 1850-1900
start_year = 1850
if cutoff_year is not None:
    # index of the last gmst point to use
    last_idx = cutoff_year - start_year
    gmst = gmst[: last_idx + 1]

def rmse(obs, mod):
    return np.sqrt(np.sum((obs - mod) ** 2) / len(obs))


weights = np.ones(52)
weights[0] = 0.5
weights[-1] = 0.5

rmse_temp = np.zeros((samples))

if plots:
    fig, ax = pl.subplots(figsize=(5, 5))
    ax.fill_between(
        np.arange(1850, 2102),
        np.min(temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
        np.max(temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2102),
        np.percentile(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 5, axis=1
        ),
        np.percentile(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 95, axis=1
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2102),
        np.percentile(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 16, axis=1
        ),
        np.percentile(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 84, axis=1
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.plot(
        np.arange(1850, 2102),
        np.median(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1
        ),
        color="#000000",
    )
    ax.plot(np.arange(1850.5, 2024), gmst, color="b")

    ax.set_xlim(1850, 2100)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    pl.title("Prior ensemble")
    pl.tight_layout()
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/", exist_ok=True
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_ssp245.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_ssp245.pdf"
    )
    pl.close()

# temperature is on timebounds, and observations are midyears
# but, this is OK, since we are subtracting a consistent baseline (1850-1900, weighting
# the bounding timebounds as 0.5)
# e.g. 1993.0 timebound has big pinatubo hit, timebound 143
# in obs this is 1992.5, timepoint 142
# compare the timebound after the obs, since the forcing has had chance to affect both
# the obs timepoint and the later timebound.
# the goal of RMSE is as much to match the shape of warming as the magnitude; we do not
# want to average out internal variability in the model or the obs.
#for i in tqdm(range(samples), disable=1 - progress):
#    rmse_temp[i] = rmse(
#        gmst[:174],
#        temp_in[1:175, i] - np.average(temp_in[:52, i], weights=weights, axis=0),    )

for i in tqdm(range(samples), disable=1 - progress):
     # baseline subtraction is always over the first 52 years (1850–1901)
     baseline = np.average(temp_in[:52, i], weights=weights, axis=0)

     if cutoff_year is None:
         # original full‐period constraint
         model_slice = temp_in[1:175, i] - baseline
         n_obs = 174
     else:
         n_obs = last_idx + 1              # e.g. cutoff_year=2000 → last_idx=150, n_obs=151
         model_slice = temp_in[1 : n_obs+1, i] - baseline
         #this matches the shift by 1 that was done by Chris Smith, see above explanation
     rmse_temp[i] = rmse(gmst[0:n_obs], model_slice)


accept_temp = rmse_temp < 0.17
print("Passing RMSE constraint:", np.sum(accept_temp))
valid_temp = np.arange(samples, dtype=int)[accept_temp]

# get 10 largest (but passing) and 10 smallest RMSEs
rmse_temp_accept = rmse_temp[accept_temp]
just_passing = np.argpartition(rmse_temp_accept, -10)[-10:]
smashing_it = np.argpartition(rmse_temp_accept, 10)[:10]
print(just_passing)
print(rmse_temp_accept[just_passing])
print(rmse_temp_accept[smashing_it])


if plots:
    # plot top 10 and "just squeaking in 10"
    fig, ax = pl.subplots(figsize=(5, 5))
    ax.plot(
        np.arange(1850.5, 2102),
        (
            temp_in[:, valid_temp[just_passing]]
            - np.average(
                temp_in[:52, valid_temp[just_passing]], weights=weights, axis=0
            )
        ),
        color="#ff0000",
        label=[r"RMSE $\approx$ 0.17°C"] + [""] * 9,
    )
    ax.plot(
        np.arange(1850.5, 2102),
        (
            temp_in[:, valid_temp[smashing_it]]
            - np.average(temp_in[:52, valid_temp[smashing_it]], weights=weights, axis=0)
        ),
        color="#0000ff",
        label=[r"RMSE $\approx$ 0.10°C"] + [""] * 9,
    )
    ax.axhspan(0.67, 0.99, color="k", alpha=0.15, lw=0)
    ax.axvspan(1995, 2015, color="k", alpha=0.15, lw=0)
    ax.plot(np.arange(1850.5, 2024), gmst, color="k", label="Best estimate historical")
    ax.set_xlim(1850, 2100)
    ax.set_ylim(-1, 4)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.text(1860, 0.83, "IPCC AR6 5--95% range", va="center")
    ax.legend(loc="upper left")
    pl.title("Historical + SSP2-4.5 GMST")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "post_rsme_top10_bottom10_ssp245.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "post_rsme_top10_bottom10_ssp245.pdf"
    )
    pl.close()

    # ensemble wide
    fig, ax = pl.subplots(figsize=(5, 5))
    ax.fill_between(
        np.arange(1850, 2102),
        np.min(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            axis=1,
        ),
        np.max(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850.5, 2102),
        np.percentile(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            5,
            axis=1,
        ),
        np.percentile(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            95,
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850.5, 2102),
        np.percentile(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            16,
            axis=1,
        ),
        np.percentile(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            84,
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.plot(
        np.arange(1850.5, 2102),
        np.median(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            axis=1,
        ),
        color="#000000",
    )

    ax.plot(np.arange(1850.5, 2024), gmst, color="b")

    ax.set_xlim(1850, 2100)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    pl.title("After RMSE constraint")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "post_rsme_ssp245.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "post_rsme_ssp245.pdf"
    )
    pl.close()

valid_temp = np.arange(samples, dtype=int)[accept_temp]
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors",
    exist_ok=True,
)
np.savetxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_pass.csv",
    valid_temp.astype(int),
    fmt="%d",
)
