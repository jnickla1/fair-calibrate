#!/usr/bin/env python
# coding: utf-8

"""Apply posterior weighting"""

# mention in paper: skew-normal distribution
# this is where Zeb earns his corn

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from dotenv import load_dotenv
from fair import __version__
from fair.earth_params import mass_atmosphere, molecular_weight_air
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
from netCDF4 import Dataset
pl.switch_backend("agg")

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
ens_run = int(os.getenv("ENSEMBLE_RUN")) #starts at 1

orig_const = os.getenv("CONSTRAINT_SET")
ens_constraint_set = f"{orig_const}r1"
constraint_set = orig_const+"r"+os.getenv("ENSEMBLE_RUN")+"cut"+os.getenv("CUTOFF_YEAR")

samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
pl.style.use("../../../../../defaults.mplstyle")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
cutoff_env = os.getenv("CUTOFF_YEAR")
if cutoff_env:
    cutoff_int = int(cutoff_env)-1850


def average_every_n(lst, n):
    """Calculates the average of every n elements in a list."""
    return np.array([np.mean(lst[i:i + n]) for i in range(0, len(lst), n)])

def gen_orig_number(new_member_number,sz_ens):
    #fix lexicographic reshuffling
    nums = np.arange(1, sz_ens+1)
    reshuffled = sorted([f"{n}|" for n in nums])
    recovered_order = [int(s.rstrip("|")) for s in reshuffled]
    if new_member_number==-1:
        return recovered_order
    else:
        return recovered_order[new_member_number]



assert fair_v == __version__

print("Doing reweighting...")


NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

valid_temp = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_pass.csv").astype(np.int64)

input_ensemble_size = len(valid_temp)

assert input_ensemble_size > output_ensemble_size

temp_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{ens_constraint_set}/prior_runs/"
    "temperature_1850-2101.npy", mmap_mode='r')
ohc_in_all = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{ens_constraint_set}/prior_runs/"
    "ocean_heat_content_1850-2101.npy", mmap_mode='r')
#also need to change this to save not just the most recent year difference
fari_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{ens_constraint_set}/prior_runs/"
    "forcing_ari_2005-2014_mean.npy", mmap_mode='r')
faci_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{ens_constraint_set}/prior_runs/"
    "forcing_aci_2005-2014_mean.npy", mmap_mode='r')
co2_in_all = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{ens_constraint_set}/prior_runs/"
    "concentration_co2_1850-2101.npy", mmap_mode='r')
#need to change this co2 to save to the most recent year
ecs_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{ens_constraint_set}/prior_runs/ecs.npy"
, mmap_mode='r')
tcr_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{ens_constraint_set}/prior_runs/tcr.npy"
, mmap_mode='r')
#af_in = np.load(
#    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
#    "airborne_fraction_1pctCO2_y70_y140.npy")
faer_in = fari_in + faci_in


def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)

model_ecs_offset = 2.86 - 3.0
ecs_params = scipy.optimize.root(opt, [1, 1, 1], args=(2+model_ecs_offset, 3+model_ecs_offset, 5+model_ecs_offset)).x

samples = {}
samples["ECS"] = scipy.stats.skewnorm.rvs(
    ecs_params[0],
    loc=ecs_params[1],
    scale=ecs_params[2],
    size=10**5,
    random_state=91603,
)
samples["TCR"] = scipy.stats.norm.rvs(
    loc=1.39, scale=0.6 / NINETY_TO_ONESIGMA, size=10**5, random_state=18196
)
# note fair produces, and we here report, total earth energy uptake, not just ocean
# this value from IGCC 2023. Use new uncertainties for ocean, assume same uncertainties
# for land, atmosphere and cryopshere.

df_ohc = pd.read_csv("../../../../../data/concentrations/long4OHCA.csv")
#this is the only annoying one not already converted to csv
#df_ohc_model = pd.read_csv("../../../../../data/cmip-thorne/rcp45Volc_ohca.nc")
ohca_meas=np.zeros(250)
model_run = ens_run-1
long_past_index = ((gen_orig_number(model_run,60) -1) // 20)
ohca_spinup = Dataset("../../../../../data/cmip-thorne/rcp45hist_ohca_mon.nc", 'r').variables['__xarray_dataarray_variable__']
ohca_s = ohca_spinup[:].__array__()
ohca_s_yr = average_every_n(ohca_s[long_past_index,:], 12)
ohca_meas[(1850-1850):(2006-1850)]= ohca_s_yr - ohca_s_yr[0]
ohca_earlier = Dataset("../../../../../data/cmip-thorne/rcp45historicalVolc_ohca.nc", 'r').variables['__xarray_dataarray_variable__']
ohca_e = ohca_earlier[:].__array__()
ohca_meas[(1980-1850):(2006-1850)] = ohca_e[(model_run if model_run< 37 else model_run -1) ,:] + 2*ohca_meas[(1979-1850)] - ohca_meas[(1978-1850)] #missing one model_run ohca, so shift them
ohca_later = Dataset("../../../../../data/cmip-thorne/rcp45Volc_ohca.nc", 'r').variables['__xarray_dataarray_variable__']
ohca_l = ohca_later[:].__array__()
ohca_meas[(2006-1850):(2100-1850)]= ohca_l[model_run,:] + 2*ohca_meas[(2005-1850)]-ohca_meas[(2004-1850)]

ohc_sd = df_ohc["stdev"].values
max_idx = len(ohc_sd) - 1
i1ohc = min(cutoff_int, max_idx)
i2ohc = min(cutoff_int - 50, max_idx) 
samples["OHC"] = scipy.stats.norm.rvs(
    loc=(ohca_meas[cutoff_int]-ohca_meas[cutoff_int-50]),
    scale=np.sqrt(ohc_sd[i1ohc]**2 + ohc_sd[i2ohc]**2), size=10**5, random_state=43178)

df_gmst_model = pd.read_csv("../../../../../data/cmip-thorne/ts_NorESM_rcp45Volc_full.csv")
gmst = df_gmst_model.iloc[:,ens_run]
gmst_rebased = gmst - np.mean(gmst[0:51])
df_gmst = pd.read_csv("../../../../../data/forcing/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv")
preind_var_unc = ((-np.mean( df_gmst["Lower confidence limit (2.5%)"].values[0:51]) +
          np.mean( df_gmst["Upper confidence limit (97.5%)"].values[0:51]))/4)**2
preind_var_int = np.var(gmst[0:51])/51
rvi1=min(cutoff_int+2,max_idx)
recent_var_unc =((-np.mean( df_gmst["Lower confidence limit (2.5%)"].values[(rvi1-20):(rvi1)]) +
          np.mean( df_gmst["Upper confidence limit (97.5%)"].values[(rvi1-20):(rvi1)]))/4)**2 
recent_var_int = np.var(gmst[(cutoff_int-18):(cutoff_int+2)])/20
samples["temperature l20yrs"] = scipy.stats.norm.rvs(
        loc=np.mean(gmst_rebased[(cutoff_int-18):(cutoff_int+2)]), 
        scale=np.sqrt(preind_var_unc+preind_var_int+recent_var_unc+recent_var_int), 
        size=10**5, random_state=19387,
)
#loc matches perfectly
#std is just slightly higher at 0.083 than 0.075 if we add 4 sources of variability then take sqrt


samples["ERFari"] = scipy.stats.norm.rvs(
    loc=-0.3, scale=0.3 / NINETY_TO_ONESIGMA, size=10**5, random_state=70173
)
samples["ERFaci"] = scipy.stats.norm.rvs(
    loc=-1.0, scale=0.7 / NINETY_TO_ONESIGMA, size=10**5, random_state=91123
)
samples["ERFaer"] = scipy.stats.norm.rvs(
    loc=-1.3,
    scale=np.sqrt(0.7**2 + 0.3**2) / NINETY_TO_ONESIGMA,
    size=10**5,
    random_state=3916153,
)
# IGCC paper: 417.1 +/- 0.4
# IGCC dataset: 416.9
# my assessment 417.0 +/- 0.5

df_co2 = pd.read_csv("../../../../../data/concentrations/co2_etheridge_keeling.csv")
df_co2_model = pd.read_csv("../../../../../data/cmip-thorne/ERFanthro_NorESM_rcp45Volc_full.csv")
co2_factor = (df_co2['mean'].values[0] -392) / (df_co2_model['CO2'].values[0]-392)
samples["CO2 concentration"] = scipy.stats.norm.rvs(
    loc= (co2_factor*(df_co2_model['CO2'].values[cutoff_int]-392)+392), scale=4*df_co2['unc'].values[i1ohc],
    size=10**5, random_state=81693)
#multiplying by 4 to expand uncert to match existing FaIR



##PLAN
#change mean temperature to last 20 years since current year
#done
#change CO2 concentration to most recent CO2 measurement
#
#OHC change use last 50 year difference with undertainty - need to reconstruct this
#
#discuss with Dr. Thorne what to do about OHC



##for future runs
#reevaluate OHC based on model run up to that year
#reevaluate mean temperature
#also change CO2 concentration and uncertainty

#turn off ECS
#turn off TCR


#collect relevant measures 
ohc_in = np.mean(ohc_in_all[(cutoff_int-1):(cutoff_int+1),:], axis = 0) \
              - np.mean(ohc_in_all[(cutoff_int-51):(cutoff_int-49),:], axis = 0)
#was 170:172 and 121:123
co2_in = np.mean(co2_in_all[(cutoff_int-1):(cutoff_int+1),:], axis = 0) 
#was 172:174


ar_distributions = {}
for constraint in [
    "ECS",
    "TCR",
    "OHC",
    "temperature l20yrs",
    "ERFari",
    "ERFaci",
    "ERFaer",
    "CO2 concentration",
]:
    ar_distributions[constraint] = {}
    ar_distributions[constraint]["bins"] = np.histogram(
        samples[constraint], bins=100, density=True
    )[1]
    ar_distributions[constraint]["values"] = samples[constraint]

weights_20yr = np.ones(21)
weights_20yr[0] = 0.5
weights_20yr[-1] = 0.5
weights_51yr = np.ones(52)
weights_51yr[0] = 0.5
weights_51yr[-1] = 0.5

co2_1850 = 284.3169988
co2_1920 = co2_1850 * 1.01**70  # NOT 2x (69.66 yr), per definition of TCRE
mass_factor = 12.011 / molecular_weight_air * mass_atmosphere / 1e21
##breakpoint()
accepted = pd.DataFrame(
    {
        "ECS": ecs_in[valid_temp],
        "TCR": tcr_in[valid_temp],
        "OHC": ohc_in[valid_temp] / 1e21,
        "temperature l20yrs": np.average(
            temp_in[(cutoff_int-19):(cutoff_int+2), valid_temp], weights=weights_20yr, axis=0
            ) 
        - np.average(temp_in[:52, valid_temp], weights=weights_51yr, axis=0),
        #154:175 originally

        "ERFari": fari_in[valid_temp],
        "ERFaci": faci_in[valid_temp],
        "ERFaer": faer_in[valid_temp],
        "CO2 concentration": co2_in[valid_temp],
    },
    index=valid_temp,
)


def calculate_sample_weights(distributions, samples, niterations=50):
    weights = np.ones(samples.shape[0])
    gofs = []
    gofs_full = []

    unique_codes = list(distributions.keys())  # [::-1]

    for k in tqdm(
        range(niterations), desc="Iterations", leave=False, disable=1 - progress
    ):
        gofs.append([])
        if k == (niterations - 1):
            weights_second_last_iteration = weights.copy()
            weights_to_average = []

        for j, unique_code in enumerate(unique_codes):
            unique_code_weights, our_values_bin_idx = get_unique_code_weights(
                unique_code, distributions, samples, weights, j, k
            )
            if k == (niterations - 1):
                weights_to_average.append(unique_code_weights[our_values_bin_idx])

            weights *= unique_code_weights[our_values_bin_idx]

            gof = ((unique_code_weights[1:-1] - 1) ** 2).sum()
            gofs[-1].append(gof)

            gofs_full.append([unique_code])
            for unique_code_check in unique_codes:
                unique_code_check_weights, _ = get_unique_code_weights(
                    unique_code_check, distributions, samples, weights, 1, 1
                )
                gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
                gofs_full[-1].append(gof)

    weights_stacked = np.vstack(weights_to_average).mean(axis=0)
    weights_final = weights_stacked * weights_second_last_iteration

    gofs_full.append(["Final iteration"])
    for unique_code_check in unique_codes:
        unique_code_check_weights, _ = get_unique_code_weights(
            unique_code_check, distributions, samples, weights_final, 1, 1
        )
        gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
        gofs_full[-1].append(gof)

    return (
        weights_final,
        pd.DataFrame(np.array(gofs), columns=unique_codes),
        pd.DataFrame(np.array(gofs_full), columns=["Target marginal"] + unique_codes),
    )


def get_unique_code_weights(unique_code, distributions, samples, weights, j, k):
    bin_edges = distributions[unique_code]["bins"]
    our_values = samples[unique_code].copy()

    our_values_bin_counts, bin_edges_np = np.histogram(our_values, bins=bin_edges)
    np.testing.assert_allclose(bin_edges, bin_edges_np)
    assessed_ranges_bin_counts, _ = np.histogram(
        distributions[unique_code]["values"], bins=bin_edges
    )

    our_values_bin_idx = np.digitize(our_values, bins=bin_edges)

    existing_weighted_bin_counts = np.nan * np.zeros(our_values_bin_counts.shape[0])
    for i in range(existing_weighted_bin_counts.shape[0]):
        existing_weighted_bin_counts[i] = weights[(our_values_bin_idx == i + 1)].sum()

    if np.equal(j, 0) and np.equal(k, 0):
        np.testing.assert_equal(
            existing_weighted_bin_counts.sum(), our_values_bin_counts.sum()
        )

    unique_code_weights = np.nan * np.zeros(bin_edges.shape[0] + 1)

    # existing_weighted_bin_counts[0] refers to samples outside the
    # assessed range's lower bound. Accordingly, if `our_values` was
    # digitized into a bin idx of zero, it should get a weight of zero.
    unique_code_weights[0] = 0
    # Similarly, if `our_values` was digitized into a bin idx greater
    # than the number of bins then it was outside the assessed range
    # so get a weight of zero.
    unique_code_weights[-1] = 0

    for i in range(1, our_values_bin_counts.shape[0] + 1):
        # the histogram idx is one less because digitize gives values in the
        # range bin_edges[0] <= x < bin_edges[1] a digitized index of 1
        histogram_idx = i - 1
        if np.equal(assessed_ranges_bin_counts[histogram_idx], 0):
            unique_code_weights[i] = 0
        elif np.equal(existing_weighted_bin_counts[histogram_idx], 0):
            # other variables force this box to be zero so just fill it with
            # one
            unique_code_weights[i] = 1
        else:
            unique_code_weights[i] = (
                assessed_ranges_bin_counts[histogram_idx]
                / existing_weighted_bin_counts[histogram_idx]
            )

    return unique_code_weights, our_values_bin_idx


weights, gofs, gofs_full = calculate_sample_weights(
    ar_distributions, accepted, niterations=30
)

effective_samples = int(np.floor(np.sum(np.minimum(weights, 1))))
print("Number of effective samples:", effective_samples)

#assert effective_samples >= output_ensemble_size

draws = []
if (effective_samples >= output_ensemble_size):
    drawn_samples = accepted.sample(
        n=output_ensemble_size, replace=False, weights=weights, random_state=10099)
elif (effective_samples >=10):
    #could be bootstrapping here - dont really care
    drawn_samples = accepted.sample(
        n=effective_samples, replace=False, weights=weights, random_state=10099)
else:
    raise ValueError("Too few effective samples - need to increase the original number of samples.")
    sys.exit(1)

draws.append((drawn_samples))

target_ecs = scipy.stats.gaussian_kde(samples["ECS"])
prior_ecs = scipy.stats.gaussian_kde(ecs_in)
post1_ecs = scipy.stats.gaussian_kde(ecs_in[valid_temp])
post2_ecs = scipy.stats.gaussian_kde(draws[0]["ECS"])

target_tcr = scipy.stats.gaussian_kde(samples["TCR"])
prior_tcr = scipy.stats.gaussian_kde(tcr_in)
post1_tcr = scipy.stats.gaussian_kde(tcr_in[valid_temp])
post2_tcr = scipy.stats.gaussian_kde(draws[0]["TCR"])

target_temp = scipy.stats.gaussian_kde(samples["temperature l20yrs"])
prior_temp = scipy.stats.gaussian_kde(
    np.average(temp_in[(cutoff_int-19):(cutoff_int+2), :], weights=weights_20yr, axis=0)
    - np.average(temp_in[:52, :], weights=weights_51yr, axis=0)
)
post1_temp = scipy.stats.gaussian_kde(
    np.average(temp_in[(cutoff_int-19):(cutoff_int+2), valid_temp], weights=weights_20yr, axis=0)
    - np.average(temp_in[:52, valid_temp], weights=weights_51yr, axis=0)
)
post2_temp = scipy.stats.gaussian_kde(draws[0]["temperature l20yrs"])

target_ohc = scipy.stats.gaussian_kde(samples["OHC"])
prior_ohc = scipy.stats.gaussian_kde(ohc_in / 1e21)
post1_ohc = scipy.stats.gaussian_kde(ohc_in[valid_temp] / 1e21)
post2_ohc = scipy.stats.gaussian_kde(draws[0]["OHC"])

target_aer = scipy.stats.gaussian_kde(samples["ERFaer"])
prior_aer = scipy.stats.gaussian_kde(faer_in)
post1_aer = scipy.stats.gaussian_kde(faer_in[valid_temp])
post2_aer = scipy.stats.gaussian_kde(draws[0]["ERFaer"])

target_aci = scipy.stats.gaussian_kde(samples["ERFaci"])
prior_aci = scipy.stats.gaussian_kde(faci_in)
post1_aci = scipy.stats.gaussian_kde(faci_in[valid_temp])
post2_aci = scipy.stats.gaussian_kde(draws[0]["ERFaci"])

target_ari = scipy.stats.gaussian_kde(samples["ERFari"])
prior_ari = scipy.stats.gaussian_kde(fari_in)
post1_ari = scipy.stats.gaussian_kde(fari_in[valid_temp])
post2_ari = scipy.stats.gaussian_kde(draws[0]["ERFari"])

target_co2 = scipy.stats.gaussian_kde(samples["CO2 concentration"])
prior_co2 = scipy.stats.gaussian_kde(co2_in)
post1_co2 = scipy.stats.gaussian_kde(co2_in[valid_temp])
post2_co2 = scipy.stats.gaussian_kde(draws[0]["CO2 concentration"])

colors = {"prior": "#207F6E", "post1": "#684C94", "post2": "#EE696B", "target": "black"}

if plots:
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/", exist_ok=True
    )
    fig, ax = pl.subplots(3, 3, figsize=(18 / 2.54, 18 / 2.54))
    start = 0
    stop = 8
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        prior_ecs(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        post1_ecs(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        post2_ecs(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[0, 0].plot(
        np.linspace(start, stop, 1000),
        target_ecs(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[0, 0].set_xlim(start, stop)
    ax[0, 0].set_ylim(0, 0.6)
    ax[0, 0].set_title("ECS")
    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xlabel("°C")
    ax[0, 0].set_ylabel("Probability density")

    start = 0
    stop = 4
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        prior_tcr(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        post1_tcr(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        post2_tcr(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[0, 1].plot(
        np.linspace(start, stop, 1000),
        target_tcr(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[0, 1].set_xlim(start, stop)
    ax[0, 1].set_ylim(0, 1.5)
    ax[0, 1].set_title("TCR")
    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_xlabel("°C")

    start = 0.6
    stop = 1.4
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        target_temp(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        prior_temp(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        post1_temp(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[0, 2].plot(
        np.linspace(start, stop, 1000),
        post2_temp(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[0, 2].set_xlim(start, stop)
    ax[0, 2].set_ylim(0, 6)
    ax[0, 2].set_title("Temperature anomaly")
    ax[0, 2].set_yticklabels([])
    ax[0, 2].set_xlabel("°C, 2003-2022 minus 1850-1900")

    start = -1.0
    stop = 0.4
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        target_ari(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        prior_ari(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        post1_ari(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[1, 0].plot(
        np.linspace(start, stop, 1000),
        post2_ari(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[1, 0].set_xlim(start, stop)
    ax[1, 0].set_ylim(0, 2.5)
    ax[1, 0].set_title("Aerosol ERFari")
    ax[1, 0].set_yticklabels([])
    ax[1, 0].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")
    ax[1, 0].set_ylabel("Probability density")

    start = -2.25
    stop = 0.25
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        target_aci(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        prior_aci(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        post1_aci(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[1, 1].plot(
        np.linspace(start, stop, 1000),
        post2_aci(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[1, 1].set_xlim(start, stop)
    ax[1, 1].set_ylim(0, 1.1)
    ax[1, 1].set_title("Aerosol ERFaci")
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

    start = -3
    stop = 0.4
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        target_aer(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        prior_aer(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        post1_aer(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[1, 2].plot(
        np.linspace(start, stop, 1000),
        post2_aer(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[1, 2].set_xlim(start, stop)
    ax[1, 2].set_ylim(0, 1.1)
    ax[1, 2].set_title("Aerosol ERF")
    ax[1, 2].set_yticklabels([])
    ax[1, 2].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

    start = 413
    stop = 421
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        target_co2(np.linspace(start, stop, 1000)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        prior_co2(np.linspace(start, stop, 1000)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        post1_co2(np.linspace(start, stop, 1000)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[2, 0].plot(
        np.linspace(start, stop, 1000),
        post2_co2(np.linspace(start, stop, 1000)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[2, 0].set_xlim(start, stop)
    ax[2, 0].set_ylim(0, 1.0)
    ax[2, 0].set_ylabel("Probability density")
    ax[2, 0].set_title("CO$_2$ concentration")
    ax[2, 0].set_yticklabels([])
    ax[2, 0].set_xlabel("ppm, 2022")

    start = 100
    stop = 900
    ax[2, 1].plot(
        np.linspace(start, stop),
        target_ohc(np.linspace(start, stop)),
        color=colors["target"],
        label="Target",
        lw=2,
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        prior_ohc(np.linspace(start, stop)),
        color=colors["prior"],
        label="Prior",
        lw=2,
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        post1_ohc(np.linspace(start, stop)),
        color=colors["post1"],
        label="Temperature RMSE",
        lw=2,
    )
    ax[2, 1].plot(
        np.linspace(start, stop),
        post2_ohc(np.linspace(start, stop)),
        color=colors["post2"],
        label="All constraints",
        lw=2,
    )
    ax[2, 1].set_xlim(start, stop)
    ax[2, 1].set_ylim(0, 0.007)
    ax[2, 1].set_title("Ocean heat content change")
    ax[2, 1].set_yticklabels([])
    ax[2, 1].set_xlabel("ZJ, 2020 minus 1971")

    ax[2, 2].axis("off")
    legend_lines = [
        Line2D([0], [0], color=colors["prior"], lw=2),
        Line2D([0], [0], color=colors["post1"], lw=2),
        Line2D([0], [0], color=colors["post2"], lw=2),
        Line2D([0], [0], color=colors["target"], lw=2),
    ]
    legend_labels = ["Prior", "Temperature RMSE", "All constraints", "Target"]
    ax[2, 2].legend(legend_lines, legend_labels, frameon=False, loc="upper left")

    fig.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "constraints.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "constraints.pdf"
    )
    pl.close()


if plots:
    pl.scatter(draws[0]["TCR"], draws[0]["ECS"])
    pl.xlabel("TCR, °C")
    pl.ylabel("ECS, °C")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ecs_tcr_constrained.png"
    )
    pl.close()


if plots:
    pl.scatter(draws[0]["TCR"], draws[0]["ERFaci"] + draws[0]["ERFari"])
    pl.xlabel("TCR, °C")
    pl.ylabel("Aerosol ERF, W m$^{-2}$, 2005-2014 minus 1750")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "tcr_aer_constrained.png"
    )
    pl.close()


# move these to the validation script
print("Constrained, reweighted parameters:")
print("ECS:", np.percentile(draws[0]["ECS"], (5, 50, 95)))
print("TCR:", np.percentile(draws[0]["TCR"], (5, 50, 95)))
print(
    "CO2 concentration current:", np.percentile(draws[0]["CO2 concentration"], (5, 50, 95))
)
print(
    "Temperature l20yrs rel. 1850-1900:",
    np.percentile(draws[0]["temperature l20yrs"], (5, 50, 95)),
)
print(
    "Aerosol ERFari 2005-2014 rel. 1750:",
    np.percentile(draws[0]["ERFari"], (5, 50, 95)),
)
print(
    "Aerosol ERFaci 2005-2014 rel. 1750:",
    np.percentile(draws[0]["ERFaci"], (5, 50, 95)),
)
print(
    "Aerosol ERF 2005-2014 rel. 1750:",
    np.percentile(draws[0]["ERFaci"] + draws[0]["ERFari"], (5, 50, 95)),
)
print("OHC change l50yrs:", np.percentile(draws[0]["OHC"], (5, 50, 95)))

print("*likely range")

if plots:
    df_gmst = pd.read_csv("../../../../../data/forcing/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.rebased_1850-1900.csv")
    gmst = df_gmst["gmst"].values

    fig, ax = pl.subplots(figsize=(5, 5))
    ax.fill_between(
        np.arange(1850, 2102),
        np.min(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            axis=1,
        ),
        np.max(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2102),
        np.percentile(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            5,
            axis=1,
        ),
        np.percentile(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            95,
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2102),
        np.percentile(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            16,
            axis=1,
        ),
        np.percentile(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            84,
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.plot(
        np.arange(1850, 2102),
        np.median(
            temp_in[:, draws[0].index]
            - np.average(temp_in[:52, draws[0].index], weights=weights_51yr, axis=0),
            axis=1,
        ),
        color="#000000",
    )

    ax.plot(np.arange(1850.5, 2024), gmst, color="b", label="Observations")

    ax.legend(frameon=False, loc="upper left")

    ax.set_xlim(1850, 2100)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    pl.title("Constrained, reweighted posterior")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "final_reweighted_ssp245.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "final_reweighted_ssp245.pdf"
    )
    pl.close()

np.savetxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_reweighted_pass.csv",
    sorted(draws[0].index),
    fmt="%d",
)
