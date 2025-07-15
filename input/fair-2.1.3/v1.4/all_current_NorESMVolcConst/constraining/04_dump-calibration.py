#!/usr/bin/env python
# coding: utf-8

"""Takes constrained runs and dumps parameters into the output file"""

import os
import pdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import pdb
load_dotenv()

print("Dumping output...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")+"r"+os.getenv("ENSEMBLE_RUN")+"cut"+os.getenv("CUTOFF_YEAR")
orig_const = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))

df_cc = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{orig_const}/priors/"
    "carbon_cycle.csv"
)
df_cr = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{orig_const}/priors/"
    "climate_response_ebm3.csv"
)
df_aci = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{orig_const}/priors/"
    "aerosol_cloud.csv"
)
df_ari = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{orig_const}/priors/"
    "aerosol_radiation.csv"
)
df_ozone = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{orig_const}/priors/ozone.csv"
)
df_scaling = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{orig_const}/priors/"
    "forcing_scaling.csv"
)
df_1750co2 = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{orig_const}/priors/"
    "co2_concentration_1750.csv"
)

valid_all = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_reweighted_pass.csv"
).astype(
    np.int64
)  # [:1000]
valid_all

seed = 1355763 + 399 * valid_all

# concatenate each param dataframe and prefix with its model element to avoid
# namespace conflicts (and a bit of user intuivity)
params_out = pd.concat(
    (
        df_cr.loc[valid_all, :].rename(columns=lambda x: "clim_" + x),
        df_cc.loc[valid_all, :].rename(columns=lambda x: "cc_" + x),
        df_ari.loc[valid_all, :].rename(columns=lambda x: "ari_" + x),
        df_aci.loc[valid_all, :].rename(columns=lambda x: "aci_" + x),
        df_ozone.loc[valid_all, :].rename(columns=lambda x: "o3_" + x),
        df_scaling.loc[valid_all, :].rename(columns=lambda x: "fscale_" + x),
        df_1750co2.loc[valid_all, :].rename(
            columns={"co2_concentration": "cc_co2_concentration_1750"}
        ),
        pd.DataFrame(seed, index=valid_all, columns=["seed"]),
    ),
    axis=1,
)
df_newformat=pd.read_csv("../../../../../data2/calibration/v1.4.1/calibrated_constrained_parameters_1.4.1.csv")

params_out.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "calibrated_constrained_parameters.csv"
)

new_df = pd.DataFrame()

# 1. Rename columns[0:37] → df_newformat.columns[1:38]
new_df[df_newformat.columns[1:38]] = params_out.iloc[:, 0:37].copy()

# 2. Replicate params_out.columns[36] → df_newformat.columns[37:77]
replicated_col = params_out.iloc[:, 36]
for col in df_newformat.columns[37:77]:
    new_df[col] = replicated_col

# 3. Add params_out.columns[37:41] → df_newformat.columns[77:81]
new_df[df_newformat.columns[77:81]] = params_out.iloc[:, 37:41].copy()

solar_trend_values = params_out.iloc[:,41].values.squeeze() #dropped in later versions, retaining here

# 5. Add params_out.columns[42:46] → df_newformat.columns[81:85]
new_df[df_newformat.columns[81:85]] = params_out.iloc[:, 42:46].copy()

# 6. Add df_newformat.columns[85:87] with all True values
new_df[df_newformat.columns[85:87]] = True

new_df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "calibrated_constrained_parameters222.csv"
)


#import logging

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties
import scipy.stats
import xarray as xr

scenarios = ["all", "no_ghgs", "no_aerosols", "no_other", "no_natural", "no_anthro"]
#logger = logging.getLogger('fair')
#logger.setLevel(level=logging.CRITICAL)
output_ensemble_size=841
scenarios = ["all"]
f = FAIR()
f.define_time(1750, 2101, 1)
f.define_scenarios(scenarios)
species, properties = read_properties('../../../../../data2/calibration/v1.4.1/species_configs_properties_1.4.1.csv')
f.define_species(species, properties)
f.ch4_method='Thornhill2021'
#df_configs = pd.read_csv('../../../../../data2/calibration/v1.4.1/calibrated_constrained_parameters_1.4.1.csv', index_col=0)
df_configs = new_df
#we will grab new calibrated constrained parameters based on the current cutoff year
f.define_configs(df_configs.index)
f.allocate()
trend_shape = np.ones((2101-1750+1))
trend_shape[:271] = np.linspace(0, 1, 271)
f.fill_from_csv(
    emissions_file='../../../../../data2/emissions/v1.4.1/historical_1750-2023.csv',
    forcing_file='../../../../../data2/forcing/v1.4.1/volcanic_solar.csv',
)

da_emissions = xr.load_dataarray(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{orig_const}/emissions/"
        "ssps_harmonized_1750-2499.nc")
df_solar = pd.read_csv(
        "../../../../../data/forcing/solar_erf_timebounds.csv", index_col="year"
    )
df_volcanic = pd.read_csv(
        "../../../../../data/forcing/volcanic_ERF_1750-2101_timebounds.csv",
        index_col="timebounds",
    )

volcanic_forcing = df_volcanic["erf"].loc[1750:2101].values

volc_new_df = pd.read_csv("../../../../../data/cmip-thorne/ERF_NorESM_rcp45VolcConst_full.csv")
volcanic_forcing[100:350]= volc_new_df["ERF_natural"].values - df_solar["erf"].loc[1850:2099].values
solar_forcing = df_solar["erf"].loc[1750:2101].values

da = da_emissions.loc[dict(config="unspecified", scenario="ssp245")][:351, ...]
fe = da.expand_dims(dim=["scenario", "config"], axis=(1, 2))
config_values = f.emissions.coords['config']
small_da  = fe.assign_coords(scenario=["all"]).squeeze('config', drop=True).copy()
broadcasted = small_da.expand_dims(config=config_values)

f.emissions.loc[:] = broadcasted
fill(
    f.forcing,
    volcanic_forcing[:,None,None] * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
    specie="Volcanic",
)
fill(
    f.forcing,
    solar_forcing[:,None,None] * df_configs["forcing_scale[Solar]"].values.squeeze() + \
            trend_shape[:, None, None] * solar_trend_values ,
    specie="Solar",
)

f.fill_species_configs("../../../../../data2/calibration/v1.4.1/species_configs_properties_1.4.1.csv")
#f.override_defaults("../../../../../data2/calibration/v1.4.1/calibrated_constrained_parameters_1.4.1.csv")
f.override_defaults(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "calibrated_constrained_parameters222.csv"
)

# initial conditions
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)
f.run() #unclear to me if we need to run here
ghgs = [
 'CO2','CH4','N2O',
 'CFC-11','CFC-12','CFC-113','CFC-114','CFC-115','HCFC-22','HCFC-141b','HCFC-142b',
 'CCl4','CHCl3','CH2Cl2','CH3Cl','CH3CCl3','CH3Br','Halon-1211','Halon-1301','Halon-2402',
 'CF4','C2F6','C3F8','c-C4F8','C4F10','C5F12','C6F14','C7F16','C8F18','NF3','SF6','SO2F2',
 'HFC-125','HFC-134a','HFC-143a','HFC-152a','HFC-227ea','HFC-23','HFC-236fa','HFC-245fa','HFC-32','HFC-365mfc','HFC-4310mee',
]
aerosols = [
 'Aerosol-radiation interactions',
 'Aerosol-cloud interactions',
]
natural = [
 'Solar',
 'Volcanic',
]
other = [
 'Ozone',
 'Light absorbing particles on snow and ice',
 'Stratospheric water vapour',
 'Land use',
]
anthro = list(set(f.species) - set(natural))

scenarios = ["all", "no_ghgs", "no_aerosols", "no_other", "no_natural", "no_anthro"]

ff = FAIR()
ff.define_time(1750, 2101, 1)
ff.define_scenarios(scenarios)
ff.define_configs(df_configs.index)

species = ["bulk"]
properties = {
    "bulk": {
        "type": "unspecified",
        "input_mode": "forcing",
        "greenhouse_gas": False,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    }
}

ff.define_species(species, properties)
ff.allocate()

fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all') - f.forcing.sel(specie=ghgs).sum(dim='specie').sel(scenario='all'),
    specie="bulk",
    scenario="no_ghgs"
)

fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all') - f.forcing.sel(specie=aerosols).sum(dim='specie').sel(scenario='all'),
    specie="bulk",
    scenario="no_aerosols"
)

fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all') - f.forcing.sel(specie=other).sum(dim='specie').sel(scenario='all'),
    specie="bulk",
    scenario="no_other"
)

fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all') - f.forcing.sel(specie=natural).sum(dim='specie').sel(scenario='all'),
    specie="bulk",
    scenario="no_natural"
)

fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all') - f.forcing.sel(specie=anthro).sum(dim='specie').sel(scenario='all'),
    specie="bulk",
    scenario="no_anthro"
)

fill(
    ff.forcing,
    f.forcing_sum.sel(scenario='all'),
    specie="bulk",
    scenario="all"
)

# climate response #here we grab the newly selected parameters
ff.override_defaults(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "calibrated_constrained_parameters222.csv"
)


# initial conditions
initialise(ff.forcing, 0)
initialise(ff.temperature, 0)
initialise(ff.ocean_heat_content_change, 0)
ff.run()

base = np.arange(1850, 1901)
temp_ghgs = ((
        ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')
    ) - (
        ff.temperature.sel(scenario="no_ghgs", layer=0) - ff.temperature.sel(scenario="no_ghgs", layer=0, timebounds=base).mean(dim='timebounds')
    ))
temp_aerosols = ((
        ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')
    ) - (
        ff.temperature.sel(scenario="no_aerosols", layer=0) - ff.temperature.sel(scenario="no_aerosols", layer=0, timebounds=base).mean(dim='timebounds')
    ))
temp_natural = ((
        ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')
    ) - (
        ff.temperature.sel(scenario="no_natural", layer=0) - ff.temperature.sel(scenario="no_natural", layer=0, timebounds=base).mean(dim='timebounds')
    ))
temp_other = ((
        ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')
    ) - (
        ff.temperature.sel(scenario="no_other", layer=0) - ff.temperature.sel(scenario="no_other", layer=0, timebounds=base).mean(dim='timebounds')
    ))
temp_nonat = ff.temperature.sel(scenario="no_natural", layer=0) - ff.temperature.sel(scenario="no_natural", layer=0, timebounds=base).mean(dim='timebounds')
temp_anthro = ((
        ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')
    ) - (
        ff.temperature.sel(scenario="no_anthro", layer=0) - ff.temperature.sel(scenario="no_anthro", layer=0, timebounds=base).mean(dim='timebounds')
    ))
temp_all = ff.temperature.sel(scenario="all", layer=0) - ff.temperature.sel(scenario="all", layer=0, timebounds=base).mean(dim='timebounds')

np.save(f"../../../../../output/{orig_const}/{constraint_set}_temp_anthro", temp_anthro)
np.save(f"../../../../../output/{orig_const}/{constraint_set}_temp_all", temp_all)
np.save(f"../../../../../output/{orig_const}/{constraint_set}_temp_nonat", temp_nonat)
