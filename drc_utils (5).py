'''
Functions to help with measuring and characterising drug response curves (DRCs).
'''
import pandas as pd
import numpy as np
from itertools import product
from scipy.optimize import curve_fit
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------- Load DRC data --------------------------------
def load_drc_data(fname, layout_file, sheet_num, drug_names = ["Drug 1", "Drug 2"], max_concentrations = [1, 1], cell_type_names = ["Type 1", "Type 2"], 
              sheet_prefix = "Sheet", machine = "tecan", idx_data=42, usecols="A:M", negative_control="AB", drop_control = True, normalise_by = "mean", detailed_output=False):
    '''
    Load the fluorescence data from an excel file and return a dataframe with the drug concentrations and normalised fluorescence values.
    fname: The name of the excel file containing the fluorescence data.
    layout_file: The name of the excel file containing the layout information.
    sheet_num: The sheet number in the excel file containing the fluorescence data.
    drug_names: A list of drug names. The drug names are used to identify the drug wells in the layout file.
    max_concentrations: A list of maximum drug concentrations. The maximum drug concentrations are used to convert relative drug concentrations to absolute concentrations.
    cell_type_names: A list of names of the cell types used in the experiment. Will be mapped to Type 1 etc from the layout file.
    sheet_prefix: The prefix of the sheet name in the excel file containing the fluorescence data.
    machine: The machine used to perform the read. This is used to determine the header row of the fluorescence data.
    idx_data: The index of the header row where the data starts in the tecan excel file.
    usecols: The columns to use in the tecan excel file.
    negative_control: The name of the negative control wells in the layout file.
    drop_control: Whether to drop the positive/negative control wells from the results.
    '''
    # Load fluorescence data from the excel file. 
    # Often the excel files have other information on the read (e.g. machine settings). Process the raw sheet to get just the readout values.
    sheet_name = sheet_prefix + str(sheet_num) if sheet_prefix is not None else sheet_num
    data_raw_df = None
    if machine=="biotek":
        # The biotek machine will produce an extra column at the end with the excitation/absorbance values. Drop if present.
        data_raw_df = pd.read_excel(fname, sheet_name=sheet_name, header=0, index_col=0)
        data_raw_df = data_raw_df.drop(data_raw_df.columns[12], axis=1)
    if machine == "tecan":
        data_raw_df = pd.read_excel(fname, sheet_name=sheet_name, header=idx_data, index_col=0, usecols=usecols)
        data_raw_df.columns = data_raw_df.columns.astype(int)
    if machine == "jinling":
        data_raw_df = pd.read_excel(fname, sheet_name=sheet_name, header=None)
        n_rows = data_raw_df.shape[0]
        data_raw_df.index = list("BCDEFGH")[7-n_rows:]
        data_raw_df.columns = data_raw_df.columns.astype(int) + 1
    if machine == "jinling_cs":
        data_raw_df = pd.read_excel(fname, sheet_name=sheet_name, header=None)
        n_rows = data_raw_df.shape[0]
        data_raw_df.index = list("BCDE")
        data_raw_df.columns = data_raw_df.columns.astype(int) + 1

    # Make sure that data was loaded
    if data_raw_df is None:
        raise ValueError("Machine type not recognised. Please choose from 'tecan', 'biotek', 'jinling' or 'jinling_cs'.")
    
    # Load the layout file. This assumes that the first 8 rows contain the layout information and rows 12-20 contain the drug concentrations.
    layout_df = pd.read_excel(layout_file, header=1, nrows=8)
    drug_concentrations_df = pd.read_excel(layout_file, header=12, nrows=8, index_col=0)
    cell_type_df = pd.read_excel(layout_file, header=23, nrows=8, index_col=0)

    # Extract and combine the readout and drug concentration information by first getting all the non-empty wells 
    # from the layout file, and then mapping them to the drug concentration and readout values.
    results_df_long = pd.melt(layout_df, id_vars=["Unnamed: 0"], var_name='Drug', value_name='Concentration')
    results_df_long = results_df_long.rename(columns={"Unnamed: 0": "RowId", "Drug": "ColumnId", "Concentration": "Type"})
    results_df_long['WellId'] = results_df_long['RowId'] + results_df_long['ColumnId'].astype(str)
    results_df_long = results_df_long.dropna(subset=['Type']) # Drop all empty wells

    # For each drug well, find the corresponding drug concentration and readout value and add them to results_df_long.
    for i, well in results_df_long.iterrows():
        concentration = drug_concentrations_df.loc[well['RowId'], well['ColumnId']]
        fluorescence_raw = data_raw_df.loc[well['RowId'], well['ColumnId']]
        if fluorescence_raw == "OVER": fluorescence_raw = np.nan # If the fluorescence is over the limit of detection, set to nan.
        # Get the maximum drug concentration from the layout file to convert relative values to absolute concentrations.
        drugId = well['Type'].split(' ')
        max_concentration = max_concentrations[int(drugId[1]) - 1] if len(drugId) > 1 else np.nan
        # Add the drug concentration and readout values to results_df_long.
        results_df_long.loc[results_df_long['WellId']==well['WellId'], 'DrugConcentration'] = concentration * max_concentration
        results_df_long.loc[results_df_long['WellId']==well['WellId'], 'Fluorescence_raw'] = fluorescence_raw
        # Annotate the cell type
        cell_type_id = str(cell_type_df.loc[well['RowId'], well['ColumnId']]).split(' ')
        cell_type = cell_type_names[int(cell_type_id[1]) - 1] if len(cell_type_id) > 1 else np.nan
        results_df_long.loc[results_df_long['WellId']==well['WellId'], 'CellType'] = cell_type

    # Normalise the fluorescence readings by subtracting the background fluorescence from wells marked as negative control () in the layout file.
    background_fluorescence = results_df_long.loc[results_df_long['Type']==negative_control, 'Fluorescence_raw'].mean()
    if np.isnan(background_fluorescence): 
        background_fluorescence = 0
        # Issue warning that no negative control wells were found.
        warnings.warn("No negative control wells found. Setting background fluorescence to 0.")
    results_df_long['Fluorescence_normalised'] = results_df_long['Fluorescence_raw'] - background_fluorescence
    if detailed_output:
        results_df_long['Background'] = background_fluorescence
    # Normalise the fluorescence readings for each drug by dividing by the fluorescence of wells with DrugConcentration=0.
    n_drugs = len(drug_names)
    for drug_id, cell_type in product(range(1, n_drugs+1), results_df_long['CellType'].unique()):
        mask = (results_df_long['Type']=="Drug %d"%drug_id) & (results_df_long['CellType']==cell_type)
        curr_subset = results_df_long[mask]
        avg_no_drug = 1
        if normalise_by == "mean":
            avg_no_drug = curr_subset.loc[results_df_long['DrugConcentration']==0, 'Fluorescence_normalised'].mean()
        elif normalise_by == "median":
            avg_no_drug = curr_subset.loc[results_df_long['DrugConcentration']==0, 'Fluorescence_normalised'].median()
        results_df_long.loc[mask, 'Fluorescence_normalised'] = results_df_long.loc[mask, 'Fluorescence_normalised'] / avg_no_drug
        if detailed_output:
            results_df_long.loc[mask, 'AverageFluorescence_noDrug'] = avg_no_drug
    results_df_long['Viability'] = results_df_long['Fluorescence_normalised']*100

    # Clean up
    results_df_long['Type'] = results_df_long['Type'].replace(dict(zip(["Drug %d"%x for x in range(1, n_drugs+1)], drug_names)))
    results_df_long['Machine'] = machine
    results_df_long = results_df_long.rename(columns={"Type": "Drug"})
    if drop_control:
        results_df_long = results_df_long.dropna(subset=['DrugConcentration']) # Drop the positive/negative control wells
    return results_df_long

# -------------------------------- 4-param logistic DRC model --------------------------------
def ll4_log(x, e_max, e_min, b, ic50):
    '''This function is a copy of the LL.4 (4 parameter logistic DRC model) function 
    from the R drc package. Adopted from: https://gist.github.com/yannabraham/5f210fed773785d8b638
    It performs the fitting in log x space, which is more stable:
    - x: dose
    - e_max: max response
    - e_min: min response
    - b: hill coefficient
    - ic50: half-maximal inhibitory concentration
    '''
    return e_min + (e_max-e_min) / (1+np.exp(b*(np.log(x)-np.log(ic50))))

# ------------------------------------ Fit Dose Response ------------------------------------
def bootstrap_DRC_fit(data_df, model, x="Concentration", y="Survival",
                  n_boot=1000, significance_level_ci=0.95, 
                  bounds_list=(0, [1, 1, 10, 1]), p0=[100, 25, 1, 1], 
                  plot_bootstraps=False, return_bootstrap=False, show_progress=True, suppress_warnings=False):
    '''This function fits a model to the data and performs bootstrapping to estimate the confidence intervals
    - data_df: a pandas dataframe containing the data
    - model: the model to fit
    - x: the x variable
    - y: the y variable
    - replicate_id: the replicate id
    - bounds_list: the bounds for the parameters
    - n_boot: the number of bootstraps
    - significance_level_ci: the significance level for the confidence intervals
    - return_bootstrap: whether to return the bootstrapped parameters and predictions
    '''
    model_fun = {"ll4_log": ll4_log}[model]

    # Check that the 0 concentration is not included, and if so remove it
    if 0 in data_df[x].values:
        data_df = data_df[data_df[x] > 0]
        if not suppress_warnings:
            warnings.warn("The 0 concentration was removed from the data to allow fitting of the dose {} model".format(model))

    # Check for nans
    if np.any(np.isnan(data_df[y])):
        data_df = data_df.dropna(subset=[y])
        if not suppress_warnings:
            warnings.warn("NaN values were found in the data. Dropping these values.")

    # Compute the fit to the average data
    data_df_avg = data_df.groupby(x).mean(numeric_only=True).reset_index()
    popt_mle, pcov = curve_fit(f=model_fun, xdata=data_df_avg[x], ydata=data_df_avg[y],
                        bounds=bounds_list, p0=p0)
    predicted_y = model_fun(data_df_avg[x], *popt_mle)
    param_df = pd.DataFrame({"Parameter": ["e_max", "e_min", "b", "ic50", "auc"], 
                             "MLE": [*popt_mle, np.trapezoid(y=predicted_y, x=data_df_avg[x])]})
    # Generate the prediction based on the average fit
    prediction_df = pd.DataFrame({x: data_df_avg[x], y: predicted_y})

    # Perform non-parametric bootstrapping
    bootstrap_palette = sns.color_palette("tab10", n_boot)
    if n_boot > 0:
        tmp_param_list = []
        tmp_prediction_list = []
        for i in tqdm(np.arange(n_boot), disable=(show_progress==False)):
            tmp_df = data_df.groupby(x, group_keys=False).sample(n=1, replace=True).reset_index(drop=True)
            try:
                popt_bootstrap, pcov = curve_fit(f=model_fun, xdata=tmp_df[x], ydata=tmp_df[y],
                                    bounds=bounds_list, p0=p0)
                predicted_y = model_fun(tmp_df[x], *popt_bootstrap)
                tmp_prediction_list.append(pd.DataFrame({x: tmp_df[x], y: predicted_y, "BootstrapId": [i] * len(tmp_df[x])}))
                tmp_param_list.append([*popt_bootstrap, np.trapezoid(y=predicted_y, x=tmp_df[x])])
                if plot_bootstraps:
                    plt.plot(tmp_df[x], tmp_df[y], "o", markersize=4, color=bootstrap_palette[i])
                    plt.plot(tmp_prediction_list[-1][x], tmp_prediction_list[-1][y], color=bootstrap_palette[i])
            except:
                if not suppress_warnings:
                    warnings.warn("Bootstrap {} failed".format(i))
                continue
        prediction_df_bootstrap = pd.concat(tmp_prediction_list)
        param_df_bootstrap = pd.DataFrame(tmp_param_list, columns=["e_max", "e_min", "b", "ic50", "auc"])
        param_df_bootstrap = param_df_bootstrap.melt(var_name="Parameter", value_name="Value", value_vars=["e_max", "e_min", "b", "ic50", "auc"])

        # Compute the confidence intervals
        # 1. Predictions
        lower_ci = (1-significance_level_ci)/2
        upper_ci = significance_level_ci + (1-significance_level_ci)/2
        prediction_df_ci = prediction_df_bootstrap.groupby(x).quantile([lower_ci, upper_ci]).reset_index().pivot(index=x, columns="level_1", values=y).reset_index()
        prediction_df = prediction_df.merge(prediction_df_ci, on=x)
        prediction_df.rename(columns={lower_ci: "LowerCI", upper_ci: "UpperCI"}, inplace=True)
        # 2. Parameters
        param_df_bootstrap["Value"] = pd.to_numeric(param_df_bootstrap["Value"], errors="coerce")
        param_df_ci = param_df_bootstrap.groupby("Parameter")["Value"].quantile([lower_ci, upper_ci]).reset_index().pivot(index="Parameter", columns="level_1", values="Value").reset_index()
        param_df_ci.rename(columns={lower_ci: "LowerCI", upper_ci: "UpperCI"}, inplace=True)
        param_df = param_df.merge(param_df_ci, on="Parameter", how="left")
    if return_bootstrap:
            return param_df, prediction_df, param_df_bootstrap, prediction_df_bootstrap
    else:
        return param_df, prediction_df