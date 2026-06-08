import pandas as pd
import numpy as np

def get_data(
    country="ES",
    customer_index=[114],
    demand=[0],
    temp=[0],
    spv=[0],
    forecast_shifts=[0],
    rollout_values=[0],
    moving_average=[0],
    index_cut=None  # If provided (e.g., 10000), the index will be sliced to index[index_cut:]
):

    file_paths = [
        f"data/datasets2025/historical_metering_data_{country}.csv",
        f"data/datasets2025/holiday_{country}.xlsx",
        f"data/datasets2025/rollout_data_{country}.csv",
        "data/datasets2025/spv_ec00_forecasts_es_it.xlsx",
    ]

    dataframes = []

    # Read the files and store them in a list
    for path in file_paths:
        if path.endswith(".csv"):
            df_temp = pd.read_csv(path)
        elif path.endswith(".xlsx"):
            df_temp = pd.read_excel(path)
        else:
            continue
        # Rename the "Unnamed: 0" column to "DATETIME" if present
        if "Unnamed: 0" in df_temp.columns:
            df_temp = df_temp.rename(columns={"Unnamed: 0": "DATETIME"})
        dataframes.append(df_temp)

    # Get initial index from the first dataframe and slice if requested
    index = dataframes[0].set_index("DATETIME").index
    index = pd.to_datetime(index)
    if index_cut is not None:
        index = index[index_cut:]

    # Build the list of columns from the original dataframe
    # Remove the customer column and also remove 'DATETIME' if present
    columns = dataframes[0].columns.drop(f"VALUEMWHMETERINGDATA_customer{country}_{customer_index[0]}")
    if "DATETIME" in columns:
        columns = columns.drop("DATETIME")

    # Get the index from rollout_data and intersect with our index
    rollout_data_index = pd.to_datetime(dataframes[2].set_index("DATETIME").index)
    index = index.intersection(rollout_data_index)

    # Set the index of each dataframe to the same (trimmed) index and convert to datetime
    for i, dataframe in enumerate(dataframes):
        if "DATETIME" in dataframe.columns:
            dataframe = dataframe.set_index("DATETIME")
            dataframe.index = pd.to_datetime(dataframe.index)
        else:
            try:
                dataframe.index = pd.to_datetime(dataframe.index)
            except Exception as e:
                raise KeyError(f"'DATETIME' column not found and index conversion failed in file {i}. Error: {e}")
        index = index.intersection(dataframe.index)
        dataframes[i] = dataframe  # update the list with the modified dataframe

    main_dataframe = pd.DataFrame(index=index)

    # Demand Stuff with lags / forecasts
    demand_data = dataframes[0][f"VALUEMWHMETERINGDATA_customer{country}_{customer_index[0]}"].to_frame().copy()
    demand_data = demand_data.reindex(index)

    for demand_shift in demand:
        if demand_shift > 0:
            demand_data[f"demand_lag_{demand_shift}"] = demand_data[
                f"VALUEMWHMETERINGDATA_customer{country}_{customer_index[0]}"
            ].shift(demand_shift)

    for forecast_shift in forecast_shifts:
        if forecast_shift > 0:
            demand_data[f"demand_forecast_{forecast_shift}"] = demand_data[
                f"VALUEMWHMETERINGDATA_customer{country}_{customer_index[0]}"
            ].shift(-forecast_shift)

    # Holiday Data
    holiday_data = dataframes[1]
    holiday_dataframe = pd.DataFrame(index=index, columns=["holiday"])
    holiday_dataframe["holiday"] = holiday_dataframe.index.isin(
        pd.to_datetime(holiday_data[f"holiday_{country}"].values)
    ).astype(int)
    holiday_dataframe["weekend"] = (holiday_dataframe.index.weekday >= 5).astype(int)
    holiday_dataframe["holiday"] = holiday_dataframe["holiday"].fillna(0).astype(int)
    holiday_dataframe.fillna(0, inplace=True)

    # Rollout Data
    rollout_data = dataframes[2].copy()
    rollout_data.index = pd.to_datetime(rollout_data.index)
    rollout_data = rollout_data.loc[index]
    rollout_data = rollout_data[f"INITIALROLLOUTVALUE_customer{country}_{customer_index[0]}"].to_frame()

    for rollout_shift in rollout_values:
        if rollout_shift != 0:
            rollout_data[f"rollout_shift_{rollout_shift}"] = rollout_data[
                f"INITIALROLLOUTVALUE_customer{country}_{customer_index[0]}"
            ].shift(rollout_shift)

    # Temperature and SPV Data
    spv_data = dataframes[3]
    # Check if 'DATETIME' exists in spv_data columns before setting the index
    if "DATETIME" in spv_data.columns:
        spv_data = spv_data.set_index("DATETIME")
    else:
        spv_data.index = pd.to_datetime(spv_data.index)
    intersection_index = index.intersection(spv_data.index)
    spv_data = spv_data.loc[intersection_index]

    temp_data = spv_data["temp"].to_frame()
    temp_data = temp_data.reindex(index)
    for temp_shift in temp:
        if temp_shift > 0:
            temp_data[f"temp_lag_{temp_shift}"] = temp_data["temp"].shift(temp_shift)

    spv_data = spv_data["spv"].to_frame()
    spv_data = spv_data.reindex(index)
    for spv_shift in spv:
        if spv_shift > 0:
            spv_data[f"spv_lag_{spv_shift}"] = spv_data["spv"].shift(spv_shift)

    # Moving Averages
    ma_data = pd.DataFrame(index=index)
    for ma_window in moving_average:
        ma_data[f"moving_average_{ma_window}"] = (
            demand_data[f"VALUEMWHMETERINGDATA_customer{country}_{customer_index[0]}"]
            .rolling(ma_window)
            .mean()
        )

    # Adding the other customers as features
    other_data = dataframes[0].reindex(index)
    other_data = other_data[columns]
    main_dataframe['Mean'] = other_data.mean(axis=1)
    main_dataframe['Std'] = other_data.std(axis=1)
    main_dataframe['Median'] = other_data.median(axis=1)

    # Fourier Data (if needed, uncomment these lines)
    # fourier_data = fourier_reconstruction_extended(demand_data)
    # fourier_data = fourier_data.loc[index]

    # Assert that every DataFrame has the same index as main_dataframe
    dfs = [
        main_dataframe,
        demand_data,
        holiday_dataframe,
        temp_data,
        spv_data,
        rollout_data,
        ma_data,
        # fourier_data,
    ]
    base_index = main_dataframe.index
    for i, df in enumerate(dfs):
        assert df.index.equals(base_index), f"DataFrame at position {i} does not have the same index as main_dataframe"

    main_dataframe = pd.concat(
        [
            main_dataframe,
            demand_data,
            holiday_dataframe,
            temp_data,
            spv_data,
            rollout_data,
            # fourier_data,
        ],
        axis=1,
    )
    main_dataframe = main_dataframe.fillna(0)

    return main_dataframe


def fourier_reconstruction_extended(df, n_top=5, index=None):
    s = df.iloc[:, 0]
    T = len(s)
    fft_values = np.fft.fft(s.values)
    magnitudes = np.abs(fft_values)
    top_indices = np.argsort(magnitudes)[-n_top:]

    if index is not None:
        start = pd.to_datetime(index[0])
    else:
        start = pd.Timestamp("2022-01-01 00:00:00")

    new_index = pd.date_range(start=start, end="2024-08-31 23:00:00", freq="H")
    extended_length = len(new_index)
    t = np.arange(extended_length)

    reconstruction_signals = np.zeros((n_top, extended_length))
    for i, idx in enumerate(top_indices):
        reconstruction_signals[i, :] = (
            fft_values[idx] / T * np.exp(2j * np.pi * idx * t / T)
        ).real

    reconstruction_df = pd.DataFrame(
        reconstruction_signals,
        index=[f"freq_{i}" for i in range(n_top)],
        columns=new_index,
    ).T

    lags = [24, 24 * 7, 24 * 31, 24 * 90, 24 * 265] + [-i for i in range(1, 23, 2)]
    lagged_dfs = [reconstruction_df]
    for lag in lags:
        lagged = reconstruction_df.shift(lag)
        lagged.columns = [f"{col}_lag_{lag}" for col in lagged.columns]
        lagged_dfs.append(lagged)
    final_df = pd.concat(lagged_dfs, axis=1)
    final_df = final_df.reindex(index)

    return final_df
