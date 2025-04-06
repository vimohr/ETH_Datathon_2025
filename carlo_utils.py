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
):

    file_paths = [
        f"data/datasets2025/historical_metering_data_{country}.csv",
        f"data/datasets2025/holiday_{country}.xlsx",
        f"data/datasets2025/rollout_data_{country}.csv",
        "data/datasets2025/spv_ec00_forecasts_es_it.xlsx",
    ]  # Add your file paths here

    dataframes = []

    # Read the files and store them in a list

    for path in file_paths:
        if path.endswith(".csv"):
            df_temp = pd.read_csv(path)
        elif path.endswith(".xlsx"):
            df_temp = pd.read_excel(path)
        else:
            continue
        if "Unnamed: 0" in df_temp.columns:
            df_temp = df_temp.rename(columns={"Unnamed: 0": "DATETIME"})
        dataframes.append(df_temp)

    index = dataframes[0].set_index("DATETIME").index
    index = pd.to_datetime(index)

    columns = dataframes[0].columns
    columns = columns.drop(
        f"VALUEMWHMETERINGDATA_customer{country}_{customer_index[0]}"
    )

    # This is just so I can use the data later

    rollout_data_index = dataframes[2].set_index("DATETIME").index
    rollout_data_index = pd.to_datetime(rollout_data_index)
    index = index.intersection(rollout_data_index)

    # set the index of each dataframe to the same index
    # and convert the index to datetime
    # SO i can combine

    for dataframe in dataframes:
        if "DATETIME" in dataframe.columns:
            dataframe = dataframe.set_index("DATETIME")
            dataframe.index = pd.to_datetime(dataframe.index)
            index = index.intersection(dataframe.index)

    main_dataframe = pd.DataFrame(index=index)

    demand_series = dataframes[0][
        f"VALUEMWHMETERINGDATA_customer{country}_{customer_index[0]}"
    ].to_frame()

    # Demand Stuff with lags / forecasts

    demand_data = (
        dataframes[0][f"VALUEMWHMETERINGDATA_customer{country}_{customer_index[0]}"]
        .to_frame()
        .set_index(index)
        .copy()
    )

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
        pd.to_datetime(holiday_data["holiday_ES"])
    ).astype(int)
    holiday_dataframe["weekend"] = holiday_dataframe.index.weekday >= 5
    holiday_dataframe["holiday"] = holiday_dataframe["holiday"].fillna(0).astype(int)
    holiday_dataframe["weekend"] = holiday_dataframe["weekend"].astype(int)
    holiday_dataframe.fillna(0, inplace=True)

    # Rollout Data

    rollout_data = pd.DataFrame(dataframes[2]).set_index("DATETIME")
    rollout_data.index = pd.to_datetime(rollout_data.index)
    rollout_data = rollout_data.loc[index]
    rollout_data = rollout_data[
        f"INITIALROLLOUTVALUE_customer{country}_{customer_index[0]}"
    ].to_frame()

    for rollout_shift in rollout_values:
        if rollout_shift != 0:
            rollout_data[f"rollout_shift_{rollout_shift}"] = rollout_data[
                f"INITIALROLLOUTVALUE_customer{country}_{customer_index[0]}"
            ].shift(rollout_shift)

    # Temperature and SPV Data

    spv_data = dataframes[3].set_index("DATETIME")
    intersection_index = index.intersection(spv_data.index)
    spv_data = spv_data.loc[intersection_index]

    temp_data = spv_data["temp"].to_frame()
    temp_data.index = pd.to_datetime(temp_data.index)
    temp_data = temp_data.reindex(index)

    for temp_shift in temp:
        if temp_shift > 0:
            temp_data[f"temp_lag_{temp_shift}"] = temp_data["temp"].shift(temp_shift)

    spv_data = spv_data["spv"].to_frame()
    spv_data.index = pd.to_datetime(spv_data.index)
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

    # ADdding the other customers as features

    columns = columns.drop("DATETIME")
    other_data = dataframes[0].set_index("DATETIME")
    other_data.index = pd.to_datetime(other_data.index)
    other_data = other_data.loc[index]
    other_data = other_data[columns]

    for other_column in columns:
        main_dataframe[other_column] = other_data[other_column]
        main_dataframe[other_column].fillna(0, inplace=True)
        main_dataframe[other_column] = main_dataframe[other_column].astype(float)

    # Fourier Data

    fourier_data = fourier_reconstruction_extended(demand_series)
    fourier_data = fourier_data.loc[index]
    # List all DataFrames that need to have the same index
    dfs = [
        main_dataframe,
        demand_data,
        holiday_dataframe,
        temp_data,
        spv_data,
        rollout_data,
        ma_data,
        fourier_data,
    ]

    # Use the main_dataframe index as the reference
    base_index = main_dataframe.index

    # Assert that every DataFrame has the same index as main_dataframe
    for i, df in enumerate(dfs):
        assert df.index.equals(
            base_index
        ), f"DataFrame at position {i} does not have the same index as main_dataframe"

    main_dataframe = pd.concat(
        [
            main_dataframe,
            demand_data,
            holiday_dataframe,
            temp_data,
            spv_data,
            rollout_data,
            fourier_data,
        ],
        axis=1,
    )
    main_dataframe = main_dataframe.fillna(0)

    return main_dataframe


def fourier_reconstruction_extended(df, n_top=100, index=None):
    s = df.iloc[:, 0]
    T = len(s)
    fft_values = np.fft.fft(s.values)
    magnitudes = np.abs(fft_values)
    top_indices = np.argsort(magnitudes)[-n_top:]

    # Use provided index's first element if given; otherwise, default to the desired start.
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

    lags = [24, 24 * 7, 24 * 31, 24 * 90, 24 * 265] + [-i for i in range(1, 31)]
    lagged_dfs = [reconstruction_df]
    for lag in lags:
        lagged = reconstruction_df.shift(lag)
        lagged.columns = [f"{col}_lag_{lag}" for col in lagged.columns]
        lagged_dfs.append(lagged)
    final_df = pd.concat(lagged_dfs, axis=1)
    final_df = final_df.reindex(index)

    return final_df
