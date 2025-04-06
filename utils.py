import pandas as pd


def get_data(country, indices=[], demand=[0], temp=[0], spv=[0], rollout_values=[0], n_futures=4):

    file_paths = [
        f"data/datasets2025/historical_metering_data_{country}.csv",
        f"data/datasets2025/holiday_{country}.xlsx",
        f"data/datasets2025/rollout_data_{country}.csv",
        "data/datasets2025/spv_ec00_forecasts_es_it.xlsx",
    ]  # Add your file paths here
    dataframes = []

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

    if not indices:
        indices = [col.split("_")[-1] for col in dataframes[0].columns[1:]]

    elif isinstance(indices, int):
        indices = [indices]

    num_customers = len(indices)

    historical_data = dataframes[0]

    historical_data["DATETIME"] = pd.to_datetime(historical_data["DATETIME"])
    historical_data = historical_data.set_index("DATETIME")

    full_time_index = pd.date_range(
        start=historical_data.index.min(), end=historical_data.index.max(), freq="h"
    )

    # Load holiday data
    holiday_data = dataframes[1]

    new_dataframe = pd.DataFrame({"DATETIME": full_time_index})
    # Add a column to indicate holiday status
    new_dataframe["holiday"] = 0  # Default all values to 0

    # Loop through each time and holiday
    for i, time in enumerate(full_time_index):
        # Check if the date of the current time matches any holiday (ignore time part)
        if time in holiday_data[f"holiday_{country}"].dt.date.values or time.weekday() >= 5:
            new_dataframe.at[i, "holiday"] = 1  # Set the holiday flag to 1

    # print(new_dataframe)

    dataframes[1] = new_dataframe

    df = dataframes[2]

    first_df_datetimes = dataframes[0]["DATETIME"]

    for i, df in enumerate(dataframes):
        dataframes[i]["DATETIME"] = pd.to_datetime(dataframes[i]["DATETIME"])

    # print(dataframes[2])

    dataframes[2] = dataframes[2].fillna(0)

    # Loop through all DataFrames in the list and filter them
    for i in range(1, len(dataframes)):
        # Filter each DataFrame by the 'DATETIME' values in the first DataFrame
        dataframes[i] = dataframes[i][
            dataframes[i]["DATETIME"].isin(first_df_datetimes)
        ].reset_index(drop=True)

    # print(dataframes[2])

    # Concatenate all data into a single DataFrame
    df = pd.concat(dataframes, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    # Start feature engineering

    all_dfs = []

    all_ys = []

    all_times = []

    lags = {"demand": demand, "temp": temp, "spv": spv, "rollout": rollout_values}

    num_customers = 1

    for i in indices:
        target_col = f"VALUEMWHMETERINGDATA_customer{country}_{i}"
        init_col = f"INITIALROLLOUTVALUE_customer{country}_{i}"

        selected_columns = ["DATETIME", target_col, init_col, "holiday", "spv", "temp"]

        new_df = df[selected_columns].copy()

        # Generate lag features
        for feature, shift_values in lags.items():
            base_col = target_col if feature == "demand" else feature
            base_col = init_col if feature == "rollout" else base_col
            for lag in shift_values:
                if lag > 0:
                    col_name = f"{feature}_lag{lag}"
                    new_df[col_name] = new_df[base_col].shift(lag)
                elif lag < 0:
                    # print("hello")

                    col_name = f"{feature}_lead{abs(lag)}"
                    new_df[col_name] = new_df[base_col].shift(lag)

        new_df['hour'] = new_df['DATETIME'].dt.hour
        new_df['day'] = new_df['DATETIME'].dt.day
        new_df['month'] = new_df['DATETIME'].dt.month

        # Final cleanup
        for i in range(1, n_futures):
            new_df[target_col + f"_future{i}"] = new_df[target_col].shift(-i)
        new_df = new_df.dropna()

        times = new_df["DATETIME"]
        new_df = new_df.drop(columns=["DATETIME"]).reset_index(drop=True)

        y = pd.DataFrame()
        for i in range(n_futures):
            if i == 0:
                y["future0"] = new_df[target_col]
            else:
                y[f"future{i}"] = new_df[target_col + f"_future{i}"]
        print(new_df.shape)
        print(y.shape)
        

        X = new_df.drop(columns=[target_col]+[target_col + f"_future{i}" for i in range(1, n_futures)]).reset_index(drop=True)

        all_dfs.append(X.to_numpy())
        all_ys.append(y.to_numpy())
        all_times.append(times)

    return (
        all_dfs,
        all_ys,
        all_times,
    )
