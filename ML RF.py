import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)
from scipy.stats import spearmanr
from sklearn.ensemble import (RandomForestClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.colors as mcolors
import os
import glob
from sklearn.metrics import (matthews_corrcoef, average_precision_score)


def build_egfr_delta(
    file_path,
    col_mrn="Medical Record No.",
    col_date="Result Date",
    col_result="Result",
    output_file=None
):

    df = pd.read_excel(file_path)

    df["result_clean"] = df[col_result].astype(str).str.extract(r'([-+]?\d*\.?\d+)')
    df["result_clean"] = pd.to_numeric(df["result_clean"], errors="coerce")

    df[col_date] = pd.to_datetime(
        df[col_date],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(subset=[col_mrn, col_date, "result_clean"])

    df = df.sort_values(by=[col_mrn, col_date])

    first = df.groupby(col_mrn).first().reset_index()
    last = df.groupby(col_mrn).last().reset_index()

    result = pd.DataFrame({
        col_mrn: first[col_mrn],
        "result_awal": first["result_clean"],
        "tanggal_awal": first[col_date],
        "result_akhir": last["result_clean"],
        "tanggal_akhir": last[col_date],
    })

    # DELTA
    result["delta"] = result["result_akhir"] - result["result_awal"]

    # DAYS
    result["days"] = (
        result["tanggal_akhir"] - result["tanggal_awal"]
    ).dt.days

    if output_file:
        result.to_excel(output_file, index=False)

    return result

def build_master_dataset(
    file_pasien="RMDiuretik.xlsx",
    file_diuretik="diuretikfinal3bulan.xlsx",
    file_usia="merge_with_usia.xlsx",
    file_nefro="NefrotoksikMaksimalisasi.xlsx",
    file_egfr="egfr_delta.xlsx",
    output_file=None
):

    df_base = pd.read_excel(file_pasien)

    df_base["MRN"] = df_base["Medical Record No."].astype(str)

    patient_list = df_base["MRN"].unique()

    master = pd.DataFrame({
        "MRN": patient_list
    })

    df_diur = pd.read_excel(file_diuretik)

    df_diur = df_diur.rename(columns={
        "MR No. / Vendor Code": "MRN"
    })

    df_diur["MRN"] = df_diur["MRN"].astype(str)

    df_diur = df_diur[
        df_diur["MRN"].isin(patient_list)
    ]

    # ONE HOT
    subgol_dummies = pd.get_dummies(
        df_diur["Sub_Golongan"],
        prefix="diur"
    )

    df_diur = pd.concat([
        df_diur[["MRN"]],
        subgol_dummies
    ], axis=1)

    df_diur = df_diur.groupby("MRN").max().reset_index()

    df_kelamin = pd.read_excel(file_diuretik)

    df_kelamin = df_kelamin.rename(columns={
        "MR No. / Vendor Code": "MRN"
    })

    df_kelamin["MRN"] = df_kelamin["MRN"].astype(str)

    df_kelamin = df_kelamin[
        df_kelamin["MRN"].isin(patient_list)
    ]

    df_kelamin = df_kelamin.groupby("MRN")["Kelamin"].agg(
        lambda x: x.dropna().iloc[0]
        if len(x.dropna()) > 0
        else "N/A"
    ).reset_index()

    # ENCODE
    df_kelamin["kelamin_P"] = (
        df_kelamin["Kelamin"] == "Perempuan"
    ).astype(int)

    df_kelamin["kelamin_L"] = (
        df_kelamin["Kelamin"] == "Pria"
    ).astype(int)

    df_kelamin["kelamin_NA"] = (
        df_kelamin["Kelamin"] == "N/A"
    ).astype(int)

    df_kelamin = df_kelamin.drop(columns=["Kelamin"])

    df_usia = pd.read_excel(file_usia)

    df_usia["MRN"] = df_usia["MRN"].astype(str)

    df_usia = df_usia[
        df_usia["MRN"].isin(patient_list)
    ]

    drop_cols = [
        "billing_awal",
        "billing_akhir"
    ]

    df_usia = df_usia.drop(
        columns=[c for c in drop_cols if c in df_usia.columns]
    )

    df_usia = df_usia.drop_duplicates(subset="MRN")

    df_nefro = pd.read_excel(file_nefro)

    df_nefro = df_nefro.rename(columns={
        "MR No. / Vendor Code": "MRN"
    })

    df_nefro["MRN"] = df_nefro["MRN"].astype(str)

    df_nefro = df_nefro[
        df_nefro["MRN"].isin(patient_list)
    ]
    
    # tiap ganti file utk variasi, ini harus disesuaikan dgn kolom
    df_nefro = df_nefro[
        ["MRN", "nefrotoksik_maksimalisasi"] 
    ].drop_duplicates(subset="MRN")

    df_egfr = pd.read_excel(file_egfr)

    df_egfr["MRN"] = df_egfr[
        "Medical Record No."
    ].astype(str)

    df_egfr = df_egfr[
        df_egfr["MRN"].isin(patient_list)
    ]

    df_egfr = df_egfr[
        ["MRN", "delta", "days"]
    ].drop_duplicates(subset="MRN")

    master = master.merge(df_diur, on="MRN", how="left")

    master = master.merge(df_kelamin, on="MRN", how="left")

    master = master.merge(df_usia, on="MRN", how="left")

    master = master.merge(df_nefro, on="MRN", how="left")

    master = master.merge(df_egfr, on="MRN", how="left")

    onehot_cols = [
        c for c in master.columns
        if c.startswith("diur_")
    ]

    master[onehot_cols] = master[onehot_cols].fillna(0)

    if output_file:
        master.to_excel(output_file, index=False)

    return master

def build_master_dataset_cci_com(
    file_pasien="RMDiuretik.xlsx",
    file_diuretik="diuretikfinal3bulan.xlsx",
    file_usia="merge_with_usia.xlsx",
    file_nefro="nefrotoksikakumulasi.xlsx",
    file_egfr="egfr_delta.xlsx",
    file_cci="cci.xlsx",
    file_diag="diagnosis_feature.xlsx",
    output_file=None
):
    df_base = pd.read_excel(file_pasien)

    df_base["MRN"] = df_base["Medical Record No."].astype(str)

    patient_list = df_base["MRN"].unique()

    master = pd.DataFrame({
        "MRN": patient_list
    })


    df_diur = pd.read_excel(file_diuretik)

    df_diur = df_diur.rename(columns={
        "MR No. / Vendor Code": "MRN"
    })

    df_diur["MRN"] = df_diur["MRN"].astype(str)

    df_diur = df_diur[
        df_diur["MRN"].isin(patient_list)
    ]

    # ONE HOT
    subgol_dummies = pd.get_dummies(
        df_diur["Sub_Golongan"],
        prefix="diur"
    )

    df_diur = pd.concat([
        df_diur[["MRN"]],
        subgol_dummies
    ], axis=1)

    df_diur = df_diur.groupby("MRN").max().reset_index()


    df_kelamin = pd.read_excel(file_diuretik)

    df_kelamin = df_kelamin.rename(columns={
        "MR No. / Vendor Code": "MRN"
    })

    df_kelamin["MRN"] = df_kelamin["MRN"].astype(str)

    df_kelamin = df_kelamin[
        df_kelamin["MRN"].isin(patient_list)
    ]

    df_kelamin = df_kelamin.groupby("MRN")["Kelamin"].agg(
        lambda x: x.dropna().iloc[0]
        if len(x.dropna()) > 0
        else "N/A"
    ).reset_index()

    df_kelamin["kelamin_P"] = (
        df_kelamin["Kelamin"] == "Perempuan"
    ).astype(int)

    df_kelamin["kelamin_L"] = (
        df_kelamin["Kelamin"] == "Pria"
    ).astype(int)

    df_kelamin["kelamin_NA"] = (
        df_kelamin["Kelamin"] == "N/A"
    ).astype(int)

    df_kelamin = df_kelamin.drop(columns=["Kelamin"])

    df_usia = pd.read_excel(file_usia)

    df_usia["MRN"] = df_usia["MRN"].astype(str)

    df_usia = df_usia[
        df_usia["MRN"].isin(patient_list)
    ]

    drop_cols = [
        "billing_awal",
        "billing_akhir"
    ]

    df_usia = df_usia.drop(
        columns=[c for c in drop_cols if c in df_usia.columns]
    )

    df_usia = df_usia.drop_duplicates(subset="MRN")

    df_nefro = pd.read_excel(file_nefro)

    df_nefro = df_nefro.rename(columns={
        "MR No. / Vendor Code": "MRN"
    })

    df_nefro["MRN"] = df_nefro["MRN"].astype(str)

    df_nefro = df_nefro[
        df_nefro["MRN"].isin(patient_list)
    ]

    df_nefro = df_nefro[
        ["MRN", "nefrotoksik_akumulasi"]
    ].drop_duplicates(subset="MRN")

    df_egfr = pd.read_excel(file_egfr)

    df_egfr["MRN"] = df_egfr[
        "Medical Record No."
    ].astype(str)

    df_egfr = df_egfr[
        df_egfr["MRN"].isin(patient_list)
    ]

    df_egfr = df_egfr[
        ["MRN", "delta", "days"]
    ].drop_duplicates(subset="MRN")

    df_cci = pd.read_excel(file_cci)

    possible_mrn_cols = [
        "Medical Record No.",
        "MRN",
        "MR No. / Vendor Code"
    ]

    cci_mrn_col = next(
        c for c in possible_mrn_cols
        if c in df_cci.columns
    )

    df_cci["MRN"] = df_cci[cci_mrn_col].astype(str)

    df_cci = df_cci[
        df_cci["MRN"].isin(patient_list)
    ]

    df_cci = df_cci[
        ["MRN", "CCI_uScore"]
    ].drop_duplicates(subset="MRN")


    df_diag = pd.read_excel(file_diag)

    diag_mrn_col = next(
        c for c in possible_mrn_cols
        if c in df_diag.columns
    )

    df_diag["MRN"] = df_diag[diag_mrn_col].astype(str)

    df_diag = df_diag[
        df_diag["MRN"].isin(patient_list)
    ]

    drop_diag_cols = [
        "MRN",
        "Medical Record No.",
        "billing_awal",
        "billing_akhir"
    ]

    diag_feature_cols = [
        c for c in df_diag.columns
        if c not in drop_diag_cols
    ]

    df_diag = df_diag[
        ["MRN"] + diag_feature_cols
    ]

    df_diag = df_diag.drop_duplicates(subset="MRN")

    # MERGE SEMUA

    master = master.merge(df_diur, on="MRN", how="left")

    master = master.merge(df_kelamin, on="MRN", how="left")

    master = master.merge(df_usia, on="MRN", how="left")

    master = master.merge(df_nefro, on="MRN", how="left")

    master = master.merge(df_egfr, on="MRN", how="left")

    master = master.merge(df_cci, on="MRN", how="left")

    master = master.merge(df_diag, on="MRN", how="left")

    onehot_cols = [
        c for c in master.columns
        if c.startswith("diur_")
    ]

    master[onehot_cols] = master[onehot_cols].fillna(0)

    diag_cols = [
        c for c in diag_feature_cols
        if c in master.columns
    ]

    master[diag_cols] = master[diag_cols].fillna(0)

    if "CCI_uScore" in master.columns:
        master["CCI_uScore"] = master["CCI_uScore"].fillna(0)

    if "nefrotoksik_akumulasi" in master.columns:
        master["nefrotoksik_akumulasi"] = (
            master["nefrotoksik_akumulasi"].fillna(0)
        )


    if output_file:
        master.to_excel(output_file, index=False)

    return master

def build_master_dataset_cci_com_hiperpol(
    file_pasien="RMDiuretikFIX.xlsx",
    file_diuretik="diuretikfinal3bulan.xlsx",
    file_usia="merge_with_usia.xlsx",
    file_nefro="NefrotoksikAkumulasi.xlsx",
    file_egfr="egfr_delta.xlsx",
    file_cci="CCIAkumulasi.xlsx",
    file_diag="StatusKomorbid_GGElektro.xlsx",
    file_polifarmasi="Hiperpolifarmasi.xlsx",
    output_file=None
):
    df_base = pd.read_excel(file_pasien)

    df_base["MRN"] = df_base["Medical Record No."].astype(str)

    patient_list = df_base["MRN"].unique()

    master = pd.DataFrame({
        "MRN": patient_list
    })


    df_diur = pd.read_excel(file_diuretik)

    df_diur = df_diur.rename(columns={
        "MR No. / Vendor Code": "MRN"
    })

    df_diur["MRN"] = df_diur["MRN"].astype(str)

    df_diur = df_diur[
        df_diur["MRN"].isin(patient_list)
    ]

    # ONE HOT
    subgol_dummies = pd.get_dummies(
        df_diur["Sub_Golongan"],
        prefix="diur"
    )

    df_diur = pd.concat([
        df_diur[["MRN"]],
        subgol_dummies
    ], axis=1)

    df_diur = df_diur.groupby("MRN").max().reset_index()


    df_kelamin = pd.read_excel(file_diuretik)

    df_kelamin = df_kelamin.rename(columns={
        "MR No. / Vendor Code": "MRN"
    })

    df_kelamin["MRN"] = df_kelamin["MRN"].astype(str)

    df_kelamin = df_kelamin[
        df_kelamin["MRN"].isin(patient_list)
    ]

    df_kelamin = df_kelamin.groupby("MRN")["Kelamin"].agg(
        lambda x: x.dropna().iloc[0]
        if len(x.dropna()) > 0
        else "N/A"
    ).reset_index()

    df_kelamin["kelamin_P"] = (
        df_kelamin["Kelamin"] == "Perempuan"
    ).astype(int)

    df_kelamin["kelamin_L"] = (
        df_kelamin["Kelamin"] == "Pria"
    ).astype(int)

    df_kelamin["kelamin_NA"] = (
        df_kelamin["Kelamin"] == "N/A"
    ).astype(int)

    df_kelamin = df_kelamin.drop(columns=["Kelamin"])

    df_usia = pd.read_excel(file_usia)

    df_usia["MRN"] = df_usia["MRN"].astype(str)

    df_usia = df_usia[
        df_usia["MRN"].isin(patient_list)
    ]

    drop_cols = [
        "billing_awal",
        "billing_akhir"
    ]

    df_usia = df_usia.drop(
        columns=[c for c in drop_cols if c in df_usia.columns]
    )

    df_usia = df_usia.drop_duplicates(subset="MRN")

    df_nefro = pd.read_excel(file_nefro)

    df_nefro = df_nefro.rename(columns={
        "MR No. / Vendor Code": "MRN"
    })

    df_nefro["MRN"] = df_nefro["MRN"].astype(str)

    df_nefro = df_nefro[
        df_nefro["MRN"].isin(patient_list)
    ]

    df_nefro = df_nefro[
        ["MRN", "nefrotoksik_akumulasi"]
    ].drop_duplicates(subset="MRN")

    df_egfr = pd.read_excel(file_egfr)

    df_egfr["MRN"] = df_egfr[
        "Medical Record No."
    ].astype(str)

    df_egfr = df_egfr[
        df_egfr["MRN"].isin(patient_list)
    ]

    df_egfr = df_egfr[
        ["MRN", "delta", "days"]
    ].drop_duplicates(subset="MRN")

    df_cci = pd.read_excel(file_cci)

    possible_mrn_cols = [
        "Medical Record No.",
        "MRN",
        "MR No. / Vendor Code"
    ]

    cci_mrn_col = next(
        c for c in possible_mrn_cols
        if c in df_cci.columns
    )

    df_cci["MRN"] = df_cci[cci_mrn_col].astype(str)

    df_cci = df_cci[
        df_cci["MRN"].isin(patient_list)
    ]

    df_cci = df_cci[
        ["MRN", "CCI_uScore"]
    ].drop_duplicates(subset="MRN")


    df_diag = pd.read_excel(file_diag)

    diag_mrn_col = next(
        c for c in possible_mrn_cols
        if c in df_diag.columns
    )

    df_diag["MRN"] = df_diag[diag_mrn_col].astype(str)

    df_diag = df_diag[
        df_diag["MRN"].isin(patient_list)
    ]

    drop_diag_cols = [
        "MRN",
        "Medical Record No.",
        "billing_awal",
        "billing_akhir"
    ]

    diag_feature_cols = [
        c for c in df_diag.columns
        if c not in drop_diag_cols
    ]

    df_diag = df_diag[
        ["MRN"] + diag_feature_cols
    ]

    df_diag = df_diag.drop_duplicates(subset="MRN")
    df_poly = pd.read_excel(file_polifarmasi)

    df_poly["MRN"] = df_poly["MR"].astype(str)

    df_poly = df_poly[
        df_poly["MRN"].isin(patient_list)
    ]

    df_poly = df_poly[
        ["MRN", "max_jumlah_item_per_hari"]
    ].drop_duplicates(subset="MRN")

    # <10 = 0, >=10 = 1
    df_poly["polifarmasi"] = (
        df_poly["max_jumlah_item_per_hari"] >= 10
    ).astype(int)

    df_poly = df_poly[
        ["MRN", "polifarmasi"]
    ]


    master = master.merge(df_diur, on="MRN", how="left")

    master = master.merge(df_kelamin, on="MRN", how="left")

    master = master.merge(df_usia, on="MRN", how="left")

    master = master.merge(df_nefro, on="MRN", how="left")

    master = master.merge(df_egfr, on="MRN", how="left")

    master = master.merge(df_cci, on="MRN", how="left")

    master = master.merge(df_diag, on="MRN", how="left")

    master = master.merge(df_poly, on="MRN", how="left")

    onehot_cols = [
        c for c in master.columns
        if c.startswith("diur_")
    ]

    master[onehot_cols] = master[onehot_cols].fillna(0)

    diag_cols = [
        c for c in diag_feature_cols
        if c in master.columns
    ]

    master[diag_cols] = master[diag_cols].fillna(0)

    if "CCI_uScore" in master.columns:
        master["CCI_uScore"] = master["CCI_uScore"].fillna(0)

    if "nefrotoksik_akumulasi" in master.columns:
        master["nefrotoksik_akumulasi"] = (
            master["nefrotoksik_akumulasi"].fillna(0)
        )

    if "polifarmasi" in master.columns:
        master["polifarmasi"] = (
            master["polifarmasi"].fillna(0).astype(int)
        )

    if output_file:
        master.to_excel(output_file, index=False)

    return master

def run_ml_experiment(
    file_path,
    output_excel
):

    df = pd.read_excel(file_path)

    df = df.dropna(subset=["delta"])

    # TARGET
    y = df["delta"]

    # MRN
    mrn = df["MRN"]

    # FEATURES
    X = df.drop(columns=[
        "MRN",
        "delta"
    ])

    X = X.fillna(0)

    param_grid = {
        "n_estimators": [50, 100, 1000],
        "max_features": ["log2", "sqrt"],
        "max_depth": [
            None, 5, 10, 15, 20,
            25, 30, 35, 40, 45, 50
        ],
        "min_samples_split": [2, 3, 5, 7, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "bootstrap": [True, False]
    }

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("=== TUNING HYPERPARAMETER ===")

    grid = GridSearchCV(
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        param_grid,
        cv=kfold,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    print("Best params:", grid.best_params_)

    results = []

    for run in range(5):

        print(f"\n===== TEST RUN {run+1} =====")

        X_train, X_test, y_train, y_test, mrn_train, mrn_test = train_test_split(
            X,
            y,
            mrn,
            test_size=0.2,
            random_state=run
        )

        model = RandomForestRegressor(
            **grid.best_params_,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)

        mae = mean_absolute_error(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        spearman_corr, _ = spearmanr(
            y_test,
            y_pred
        )

        print(f"R2: {r2:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Spearman: {spearman_corr:.3f}")

        # =================================================
        # SAVE PREDICTION
        # =================================================
        pred_df = pd.DataFrame({
            "MRN": mrn_test.values,
            "y_true": y_test.values,
            "y_pred": y_pred
        })

        pred_path = output_excel.replace(
            ".xlsx",
            f"_prediction_run{run+1}.xlsx"
        )

        pred_df.to_excel(pred_path, index=False)

        # =================================================
        # Y-Y PLOT
        # =================================================
        plt.figure(figsize=(6, 6))

        plt.scatter(y_test, y_pred, alpha=0.7)

        min_val = min(
            y_test.min(),
            y_pred.min()
        )

        max_val = max(
            y_test.max(),
            y_pred.max()
        )

        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--"
        )

        plt.xlabel("Actual Delta")

        plt.ylabel("Predicted Delta")

        plt.title(f"Y-Y Plot Run {run+1}")

        plt.tight_layout()

        plot_path = output_excel.replace(
            ".xlsx",
            f"_yyplot_run{run+1}.png"
        )

        plt.savefig(plot_path, dpi=300)

        plt.close()

        results.append({
            "run": run+1,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "spearman": spearman_corr
        })

    results_df = pd.DataFrame(results)

    summary = pd.DataFrame({
        "metric": ["mean", "std"],
        "r2": [
            results_df["r2"].mean(),
            results_df["r2"].std()
        ],
        "mae": [
            results_df["mae"].mean(),
            results_df["mae"].std()
        ],
        "rmse": [
            results_df["rmse"].mean(),
            results_df["rmse"].std()
        ],
        "spearman": [
            results_df["spearman"].mean(),
            results_df["spearman"].std()
        ]
    })

    joblib.dump(model,"diur_stratify_weight_akum_diagawal_ori_ynondelta.pkl")

    with pd.ExcelWriter(output_excel) as writer:

        results_df.to_excel(
            writer,
            sheet_name="per_run",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="summary",
            index=False
        )

    print("\n=== SELESAI ===")

def run_ml_experiment_normalized(
    file_path,
    output_excel
):

    df = pd.read_excel(file_path)

    df = df.dropna(subset=[
        "delta",
        "days"
    ])

    # HINDARI DIVIDE BY ZERO
    df = df[df["days"] > 0]

    # TARGET BARU
    df["delta_per_day"] = (
        df["delta"] / df["days"]
    )

    y = df["delta_per_day"]

    mrn = df["MRN"]

    X = df.drop(columns=[
        "MRN",
        "delta",
        "delta_per_day",
        "days"
    ])

    X = X.fillna(0)

    param_grid = {
        "n_estimators": [50, 100, 1000],
        "max_features": ["log2", "sqrt"],
        "max_depth": [
            None, 5, 10, 15, 20,
            25, 30, 35, 40, 45, 50
        ],
        "min_samples_split": [2, 3, 5, 7, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "bootstrap": [True, False]
    }

    stratifykfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("=== TUNING HYPERPARAMETER NORMALIZED ===")

    grid = GridSearchCV(
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        param_grid,
        cv=stratifykfold,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    print("Best params:", grid.best_params_)

    results = []

    for run in range(5):

        print(f"\n===== TEST RUN {run+1} =====")

        X_train, X_test, y_train, y_test, mrn_train, mrn_test = train_test_split(
            X,
            y,
            mrn,
            test_size=0.2,
            random_state=run
        )

        model = RandomForestRegressor(
            **grid.best_params_,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)

        mae = mean_absolute_error(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        spearman_corr, _ = spearmanr(
            y_test,
            y_pred
        )

        print(f"R2: {r2:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Spearman: {spearman_corr:.3f}")

        # =================================================
        # SAVE PREDICTION
        # =================================================
        pred_df = pd.DataFrame({
            "MRN": mrn_test.values,
            "y_true": y_test.values,
            "y_pred": y_pred
        })

        pred_path = output_excel.replace(
            ".xlsx",
            f"_prediction_run{run+1}.xlsx"
        )

        pred_df.to_excel(pred_path, index=False)

        # =================================================
        # Y-Y PLOT
        # =================================================
        plt.figure(figsize=(6, 6))

        plt.scatter(y_test, y_pred, alpha=0.7)

        min_val = min(
            y_test.min(),
            y_pred.min()
        )

        max_val = max(
            y_test.max(),
            y_pred.max()
        )

        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--"
        )

        plt.xlabel("Actual Delta/Day")

        plt.ylabel("Predicted Delta/Day")

        plt.title(f"Y-Y Plot Run {run+1}")

        plt.tight_layout()

        plot_path = output_excel.replace(
            ".xlsx",
            f"_yyplot_run{run+1}.png"
        )

        plt.savefig(plot_path, dpi=300)

        plt.close()

        results.append({
            "run": run+1,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "spearman": spearman_corr
        })

    results_df = pd.DataFrame(results)

    summary = pd.DataFrame({
        "metric": ["mean", "std"],
        "r2": [
            results_df["r2"].mean(),
            results_df["r2"].std()
        ],
        "mae": [
            results_df["mae"].mean(),
            results_df["mae"].std()
        ],
        "rmse": [
            results_df["rmse"].mean(),
            results_df["rmse"].std()
        ],
        "spearman": [
            results_df["spearman"].mean(),
            results_df["spearman"].std()
        ]
    })
    
    joblib.dump(model,"diur_regresi_stratify_weight_akum_normalized_yperdelta.pkl")
    
    with pd.ExcelWriter(output_excel) as writer:

        results_df.to_excel(
            writer,
            sheet_name="per_run",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="summary",
            index=False
        )

    print("\n=== SELESAI ===")

def run_rf_classification(
    file_path,
    output_excel,
    batas_atas=0,
    batas_bawah=-2
):

    df = pd.read_excel(file_path)

    df = df.dropna(subset=["delta", "days"])

    df = df[df["days"] > 0]

    # NORMALISASI KE 90 HARI
    # delta_90 = (delta / days) * 90
    df["delta_90hari"] = (
        df["delta"] / df["days"]
    ) * 90

    # LABELING
    # delta > batas_atas  -> 0; delta < batas_bawah -> 1
    df = df[
        (df["delta_90hari"] > batas_atas) |
        (df["delta_90hari"] < batas_bawah)
    ].copy()

    df["target"] = np.where(
        df["delta_90hari"] < batas_bawah,
        1,
        0
    )

    print("\n=== DISTRIBUSI TARGET ===")
    print(df["target"].value_counts())

    # TARGET
    y = df["target"]

    mrn = df["MRN"]

    X = df.drop(columns=[
        "MRN",
        "delta",
        "days",
        "delta_90hari",
        "target"
    ])

    X = X.fillna(0)

    # HYPERPARAMETER
    param_grid = {
        "n_estimators": [50, 100, 1000],
        "max_features": ["log2", "sqrt"],
        "max_depth": [
            None, 1, 2, 3, 4, 5, 10, 15, 20,
            25, 30, 35, 40, 45, 50
        ],
        "min_samples_split": [2, 3, 5, 7, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "bootstrap": [True, False]
    }

    print("\n=== TUNING RF CLASSIFIER ===")

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        
        ),
        param_grid,
        cv=kfold,
        scoring="average_precision",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    print("Best params:", grid.best_params_)

    results = []

    for run in range(5):

        print(f"\n===== TEST RUN {run+1} =====")

        X_train, X_test, y_train, y_test, mrn_train, mrn_test = train_test_split(
            X,
            y,
            mrn,
            test_size=0.2,
            random_state=run,
            stratify=y
        )

        model = RandomForestClassifier(
            **grid.best_params_,
            random_state=42,
            n_jobs=-1,
        
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        # METRICS
        acc = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall   : {recall:.3f}")
        print(f"F1 Score : {f1:.3f}")
        print(f"AUC      : {auc:.3f}")

        print("\nConfusion Matrix")
        print(cm)

        # SAVE PREDICTION
        pred_df = pd.DataFrame({
            "MRN": mrn_test.values,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_prob": y_prob
        })

        pred_path = output_excel.replace(
            ".xlsx",
            f"_prediction_run{run+1}.xlsx"
        )

        pred_df.to_excel(pred_path, index=False)

        # SAVE RESULT
        results.append({
            "run": run+1,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        })

    # SAVE MODEL
    joblib.dump(model,"kfold_Precision_noclassweight_numweight_akum_diur.pkl")

    # SUMMARY
    results_df = pd.DataFrame(results)

    summary = pd.DataFrame({
        "metric": ["mean", "std"],

        "accuracy": [
            results_df["accuracy"].mean(),
            results_df["accuracy"].std()
        ],

        "precision": [
            results_df["precision"].mean(),
            results_df["precision"].std()
        ],

        "recall": [
            results_df["recall"].mean(),
            results_df["recall"].std()
        ],

        "f1": [
            results_df["f1"].mean(),
            results_df["f1"].std()
        ],

        "auc": [
            results_df["auc"].mean(),
            results_df["auc"].std()
        ]
    })

    with pd.ExcelWriter(output_excel) as writer:

        results_df.to_excel(
            writer,
            sheet_name="per_run",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="summary",
            index=False
        )

    print("\n=== SELESAI RF CLASSIFICATION ===")

def plot_confusion_matrix_excel(
    excel_path,
    output_png="confusion_matrix.png",
    y_true_col="y_true",
    y_pred_col="y_pred"
):

    df = pd.read_excel(excel_path)

    y_true = df[y_true_col]
    y_pred = df[y_pred_col]

    cm = confusion_matrix(y_true, y_pred)

    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    # CUSTOM COLOR MATRIX
    # urutan:
    # [[TN, FP],
    #  [FN, TP]]

    color_matrix = np.array([
        ["pink", "lightgreen"],
        ["lightblue", "steelblue"]
    ])

    # CONVERT COLOR TO RGB
    rgb_matrix = np.empty((2,2,4))

    for i in range(2):
        for j in range(2):
            rgb_matrix[i,j] = mcolors.to_rgba(
                color_matrix[i,j]
            )

    # PLOT
    fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(rgb_matrix)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                color="black",   # TEXT HITAM
                fontweight="bold"
            )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    ax.set_title("Confusion Matrix", fontsize=14)

    plt.tight_layout()

    # SAVE
    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nPlot saved to: {output_png}")

def extract_rf_feature_importance(
    model_path,
    feature_columns,
    output_excel=None,
    top_n=None,
    plot=True
):
    """
    Extract feature importance dari RandomForestClassifier (.pkl)

    Parameters
    ----------
    model_path : str
        Path file .pkl hasil joblib.dump()

    feature_columns : list
        List nama feature sesuai urutan saat training

    output_excel : str, optional
        Path excel output

    top_n : int, optional
        Ambil top N feature saja

    plot : bool
        Apakah ditampilkan barplot

    Returns
    -------
    importance_df : pd.DataFrame
    """

    model = joblib.load(model_path)

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    # TOP N
    if top_n is not None:
        importance_df = importance_df.head(top_n)

    print("\n=== FEATURE IMPORTANCE ===")
    print(importance_df)

    if output_excel is not None:
        importance_df.to_excel(output_excel, index=False)
        print(f"\nSaved to: {output_excel}")

    if plot:

        plt.figure(figsize=(10, max(5, len(importance_df)*0.3)))

        plt.barh(
            importance_df["feature"][::-1],
            importance_df["importance"][::-1]
        )

        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Random Forest Feature Importance")

        plt.tight_layout()
        plt.show()

    return importance_df

def evaluate_prediction_folder(
    folder_path,
    output_file="summary_mcc_auprc.xlsx",
    true_col="y_true",
    pred_col="y_pred",
    prob_col="y_prob"
):
    """
    Hitung MCC dan AUPRC untuk semua file excel dalam folder.
    """
    files = glob.glob(os.path.join(folder_path, "*.xlsx"))

    files = [
    f for f in files
    if "run" in os.path.basename(f).lower()
    ]

    results = []

    for file in files:

        try:
            df = pd.read_excel(file)

            mcc = matthews_corrcoef(
                df[true_col],
                df[pred_col]
            )

            auprc = average_precision_score(
                df[true_col],
                df[prob_col]
            )

            results.append({
                "file": os.path.basename(file),
                "n_data": len(df),
                "MCC": mcc,
                "AUPRC": auprc
            })

            print(f"OK : {os.path.basename(file)}")

        except Exception as e:
            print(f"ERROR : {os.path.basename(file)} -> {e}")

    result_df = pd.DataFrame(results)

    mean_row = {
        "file": "MEAN",
        "n_data": result_df["n_data"].mean(),
        "MCC": result_df["MCC"].mean(),
        "AUPRC": result_df["AUPRC"].mean()
    }

    std_row = {
        "file": "STD",
        "n_data": result_df["n_data"].std(),
        "MCC": result_df["MCC"].std(),
        "AUPRC": result_df["AUPRC"].std()
    }

    result_df = pd.concat(
        [
            result_df,
            pd.DataFrame([mean_row, std_row])
        ],
        ignore_index=True
    )

    output_path = os.path.join(folder_path, output_file)

    result_df.to_excel(output_path, index=False)

    print("\nHasil disimpan ke:")
    print(output_path)

    return result_df

if __name__ == "__main__":  
    if 0: # eGFRR delta
        df_delta = build_egfr_delta(
        "D:\File_Syaki\SKRIPSINEW\kurasiegfrcombinedinterval.xlsx",
        output_file="D:\File_Syaki\SKRIPSINEW\egfr_delta.xlsx"
        )

        print("selesai")

    if 0: # Build Master Dataset (Menggabungkan Data untuk ML)
        df_master = build_master_dataset(
                file_pasien=r"D:\File_Syaki\SKRIPSINEW\variabelX\RMDiuretikFIX.xlsx",
                file_diuretik=r"D:\File_Syaki\SKRIPSINEW\variabelX\diuretikfinal3bulan.xlsx",
                file_usia=r"d:\File_Syaki\SKRIPSINEW\variabelX\merge_with_usia.xlsx",
                file_nefro=r"D:\File_Syaki\SKRIPSINEW\variabelX\NefrotoksikAkumulasi.xlsx",
                file_egfr=r"D:\File_Syaki\SKRIPSINEW\variabelX\egfr_delta.xlsx",
                output_file=r"D:\File_Syaki\SKRIPSINEW\filemaster\master_onehot_akum_diuretik.xlsx"
            )
        
        print("selesai")

    if 0: # Build Master Dataset (add CCI, Komorbid, dan Gangguan Elektrolit)
            df_master = build_master_dataset_cci_com(
                file_pasien=r"D:\File_Syaki\SKRIPSINEW\variabelX\RMDiuretikFIX.xlsx",
                file_diuretik=r"D:\File_Syaki\SKRIPSINEW\variabelX\diuretikfinal3bulan.xlsx",
                file_usia=r"d:\File_Syaki\SKRIPSINEW\variabelX\merge_with_usia.xlsx",
                file_nefro=r"D:\File_Syaki\SKRIPSINEW\variabelX\NefrotoksikAkumulasi.xlsx",
                file_egfr=r"D:\File_Syaki\SKRIPSINEW\variabelX\egfr_delta.xlsx",
                file_cci=r"D:\File_Syaki\SKRIPSINEW\variabelX\CCIAkumulasi.xlsx",
                file_diag=r"D:\File_Syaki\SKRIPSINEW\variabelX\StatusKomorbid_GGElektro.xlsx",
                output_file=r"D:\File_Syaki\SKRIPSINEW\filemaster\master_weight_akum_diuretik.xlsx"
                )
            
            print("selesai")

    if 0: # Build Master Dataset (add Hiperpolifarmasi)
            df_master = build_master_dataset_cci_com_hiperpol(
                file_pasien=r"D:\File_Syaki\SKRIPSINEW\variabelX\RMDiuretikFIX.xlsx",
                file_diuretik=r"D:\File_Syaki\SKRIPSINEW\variabelX\diuretikfinal3bulan.xlsx",
                file_usia=r"d:\File_Syaki\SKRIPSINEW\variabelX\merge_diagnosaawal_diur_with_usia.xlsx",
                file_nefro=r"D:\File_Syaki\SKRIPSINEW\variabelX\NefrotoksikAkumulasi.xlsx",
                file_egfr=r"D:\File_Syaki\SKRIPSINEW\variabelX\egfr_delta.xlsx",
                file_cci=r"D:\File_Syaki\SKRIPSINEW\variabelX\CCIAkumulasi.xlsx",
                file_diag=r"D:\File_Syaki\SKRIPSINEW\variabelX\StatusKomorbid_GGElektro.xlsx",
                file_polifarmasi=r"D:\File_Syaki\SKRIPSINEW\variabelX\Hiperpolifarmasi.xlsx",
                output_file=r"D:\File_Syaki\SKRIPSINEW\filemaster\master_weight_akum_diuretik_diagawal.xlsx"
                )
            
            print("selesai")

    if 0: # ML RF REGRESI ORIGINAL DELTA   
        run_ml_experiment(
            file_path=r"D:\File_Syaki\SKRIPSINEW\filemaster\master_weight_akum_diuretik_diagawal.xlsx",
            output_excel=r"D:\File_Syaki\SKRIPSINEW\HASILRF\Regresi\diag awal\stratify_weight_akum_diur_diagawal_ori.xlsx"
            )
        
        print("selesai original")

    if 1: # ML RF REGRESI NORMALIZED DELTA (y adalah y dibagi delta days, tidak dijadikan x lagi delta daysnya)
        run_ml_experiment_normalized(
            file_path=r"D:\File_Syaki\SKRIPSINEW\filemaster\master_weight_akum_diuretik.xlsx",
            output_excel=r"D:\File_Syaki\SKRIPSINEW\HASILRF\Regresi\diagawal\stratify_weight_akum_diur_diag awal_normalized.xlsx"
            )

        print("selesai normalized")

    if 0: # ML RF KLASIFIKASI
        run_rf_classification(
            file_path=r"D:\File_Syaki\SKRIPSINEW\filemaster\master_weight_akum_diuretik.xlsx",
            output_excel=r"D:\File_Syaki\SKRIPSINEW\HASILRF\Klasifikasi\kfold_Precision_noclassweight_numweight_akum_diur.xlsx",
            batas_atas=0,
            batas_bawah=-2
            )
        
        print("selesai rf classification")

    if 0: # Confussion Matrix
        plot_confusion_matrix_excel(
            excel_path=r"D:\File_Syaki\SKRIPSINEW\HASILRF\Klasifikasi\precision_weight_akum_diur_original\Precision_classweight_numweight_akum_diur_prediction_run3.xlsx",
            output_png="cm_run1.png"
            )
        
        print("selesai CM") 

    if 0: # Feature Importance
        model_path=r"C:/Users/rashi/SKRIPSI/HASILRF/Regresi/raasi_onehot_akum_yperdelta.pkl"
        file_path=r"C:/Users/rashi/SKRIPSI/MasterRAASi/master_onehot_akum_raasi.xlsx"
        output_excel=r"C:/Users/rashi/SKRIPSI/HASILRF/Regresi/hasil_onehot_akum_raasi_normalized_feature_importance.xlsx"
        batas_atas=0,
        batas_bawah=-2

        df = pd.read_excel(file_path)

        #df = df.dropna(subset=["delta", "days"])

        df = df[df["days"] > 0]

        df["delta_90hari"] = (
            df["delta"] / df["days"]
        ) * 90

        df = df[
            (df["delta_90hari"] > batas_atas) |
            (df["delta_90hari"] < batas_bawah)
        ].copy()

        df["target"] = np.where(
            df["delta_90hari"] < batas_bawah,
            1,
            0
        )

        X = df.drop(columns=[
            "MRN",
            "delta",
            "days",
            "delta_90hari",
            "target"
        ])

        X = X.fillna(0)

        feat_imp = extract_rf_feature_importance(
            model_path,
            feature_columns=X.columns.tolist(),
            output_excel=output_excel.replace(".xlsx", "_feature_importance.xlsx"),

            top_n=30
        )

        print("selesai")
    
    if 0: # MCC & AUPRC
        folder = r"D:\File_Syaki\SKRIPSINEW\HASILRF\Klasifikasi\Precision_classweight_numweight_akum"

        summary = evaluate_prediction_folder(
            folder_path=folder,
            output_file=f"{folder}/summary_MCC_AUPRC.xlsx"
        )

        print(summary)
