import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import joblib

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
    
    # tiap ganti file utk variasi ini harus disesuaikan dgn kolom
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


from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from scipy.stats import spearmanr


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

    joblib.dump(model,"diur_weight_maks_ynondelta.pkl")

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

    kfold = KFold(
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
    
    joblib.dump(model,"diur_weight_maks_yperdelta.pkl")
    
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


from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             roc_auc_score, 
                             confusion_matrix)

from sklearn.ensemble import (RandomForestClassifier)

def run_rf_classification(
    file_path,
    output_excel,
    batas_atas=-9,
    batas_bawah=-15
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
            None, 5, 10, 15, 20,
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
            n_jobs=-1
        ),
        param_grid,
        cv=kfold,
        scoring="f1",
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
            n_jobs=-1
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
    joblib.dump(
        model,
        output_excel.replace(".xlsx", ".pkl")
    )

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

if __name__ == "__main__":  
    if 0: # eGFRR delta
        df_delta = build_egfr_delta(
        "/Users/lathifasyakira/Desktop/SKRIPSI/kurasiegfrcombinedinterval.xlsx",
        output_file="/Users/lathifasyakira/Desktop/SKRIPSI/egfr_delta.xlsx"
        )

        print("selesai")

    if 0: # Build Master Dataset (Menggabungkan Data untuk ML)
        df_master = build_master_dataset(
            file_pasien="/Users/lathifasyakira/Desktop/SKRIPSI/variabelX/RMDiuretik.xlsx",
            file_diuretik="/Users/lathifasyakira/Desktop/SKRIPSI/variabelX/diuretikfinal3bulan.xlsx",
            file_usia="/Users/lathifasyakira/Desktop/SKRIPSI/variabelX/merge_with_usia.xlsx",
            file_nefro="/Users/lathifasyakira/Desktop/SKRIPSI/variabelX/NefrotoksikMaksimalisasi.xlsx",
            file_egfr="/Users/lathifasyakira/Downloads/egfr_delta_sekar.xlsx",
            output_file="master_weight_maks_diuretik.xlsx"
            )
        
        print("selesai")

    if 1: # ML RF REGRESI ORIGINAL DELTA   
        run_ml_experiment(
            file_path="/Users/lathifasyakira/Desktop/SKRIPSI/master_weight_maks_diuretik.xlsx",
            output_excel="/Users/lathifasyakira/Desktop/SKRIPSI/HASILRF/Regresi/hasil_weight_maks_diur.xlsx"
            )
        
        print("selesai original")

    if 0: # ML RF REGRESI NORMALIZED DELTA (y adalah y dibagi delta days, tidak dijadikan x lagi delta daysnya)
        run_ml_experiment_normalized(
            file_path="/Users/lathifasyakira/Desktop/SKRIPSI/master_weight_maks_diuretik.xlsx",
            output_excel="/Users/lathifasyakira/Desktop/SKRIPSI/HASILRF/Regresi/hasil_weight_maks_diur_normalized.xlsx"
            )

        print("selesai normalized")

    if 0: # ML RF KLASIFIKASI
        run_rf_classification(
            file_path="/Users/lathifasyakira/Desktop/SKRIPSI/master_weight_maks_diuretik.xlsx",
            output_excel="/Users/lathifasyakira/Desktop/SKRIPSI/HASILRF/Klasifikasi/hasil_class_weight_maks_diur.xlsx",
            batas_atas=-9,
            batas_bawah=-15
            )
        
        print("selesai rf classification") 
