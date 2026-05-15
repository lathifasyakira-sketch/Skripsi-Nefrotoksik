import glob, os, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def merge_common_patients(
    raasi_file="gabunganraasi3bulanegfr.xlsx",
    diuretik_file="gabungandiuretik3bulanegfr.xlsx",
    output_file="raasi_diuretik_common.xlsx",
    patient_col="Medical Record No."
):

    df_raasi = pd.read_excel(raasi_file)
    df_diuretik = pd.read_excel(diuretik_file)

    df_raasi["SOURCE"] = "RAASI"
    df_diuretik["SOURCE"] = "DIURETIK"

    # Ambil MR unik
    raasi_mr = set(df_raasi[patient_col].dropna())
    diuretik_mr = set(df_diuretik[patient_col].dropna())

    common_mr = raasi_mr.intersection(diuretik_mr)

    print("Jumlah pasien RAASI:", len(raasi_mr))
    print("Jumlah pasien DIURETIK:", len(diuretik_mr))
    print("Pasien yang ada di keduanya:", len(common_mr))

    raasi_filtered = df_raasi[df_raasi[patient_col].isin(common_mr)]
    diuretik_filtered = df_diuretik[df_diuretik[patient_col].isin(common_mr)]

    df_combined = pd.concat([raasi_filtered, diuretik_filtered], ignore_index=True)

    df_combined.to_excel(output_file, index=False)

    print(f"File saved: {output_file}")

    return df_combined

if __name__ == "__main__":
    if 1: # Kode untuk Mendapatkan Pasien RAASi + Diuretik
        df_raasi="/Users/lathifasyakira/Desktop/SKRIPSI/gabunganraasi3bulanegfr.xlsx"
        df_diuretik="/Users/lathifasyakira/Desktop/SKRIPSI/gabungandiuretik3bulanegfr.xlsx"
        file_output="/Users/lathifasyakira/Desktop/SKRIPSI/raasi_diuretik_common.xlsx"
        merge_common_patients(df_raasi, df_diuretik, file_output)
        
        print("selesai")