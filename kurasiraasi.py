import glob, os, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast, math

def filter_excel_by_keyword(
    excel_files_pattern,
    list_sortir,
    column_name="Item Name",
    output_prefix="SC",
    output_suffix="RAASi",
):
    """
    Filter Excel berdasarkan daftar nama parameter yang berbeda
    tapi memiliki konsep sama (exact matching, case-insensitive).
    """

    excel_files = glob.glob(excel_files_pattern)
    print(f"Found {len(excel_files)} Excel files.")

    # Normalisasi list ke lowercase untuk matching
    list_lower = {str(x).lower() for x in list_sortir}

    for file_path in excel_files:
        print(f"\nProcessing: {file_path}")

        df = pd.read_excel(file_path)

        if column_name not in df.columns:
            print(f"Kolom '{column_name}' tidak ditemukan. Skip.")
            continue

        # ===== Exact concept match (case-insensitive) =====
        mask = df[column_name].astype(str).str.lower().isin(list_lower)
        df_filtered = df[mask].copy()

        if df_filtered.empty:
            print("Tidak ada parameter cocok. Skip.")
            continue

        # ===== Ambil identifier dari nama file =====
        match = re.search(r'SC\s(.+?)\.xlsx', os.path.basename(file_path))
        if match:
            identifier = match.group(1).strip()
        else:
            identifier = os.path.splitext(os.path.basename(file_path))[0]

        new_file_name = f"{output_prefix} {identifier} {output_suffix}.xlsx"

        df_filtered.to_excel(new_file_name, index=False)
        print(f"Menyimpan {len(df_filtered)} baris → '{new_file_name}'")

    print("\nSelesai.")

def load_and_combine_excel_data(glob_pattern: str, 
                                year: int = 2025,
                                ):
    
    excel_files = glob.glob(glob_pattern)
    df_list = []

    for file_path in excel_files:
        df = pd.read_excel(file_path)

        month = file_path.split(' ')[-3].replace('.xlsx', '')

        df['Bulan'] = month
        df['Tahun'] = file_path.split(' ')[-2].replace('.xlsx', '')

        df_list.append(df)

    df_combined = pd.concat(df_list, ignore_index=True)
    
    return df_combined

def patient_visit_frequency(
    df,
    patient_col="Patient Name",
    date_col="Order Date",
    date_format="%d/%m/%Y %H:%M:%S",
    plot=True
):
    """
    Menghitung distribusi frekuensi interval kunjungan pasien (hari & minggu).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame berisi data kunjungan pasien.
    patient_col : str
        Kolom ID pasien (misal patient_id).
    date_col : str
        Kolom tanggal kunjungan.
    date_format : str
        Format datetime pada kolom tanggal.
  
    Returns
    -------
    intervals_df : pandas.DataFrame
        DataFrame interval antar kunjungan (hari & minggu).
    freq_days : pandas.Series
        Distribusi frekuensi interval (hari).
    freq_weeks : pandas.Series
        Distribusi frekuensi interval (mingguan, berbasis bin).
    """

    df = df.copy()

    # Pastikan datetime
    df[date_col] = pd.to_datetime(
        df[date_col],
        format=date_format,
        errors="coerce"
    )

    intervals_days = []
    intervals_weeks = []
    billing_no = []
    med_rec = []

    # Mengitung interval antar kunjungan per pasien
    # interval ga buat billing, tapi buat nentuin perbedaan uji egfr aja
    for _, group in df.groupby(patient_col):
        dates = group[date_col].dropna().sort_values()

        if len(dates) < 2:
            continue

        deltas = dates.diff().dropna()
        intervals_week = (deltas.dt.days.sum()/7)

        if intervals_week > 12:
            intervals_days.append(deltas.dt.days.sum())
            intervals_weeks.append(deltas.dt.days.sum() / 7)
            billing_no.append(list(group["Billing No."]))
            med_rec.append(group["MR No. / Vendor Code"].values[0])
        
    intervals_df = pd.DataFrame({
        "interval_days": intervals_days,
        "interval_weeks": intervals_weeks,
        "billing": billing_no,
        "Medical Record No.": med_rec
    })

    # Distribusi hari
    freq_days = intervals_df["interval_days"].value_counts().sort_index()

    # Bin mingguan 
    bins = [12, 24, 52, 1000]
    labels = [
        "3–6 bulan",
        "6–12 bulan",
        ">12 bulan"
    ]

    intervals_df["week_bin"] = pd.cut(
        intervals_df["interval_weeks"],
        bins=bins,
        labels=labels,
        right=False
    )

    freq_weeks = intervals_df["week_bin"].value_counts().sort_index()

    # Plot opsional
    if plot:
        # Histogram hari
        """plt.figure(figsize=(8, 5))
        plt.hist(intervals_df["interval_days"], bins=30, edgecolor="black")
        plt.xlabel("Interval Kunjungan (Hari)")
        plt.ylabel("Frekuensi")
        plt.title("Distribusi Interval Antar Kunjungan Pasien (Hari)")
        plt.grid(alpha=0.3)
        plt.show()"""

        # Bar plot minggu
        freq_weeks.plot(kind="bar", figsize=(9, 5), edgecolor="black")
        plt.ylabel("Frekuensi")
        plt.xlabel("Interval Antar Kunjungan")
        plt.title("Distribusi Interval Antar Kunjungan Pasien (Mingguan)")
        plt.grid(axis="y", alpha=0.3)
        plt.show()

    return intervals_df, freq_days, freq_weeks

def clean_patient_names(df, 
                        col="Patient Name / Vendor Name", 
                        mr_col="MR No. / Vendor Code",
                        dob_col="Date of Birth"):

    df = df.copy()

    if mr_col not in df.columns:
        df[mr_col] = None

    correction_mask = df[col].str.contains(r'^correction by dmp\s*-\s*', case=False, na=False)

    correction_names = df.loc[correction_mask, col].str.replace(
        r'^correction by dmp\s*-\s*', '', regex=True, flags=re.IGNORECASE
    )

    def normalize_name(x):
        if pd.isna(x):
            return ""
        x = re.sub(r"\.\s*(TN|NY)", "", x, flags=re.IGNORECASE)
        x = re.sub(r"-\d+.*", "", x)
        return x.strip().lower()

    df["_name_norm"] = df[col].apply(normalize_name)

    for idx, name in correction_names.items():

        norm = normalize_name(name)

        match = df[
            (~correction_mask) &
            (df["_name_norm"] == norm) &
            (df[mr_col].notna())
        ]

        if len(match) > 0:
            df.at[idx, mr_col] = match.iloc[0][mr_col]

    df[col] = df[col].str.replace(
        r'^correction by dmp\s*-\s*', '', regex=True, flags=re.IGNORECASE
    )

    df["Kelamin"] = "Tidak_ada"

    df.loc[df[col].str.contains(r"\.\s*tn", case=False, na=False), "Kelamin"] = "Pria"
    df.loc[df[col].str.contains(r"\,\s*tn", case=False, na=False), "Kelamin"] = "Pria"
    df.loc[df[col].str.contains(r"\.\s*ny", case=False, na=False), "Kelamin"] = "Perempuan"
    df.loc[df[col].str.contains(r"\,\s*ny", case=False, na=False), "Kelamin"] = "Perempuan"
    df.loc[df[col].str.contains(r"\.\s*nn", case=False, na=False), "Kelamin"] = "Perempuan"
    df.loc[df[col].str.contains(r"\,\s*nn", case=False, na=False), "Kelamin"] = "Perempuan"

    extracted_mr = df[col].str.extract(r'-(?:\d+)-(\d+)')[0]

    mask = df[mr_col].isna() | (df[mr_col].astype(str).str.strip() == "")
    df.loc[mask, mr_col] = extracted_mr[mask]

    def clean_name(x):
        if pd.isna(x):
            return x

        x = re.sub(r"\.\s*(TN|NY)", "", x, flags=re.IGNORECASE)
        x = re.sub(r"-\d+.*", "", x)

        return x.strip()

    df[col] = df[col].apply(clean_name)

    df["_name_key"] = df[col].str.lower().str.strip()

    # ---- (B) & (T) ----
    df["Patient Name / Vendor Name"] = df["Patient Name / Vendor Name"].str.replace(
        r"^\([A-Za-z]\)-", 
        "", 
        regex=True
        )
    
    # ---- Nama Kapital ----
    df["Patient Name / Vendor Name"] = df["Patient Name / Vendor Name"].str.title()
   
    # ---- MR ----
    mr_map = (
        df[df[mr_col].notna()]
        .groupby("_name_key")[mr_col]
        .first()
    )

    mask = df[mr_col].isna() | (df[mr_col].astype(str).str.strip() == "")
    df.loc[mask, mr_col] = df.loc[mask, "_name_key"].map(mr_map)

    #df["MR No. / Vendor Code"] = (
    #df["MR No. / Vendor Code"]
    #.astype(str)         
    #.str.replace(".0","", regex=False)  
    #.str.zfill(8)         
    #)

    # ---- Kelamin ----
    kel_map = (
        df[df["Kelamin"] != "Tidak_ada"]
        .groupby("_name_key")["Kelamin"]
        .first()
    )

    mask = df["Kelamin"] == "Tidak_ada"
    df.loc[mask, "Kelamin"] = df.loc[mask, "_name_key"].map(kel_map)

    # ---- Date of Birth ----
    if dob_col in df.columns:

        dob_map = (
            df[df[dob_col].notna()]
            .groupby("_name_key")[dob_col]
            .first()
        )

        mask = df[dob_col].isna()
        df.loc[mask, dob_col] = df.loc[mask, "_name_key"].map(dob_map)
    
    df = df.drop(columns=["_name_norm", "_name_key"])

    return df

def filter_patient_3_months_with_coverage(
    input_file="raasifinal.xlsx",
    output_file="raasifinal3bulan_80coverage.xlsx",
    patient_col="MR No. / Vendor Code",
    golob_file ="gol_ob.xlsx",
    date_col="Created Date"
):
    #delta days raasi yang bener

    import pandas as pd

    df = pd.read_excel(input_file)

    # PREPROCESS DATE
    df[date_col] = pd.to_datetime(
        df[date_col],
        dayfirst=True,
        errors="coerce"
    ).dt.normalize()  # hapus jam

    # VISIT RANGE (MIN MAX)
    visit_range = (
        df.groupby(patient_col)[date_col]
        .agg(["min", "max"])
    )

    visit_range["delta_days"] = (
        visit_range["max"] - visit_range["min"]
    ).dt.days

    # MONTH COVERAGE
    df["year_month"] = df[date_col].dt.to_period("M")

    monthly_counts = (
        df.groupby(patient_col)["year_month"]
        .nunique()
        .rename("months_with_data")
    )

    visit_range["total_months"] = (
        (visit_range["max"].dt.to_period("M") -
         visit_range["min"].dt.to_period("M"))
        .apply(lambda x: x.n) + 1
    )

    visit_range = visit_range.join(monthly_counts)

    visit_range["month_coverage"] = (
        visit_range["months_with_data"] /
        visit_range["total_months"]
    )

    # FILTERING
    valid_patients_3m = visit_range[
        visit_range["delta_days"] >= 90
    ].index

    valid_patients_80 = visit_range[
        (visit_range["delta_days"] >= 90) &
        (visit_range["month_coverage"] >= 0.8)
    ].index

    # MAP BACK TO DF
    df["delta_days_raasi"] = df[patient_col].map(
        visit_range["delta_days"]
    )

    df["month_coverage"] = df[patient_col].map(
        visit_range["month_coverage"]
    )

    # MAP GOLONGAN OBAT
    golongan_obat = pd.read_excel(golob_file)

    df["Golongan"] = df["Item Code"].map(
        golongan_obat.set_index("Item Code")["Golongan"]
    )

    df_filtered = df[
        df[patient_col].isin(valid_patients_80)
    ]

    # LOGGING
    print("Total pasien:", df[patient_col].nunique())
    print("Pasien ≥3 bulan:", len(valid_patients_3m))
    print("Pasien ≥3 bulan + coverage ≥80%:", len(valid_patients_80))
    print("Total baris tersimpan:", len(df_filtered))


    dict_sub={"ACEi":["KAPTOPRIL 25 MG TABLET","KAPTOPRIL 12,5 MG TABLET","KAPTOPRIL 50 MG TABLET","LISINOPRIL 5 MG TABLET","LISINOPRIL 10 MG TABLET","RAMIPRIL 2,5 MG TABLET","RAMIPRIL 10 MG TABLET","RAMIPRIL 5 MG TABLET","(FOI) RAMIPRIL 5 MG TABLET","EMERTEN 5 MG TABLET"],
              "ARB":["KANDESARTAN 16 MG TABLET","KANDESARTAN 8 MG TABLET","CANDOTENS 8 MG TABLET","CANDOTENS 16 MG TABLET","VALSARTAN 160 MG TABLET","VALSARTAN 80 MG TABLET","DIOVAN 80 MG TABLET","IRBESARTAN 150 MG TABLET","IRBESARTAN 300 MG TABLET","APROVEL 300 MG TABLET","MICARDIS 40 MG TABLET","MICARDIS 80 MG TABLET","TINOV 40 MG TABLET","TINOV 80 MG TABLET"],
              "ARNI+ARB":["UPERIO 50 MG TABLET","UPERIO 200 MG TABLET"],
              "ARB+CCB":["EXFORGE 10 MG/160 MG TABLET","EXFORGE 5 MG/80 MG TABLET"],
              "ARB+HCT":[ "CO IRVELL 300/12.5 MG TABLET","CO APROVEL 150 MG/12,5 MG TABLET"]
              }

    # flatten mapping
    item_to_group = {
        item.strip().upper(): key
        for key, items in dict_sub.items()
        for item in items
    }

    df_filtered["Sub_Golongan"] = df_filtered["Item Name"].map(item_to_group)

    # SAVE
    df_filtered.to_excel(output_file, index=False)

    print(f"File saved: {output_file}")
    
def filter_patient_3_months(
    input_file="raasifinal.xlsx",
    output_file="raasifinal3bulan.xlsx",
    patient_col="MR No. / Vendor Code",
    golob_file = "gol_ob.xlsx",
    date_col="Created Date"
):
    
    df = pd.read_excel(input_file)

    df[date_col] = pd.to_datetime(
        df[date_col],
        dayfirst=True,
        errors="coerce"
    ).dt.normalize()  # hapus jam

    visit_range = (
        df.groupby(patient_col)[date_col]
        .agg(["min", "max"])
    )

    visit_range["delta_days"] = (
        visit_range["max"] - visit_range["min"]
    ).dt.days

    # ≥ 3 bulan ≈ 90 hari
    valid_patients = visit_range[
        visit_range["delta_days"] >= 90
    ].index

    df["delta_days_raasi"] = df["MR No. / Vendor Code"].map(
    visit_range["delta_days"])
    
    golongan_obat = pd.read_excel(golob_file)
    df["Golongan"] = df["Item Code"].map(
    golongan_obat.set_index("Item Code")["Golongan"]
)
    df_filtered = df[
        df[patient_col].isin(valid_patients)
    ]

    print("Total pasien:", df[patient_col].nunique())
    print("Pasien ≥3 bulan:", len(valid_patients))
    print("Total baris tersimpan:", len(df_filtered))

    df_filtered.to_excel(output_file, index=False)

    print(f"File saved: {output_file}")

def raasi_egfr_combined(
    df_raasi,
    df_egfr,
    raasi_col="MR No. / Vendor Code",
    egfr_col="Medical Record No.",
    make_plot=False
):
    """
    1. Menghapus duplikasi MR pada masing-masing dataframe
    2. Menggabungkan pasien RAASI & eGFR
    3. Menghitung distribusi frekuensi
    4. (Opsional) Membuat bar plot frekuensi
    """

    raasi_unique = df_raasi.drop_duplicates(subset=raasi_col)
    egfr_unique = df_egfr.drop_duplicates(subset=egfr_col)

    print(f"RAASI unik : {len(raasi_unique)} pasien")
    print(f"eGFR unik  : {len(egfr_unique)} pasien")

    set_raasi = set(raasi_unique[raasi_col].dropna())
    set_egfr = set(egfr_unique[egfr_col].dropna())

    both = set_raasi & set_egfr
    only_raasi = set_raasi - set_egfr
    only_egfr = set_egfr - set_raasi

    freq_dict = {
        "RAASI only": len(only_raasi),
        "eGFR only": len(only_egfr),
        "Both RAASI & eGFR": len(both),
    }

    freq_df = pd.DataFrame.from_dict(freq_dict, orient="index", columns=["count"])

    print("\nDistribusi pasien:")
    print(freq_df)

    if make_plot:
        plt.figure(figsize=(6, 4))
        plt.bar(freq_df.index, freq_df["count"])
        plt.ylabel("Jumlah Pasien")
        plt.title("Distribusi Pasien RAASI vs eGFR")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()

    df_both = pd.merge(
        raasi_unique,
        egfr_unique,
        left_on=raasi_col,
        right_on=egfr_col,
        how="inner"
    )

    print(f"\nJumlah pasien dengan RAASI & eGFR: {len(df_both)}")

    return {
        "freq_df": freq_df,
        "df_both": df_both,
        "set_raasi": set_raasi,
        "set_egfr": set_egfr,
        "both": both
    }


def visit_interval_after_merge_try(
    df_both,
    patient_col="Medical Record No.",
    date_col="Order Date",
    make_plot=True
):
    """
    Menghitung distribusi interval kunjungan pasien
    SETELAH data RAASI dan eGFR digabung.
    """

    df = df_both.copy()

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    visit_range = (
        df.groupby(patient_col)[date_col]
        .agg(["min", "max"])
    )

    visit_range["delta_days"] = (
        visit_range["max"] - visit_range["min"]
    ).dt.days

    df["delta_days_egfr"] = df[patient_col].map(
        visit_range["delta_days"]
    )
    intervals = df["delta_days_egfr"].dropna()

    df = df[df["delta_days_egfr"] >= 90]
    
    bins = [90, 180, 365, np.inf]

    labels = [
        "3–6 bulan",
        "6–12 bulan",
        ">12 bulan"
    ]

    interval_cat = pd.cut(intervals, bins=bins, labels=labels, right=True)

    freq = interval_cat.value_counts().reindex(labels)

    print("\nDistribusi interval kunjungan (pasien RAASI + eGFR):")
    print(freq)

    if make_plot:
        plt.figure(figsize=(7, 4))
        plt.bar(freq.index.astype(str), freq.values)
        plt.ylabel("Jumlah Interval")
        plt.title("Distribusi Interval Kunjungan Pasien RAASI + eGFR")
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.show()

    return freq, df

    
def visit_interval_after_merge_old(
    df_both,
    patient_col="Medical Record No.",
    date_col="Order Date",
    make_plot=True
):
    """
    Menghitung distribusi interval kunjungan pasien
    SETELAH data RAASI dan eGFR digabung.
    """

    df = df_both.copy()

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    df = df.sort_values([patient_col, date_col])

    df["delta_days"] = (
        df.groupby(patient_col)[date_col]
        .diff()
        .dt.days
    )
    #memotong di bawah 3 bulan
    df = df[df["delta_days"]>90]
    
    visit_range = (
        df.groupby(patient_col)[date_col]
        .agg(["min", "max"])
    )

    intervals = df["delta_days"].dropna()

    bins = [90, 180, 365, np.inf]

    labels = [
        "3–6 bulan",
        "6–12 bulan",
        ">12 bulan"
    ]

    interval_cat = pd.cut(intervals, bins=bins, labels=labels, right=True)

    freq = interval_cat.value_counts().reindex(labels)

    print("\nDistribusi interval kunjungan (pasien RAASI + eGFR):")
    print(freq)


    if make_plot:
        plt.figure(figsize=(7, 4))
        plt.bar(freq.index.astype(str), freq.values)
        plt.ylabel("Jumlah Interval")
        plt.title("Distribusi Interval Kunjungan Pasien RAASI + eGFR")
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.show()

    return freq, df    

def get_raasi_egfr_longitudinal(
    df_raasi,
    df_egfr,
    raasi_col="MR No. / Vendor Code",
    egfr_col="Medical Record No.", 
    file_golob ="gol_ob.xlsx"
):
    """
    Ambil seluruh riwayat eGFR untuk pasien yang pernah mendapat RAASI.
    """

    # pasien RAASI unik
    raasi_patients = df_raasi[raasi_col].dropna().unique()

    # filter eGFR hanya pasien RAASI
    df_egfr_filtered = df_egfr[
        df_egfr[egfr_col].isin(raasi_patients)
    ].copy()

    print(f"Jumlah pasien RAASI: {len(raasi_patients)}")
    print(f"Jumlah baris eGFR untuk pasien RAASI: {len(df_egfr_filtered)}")    
    print("test")
    return df_egfr_filtered

def extract_billing(x):
    try:
        if pd.isna(x):
            return pd.Series([None, None])

        # Convert ke string
        x = str(x)

        # Ambil semua angka (termasuk float)
        numbers = re.findall(r"\d+\.?\d*", x)

        # Convert ke float → int
        numbers = [int(float(n)) for n in numbers]

        if len(numbers) > 0:
            return pd.Series([numbers[0], numbers[-1]])
        else:
            return pd.Series([None, None])

    except:
        return pd.Series([None, None])


if __name__ == "__main__":
    if 0: # Kode untuk Memfilter
        filter_excel_by_keyword(
        excel_files_pattern="/Users/lathifasyakira/Desktop/SKRIPSI/StockCard/SC *.xlsx",
        list_sortir=(
                 "KAPTOPRIL 25 MG TABLET",
                 "KAPTOPRIL 12,5 MG TABLET",
                 "KAPTOPRIL 50 MG TABLET",
                 "LISINOPRIL 5 MG TABLET",
                 "LISINOPRIL 10 MG TABLET",
                 "RAMIPRIL 2,5 MG TABLET",
                 "RAMIPRIL 10 MG TABLET",
                 "RAMIPRIL 5 MG TABLET",
                 "(FOI) RAMIPRIL 5 MG TABLET",
                 "EMERTEN 5 MG TABLET",
                 "KANDESARTAN 16 MG TABLET",
                 "KANDESARTAN 8 MG TABLET",
                 "CANDOTENS 8 MG TABLET",
                 "CANDOTENS 16 MG TABLET",
                 "VALSARTAN 160 MG TABLET",
                 "VALSARTAN 80 MG TABLET",
                 "DIOVAN 80 MG TABLET",
                 "IRBESARTAN 150 MG TABLET",
                 "IRBESARTAN 300 MG TABLET",
                 "APROVEL 300 MG TABLET",
                 "MICARDIS 40 MG TABLET",
                 "MICARDIS 80 MG TABLET",
                 "TINOV 40 MG TABLET",
                 "TINOV 80 MG TABLET",
                 "UPERIO 50 MG TABLET",
                 "UPERIO 200 MG TABLET",
                 "EXFORGE 10 MG/160 MG TABLET",
                 "EXFORGE 5 MG/80 MG TABLET",
                 "CO IRVELL 300/12.5 MG TABLET",
                 "CO APROVEL 150 MG/12,5 MG TABLET",
                 ),
        )

        print("selesai")

    if 0: # Kode untuk menggabungkan
        excel_file_template = "SC {Bulan} RAASi.xlsx"
        glob_pattern = f"/Users/lathifasyakira/Desktop/SKRIPSI/kurasiraasi/{excel_file_template.replace('{Bulan}', '*')}"
        df_combined = load_and_combine_excel_data(glob_pattern)
        df_combined.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiscraasicombined.xlsx")

    """if 0: #kode untuk visit freq
        df_combined = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiscraasicombined.xlsx", index_col=0)
        patient_visit_frequency(df_combined,
                                patient_col="Patient Name / Vendor Name", 
                                date_col="Created Date")

        print(df_combined.head())
        df_combined.info()"""
    
    if 0: # Kode untuk memfilter no MR, Jenis Kelamin, Nama Kapital, (T) (B)
        df_combined=pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiscraasicombined.xlsx", index_col=0)
        df_raasifinal=clean_patient_names(df_combined)
        df_raasifinal=df_raasifinal.dropna(subset=['MR No. / Vendor Code'])
        df_raasifinal.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/raasifinal.xlsx")
        print("selesai")

    if 0: # Kode untuk filter pasien minimal durasi pengobatan 3 bulan
        #filter_patient_3_months(input_file="/Users/lathifasyakira/Desktop/SKRIPSI/raasifinal.xlsx", output_file="/Users/lathifasyakira/Desktop/SKRIPSI/raasifinal3buland.xlsx")
        filter_patient_3_months_with_coverage(input_file=r"/Users/lathifasyakira/Desktop/SKRIPSI/raasifinal.xlsx", output_file=r"/Users/lathifasyakira/Desktop/SKRIPSI/raasifinal3bulan.xlsx")

        print("selesai")

    if 1: # Analisis gabungan RAASI & eGFR
        df_raasi = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/raasifinal3bulan.xlsx")
        df_egfr = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiegfrcombinedinterval.xlsx")

        df_long = get_raasi_egfr_longitudinal(df_raasi, df_egfr)

        freq, df_interval = visit_interval_after_merge_try(
            df_long,
            patient_col="Medical Record No.",
            date_col="Order Date"
        )
        
        #kalau mau dibikin list, harus diubah kode di bawah
        df_raasi_unique = df_raasi.drop_duplicates(
        subset="MR No. / Vendor Code"
        ).set_index("MR No. / Vendor Code")

        df_interval = df_interval.set_index("Medical Record No.")
        
        df_interval["delta_days_raasi"] = df_interval.index.map(
        df_raasi_unique["delta_days_raasi"]
        )

        df_interval["Golongan"] = df_interval.index.map(
        df_raasi_unique["Golongan"]
    )   
        df_long.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/raasiegfrlongfix.xlsx")
        df_interval.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/raasiegfrintervalfix.xlsx")

        print("selesai")
    
    if 0: #kode untuk visit freq dan billing
        df_combined = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/raasifinal.xlsx", index_col=0)
        intervals_df, freq_days, freq_weeks = patient_visit_frequency(df_combined,
                                patient_col="MR No. / Vendor Code", 
                                date_col="Created Date")

        print(df_combined.head())
        df_combined.info()

        intervals_df.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiraasiinterval.xlsx")

       # list_mr_intervals = intervals_df["Medical Record No."].tolist()
       # df_combined_interval = df_combined[df_combined["MR No. / Vendor Code"].isin(list_mr_intervals)]
       # df_combined_interval.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiraasicombinedinterval.xlsx")

        print("test")
        
    if 0: #no billing awal dan akhir
        df = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiraasiinterval.xlsx")
        df[["billing_awal", "billing_akhir"]] = df["billing"].apply(extract_billing)

        df.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiraasiinterval_with_billing_split.xlsx", index=False)

        print("selesai")

    if 0: # Analisis perbandingan file sekar dan syakira
        df1=pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiscraasifinal.xlsx")
        df2=pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiscraasifinal.xlsx")
        diff_df1 = df1.merge(df2, how='left', indicator=True).query('_merge == "left_only"').drop(columns='_merge')

        missing_rows = set(df2["MR No. / Vendor Code"]) - set(df1["MR No. / Vendor Code"])

        print("selesai")

    if 0: # Filter billing split berdasarkan data pasien egfr-raasi 3 bulan
        df_billing = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasiraasiinterval_with_billing_split.xlsx")
        df_ref = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/raasiegfrintervalfix.xlsx")
        
        col = "Medical Record No."  # sesuaikan kalau beda
        
        # Ambil daftar MR dari file referensi
        valid_mr = set(df_ref[col].dropna().unique())

        # Filter billing
        df_filtered = df_billing[df_billing[col].isin(list(valid_mr))]

        print("Total MR billing awal:", df_billing[col].nunique())
        print("Total MR referensi:", df_ref[col].nunique())
        print("Total MR setelah filter:", df_filtered[col].nunique())
        print("Baris sebelum:", len(df_billing))
        print("Baris setelah:", len(df_filtered))

        df_filtered.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/Daftar No Billing Pasien RAASi.xlsx", index=False)

        print("/Users/lathifasyakira/Desktop/SKRIPSI/kurasi_raasii_egfr_billing_split.xlsx")