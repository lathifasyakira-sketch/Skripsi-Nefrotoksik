import glob, os, re
import pandas as pd
import matplotlib.pyplot as plt


def filter_excel_by_keyword(
    excel_files_pattern,
    filter_keyword,
    column_name="Test Name",
    output_prefix="LT",
    output_suffix="Kreatinin"
):
    """
    Membaca file Excel berdasarkan pola, memfilter baris berdasarkan keyword,
    dan menyimpan hasilnya ke file Excel baru.

    Parameters
    ----------
    excel_files_pattern : str
        Pola path file Excel (contoh: "/content/LT *.xlsx").
    filter_keyword : str
        Keyword yang dicari dalam kolom (case-insensitive).
    column_name : str
        Nama kolom tempat keyword dicari (default: "Item Name").
    output_prefix : str
        Awalan nama file output (default: "LT").
    output_suffix : str
        Akhiran nama file output (default: "Kreatinin").

    Returns
    -------
    None
    """

    excel_files = glob.glob(excel_files_pattern)
    print(f"Found {len(excel_files)} Excel files matching the pattern '{excel_files_pattern}'.")

    for file_path in excel_files:
        print(f"\nProcessing file: {file_path}")

        df = pd.read_excel(file_path)

        if column_name not in df.columns:
            print(f" Kolom '{column_name}' Tidak ditemukan. Skip file.")
            continue

        df_filtered = df[
            df[column_name]
            .astype(str)
            .str.contains(filter_keyword, case=True, na=False)
        ]

        if df_filtered.empty:
            print(f"Tidak ditemukan data '{filter_keyword}'. Tidak ada file disimpan.")
            continue

        # Ambil nama bulan / identifier dari nama file
        match = re.search(r'LT\s(.+?)\.xlsx', os.path.basename(file_path))
        if match:
            identifier = match.group(1).strip()
        else:
            identifier = os.path.splitext(os.path.basename(file_path))[0]

        new_file_name = f"{output_prefix} {identifier} {output_suffix}.xlsx"

        df_filtered.to_excel(new_file_name, index=False)
        print(f" Menyimpan {len(df_filtered)} baris ke '{new_file_name}'")

    print("\nProses selesai.")

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
    for _, group in df.groupby(patient_col):
        dates = group[date_col].dropna().sort_values()

        if len(dates) < 2:
            continue
        
        deltas = dates.diff().dropna()
        intervals_week = (deltas.dt.days.max() / 7)

        if intervals_week > 12:
            intervals_days.append(deltas.dt.days.max())
            intervals_weeks.append(deltas.dt.days.max() / 7)
            billing_no.append(list(group["Billing No."]))
            med_rec.append(group["Medical Record No."].values[0])


    intervals_df = pd.DataFrame({
        "interval_days": intervals_days,
        "interval_weeks": intervals_weeks,
        "billing": billing_no,
        "Medical Record No.": med_rec

    })

    intervals_df = intervals_df[intervals_df["interval_days"] >= 90]

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


if __name__ == "__main__":
    if 0: # Kode untuk Memfilter
        filter_excel_by_keyword(
        excel_files_pattern="/Users/lathifasyakira/Desktop/SKRIPSI/LabTest/LT *.xlsx",
        list_sortir=(
                    "Kreatinin Darah",
                    "Bersihan Kreatinin"
                    )
        )
    
        print("selesai")

    if 1: # Kode untuk menggabungkan
        excel_file_template = "LT {Bulan} kreatinin.xlsx"
        glob_pattern = f"/Users/lathifasyakira/Desktop/SKRIPSI/kurasikreatinin/{excel_file_template.replace('{Bulan}', '*')}"
        df_combined = load_and_combine_excel_data(glob_pattern)
        df_combined.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasikreatinincombined.xlsx")

        intervals_df, freq_days, freq_weeks= patient_visit_frequency(df_combined)


        print(df_combined.head())
        intervals_df.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasikreatinininterval.xlsx")
        list_mr_intervals = intervals_df["Medical Record No."].tolist()
        df_combined_interval = df_combined[df_combined["Medical Record No."].isin(list_mr_intervals)]
        df_combined_interval.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasikreatinincombinedinterval.xlsx")
        
        df_combined.info()

        print ("selesai kreatinin")
