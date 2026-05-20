import glob, os, re
import pandas as pd
import matplotlib.pyplot as plt

def filter_excel_by_keyword(
    excel_files_pattern,
    list_sortir,
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
    pattern = "|".join(list_sortir)
    
    for file_path in excel_files:
        print(f"\nProcessing file: {file_path}")

        df = pd.read_excel(file_path)

        if column_name not in df.columns:
            print(f" Kolom '{column_name}' Tidak ditemukan. Skip file.")
            continue

        df_filtered = df[
            df[column_name]
            .astype(str)
            .str.contains(pattern, case=False, na=False, regex=True)
        ]

        if df_filtered.empty:
            print(f"Tidak ditemukan data '{list_sortir}'. Tidak ada file disimpan.")
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

        month = file_path.split(' ')[-2].replace('.xlsx', '')

        df['Bulan'] = month
        df['Tahun'] = year

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

    # Mengitung interval antar kunjungan per pasien
    for _, group in df.groupby(patient_col):
        dates = group[date_col].dropna().sort_values()

        if len(dates) < 2:
            continue

        deltas = dates.diff().dropna()
        intervals_days.append(deltas.dt.days.max())
        intervals_weeks.append(deltas.dt.days.max() / 7)

    intervals_df = pd.DataFrame({
        "interval_days": intervals_days,
        "interval_weeks": intervals_weeks
    })

    # Distribusi hari
    freq_days = intervals_df["interval_days"].value_counts().sort_index()

    # Bin mingguan 
    bins = [1, 2, 4, 8, 12, 24, 52, 1000]
    labels = [
        "1–2 minggu",
        "2–4 minggu",
        "1–2 bulan",
        "2–3 bulan",
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
    if 1: # Kode untuk Memfilter
        filter_excel_by_keyword(
        excel_files_pattern="/Users/lathifasyakira/Desktop/SKRIPSI/LabTest/LT *.xlsx",
        list_sortir=(
                 "Kreatinin Darah",
                 "Bersihan Kreatinin"
                 )  
        )
    
        print("selesai")

    if 0: # Kode untuk menggabungkan
        excel_file_template = "LT {Bulan}.xlsx"
        glob_pattern = f"/Users/lathifasyakira/Desktop/SKRIPSI/kurasikreatinin/{excel_file_template.replace('{Bulan}', '*')}"

        df_combined = load_and_combine_excel_data(glob_pattern)
        df_combined.to_excel("/Users/lathifasyakira/Desktop/SKRIPSI/kurasikreatinincombined.xlsx")

        patient_visit_frequency(df_combined)


        print(df_combined.head())
        df_combined.info()
