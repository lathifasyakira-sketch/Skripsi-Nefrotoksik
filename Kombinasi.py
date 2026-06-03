import pandas as pd

def keep_common_mrn(
    file1,
    file2,
    output1="file1_commonMRN.xlsx",
    output2="file2_commonMRN.xlsx",
    output_overlap="MRN_overlap.xlsx",
    mrn_col="Medical Record No."
):
    """
    Menyisakan hanya MRN yang ada di kedua file.

    Returns
    -------
    df1_filtered
    df2_filtered
    overlap_df
    """

    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    df1[mrn_col] = df1[mrn_col].astype(str).str.strip()
    df2[mrn_col] = df2[mrn_col].astype(str).str.strip()

    common_mrn = (
        set(df1[mrn_col])
        &
        set(df2[mrn_col])
    )

    print(f"Jumlah MRN overlap: {len(common_mrn)}")

    df1_filtered = df1[
        df1[mrn_col].isin(common_mrn)
    ].copy()

    df2_filtered = df2[
        df2[mrn_col].isin(common_mrn)
    ].copy()

    overlap_df = pd.DataFrame(
        sorted(common_mrn),
        columns=[mrn_col]
    )

    df1_filtered.to_excel(
        output1,
        index=False
    )

    df2_filtered.to_excel(
        output2,
        index=False
    )

    overlap_df.to_excel(
        output_overlap,
        index=False
    )

    print(f"File tersimpan:")
    print(f" - {output1}")
    print(f" - {output2}")
    print(f" - {output_overlap}")

    return (
        df1_filtered,
        df2_filtered,
        overlap_df
    )

def add_combination_mrn(
    file_diuretik,
    file_raasi,
    file_target,
    output_file="target_updated.xlsx",
    mrn_col="Medical Record No."
):

    target_items = [
        "CO IRVELL 300/12.5 MG TABLET",
        "CO APROVEL 150 MG/12,5 MG TABLET"
    ]

    # BACA FILE

    df_diur = pd.read_excel(file_diuretik)
    df_raasi = pd.read_excel(file_raasi)
    df_target = pd.read_excel(file_target)

    # AMBIL MRN DARI DIURETIK

    mrn_diur = set(
        df_diur.loc[
            df_diur["Item Name"].astype(str).str.upper().isin(
                [x.upper() for x in target_items]
            ),
            "MR No. / Vendor Code"
        ].dropna().astype(str)
    )

    # AMBIL MRN DARI RAASI

    mrn_raasi = set(
        df_raasi.loc[
            df_raasi["Item Name"].astype(str).str.upper().isin(
                [x.upper() for x in target_items]
            ),
            "MR No. / Vendor Code"
        ].dropna().astype(str)
    )

    # GABUNGKAN

    mrn_obat = mrn_diur | mrn_raasi

    print("MRN pengguna obat:", len(mrn_obat))

    # CEK FILE TARGET

    existing_mrn = set(
        df_target[mrn_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    mrn_tambahan = sorted(
        mrn_obat - existing_mrn
    )

    print("MRN baru yang ditambahkan:", len(mrn_tambahan))

    # TAMBAHKAN

    if len(mrn_tambahan) > 0:

        df_new = pd.DataFrame({
            mrn_col: mrn_tambahan
        })

        df_target = pd.concat(
            [df_target, df_new],
            ignore_index=True
        )

    # SIMPAN

    df_target.to_excel(
        output_file,
        index=False
    )

    print(f"Hasil disimpan ke: {output_file}")

    return df_target

if __name__ == "__main__":
    if 0: #Menggabungkan RM Diuretik dan RAASi
        df_diur, df_raasi, overlap = keep_common_mrn(
            file1=r"D:\File_Syaki\SKRIPSINEW\RMDiuretikFIX.xlsx",
            file2=r"D:\File_Syaki\SKRIPSINEW\RMRAASi.xlsx",
            output1=r"D:\File_Syaki\SKRIPSINEW\RMDiuretik_common.xlsx",
            output2=r"D:\File_Syaki\SKRIPSINEW\RMRAASi_common.xlsx",
            output_overlap=r"D:\File_Syaki\SKRIPSINEW\kombinasi\MRN_overlap.xlsx"
            )
        
        print("selesai")

    if 1: #Menambah obat kombinasi
        add_combination_mrn(
        file_diuretik=r"D:\File_Syaki\SKRIPSINEW\diuretikfinal3bulan.xlsx",
        file_raasi=r"D:\File_Syaki\SKRIPSINEW\raasifinal3bulan.xlsx",
        file_target=r"D:\File_Syaki\SKRIPSINEW\kombinasi\MRN_overlap.xlsx",
        output_file=r"D:\File_Syaki\SKRIPSINEW\kombinasi\MRN_overlap_updated.xlsx"
        )

        print("selesai")