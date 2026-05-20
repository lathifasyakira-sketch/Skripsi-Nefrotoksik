import pandas as pd
import os

def normalize_mr(x):
    if pd.isna(x):
        return None
    
    x = str(x).strip()
    #handle float
    if x.endswith(".0"):
        x = x[:-2]

    # hapus leading zero
    x = x.lstrip("0")

    return x if x != "" else "0"


def extract_and_merge_mr(
    file_master,
    folder_path,
    output_path,
    col_master="Medical Record No.",
    col_target="MR No. / Vendor Code",
    verbose=True
):

    df_master = pd.read_excel(file_master)

    
    df_master["MR_norm"] = df_master[col_master].apply(normalize_mr)

    mr_set = set(df_master["MR_norm"].dropna())

    if verbose:
        print(f"Total MR ditemukan (normalized): {len(mr_set)}")

    df_list = []

    for file in os.listdir(folder_path):
        if not file.endswith(".xlsx"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            df = pd.read_excel(file_path)

            if col_target not in df.columns:
                if verbose:
                    print(f"Skip (kolom tidak ada): {file}")
                continue

            df["MR_norm"] = df[col_target].apply(normalize_mr)

            df_filtered = df[df["MR_norm"].isin(mr_set)]

            if not df_filtered.empty:
                df_filtered["source_file"] = file
                df_list.append(df_filtered)

                if verbose:
                    print(f"✔️ {file} -> {len(df_filtered)} rows")

        except Exception as e:
            print(f"Error di {file}: {e}")

    if df_list:
        df_final = pd.concat(df_list, ignore_index=True)
    else:
        df_final = pd.DataFrame()

    # optional: hapus kolom helper
    if "MR_norm" in df_final.columns:
        df_final = df_final.drop(columns=["MR_norm"])

    df_final.to_excel(output_path, index=False)

    if verbose:
        print(f"\nSelesai! Total rows: {len(df_final)}")
        print(f"Disimpan di: {output_path}")

    return df_final

def extract_unique_items(
    input_path,
    output_path,
    col_item_code="Item Code",
    col_item_name="Item Name"
):
    df = pd.read_excel(input_path)

    df_filtered = df[~df[col_item_code].astype(str).str.startswith("B")]

    unique_items = (
        df_filtered[col_item_name]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    df_unique = pd.DataFrame({col_item_name: unique_items})

    df_unique.to_excel(output_path, index=False)

    print(f"Selesai! Total unique item: {len(df_unique)}")
    print(f"Disimpan di: {output_path}")

    return df_unique

import pandas as pd

def hitung_nefrotoksik(
    df_konsumsi,
    df_nefro,
    col_mr='MR No. / Vendor Code',
    col_item='Item Name',
    col_nefro_1=1,
    col_nefro_05='0,5'
):
    """
    Menghasilkan dataframe berisi MR No dan status nefrotoksik (0, 0.5, 1)
    """

    col_1 = df_nefro[[col_nefro_1]].dropna().rename(columns={col_nefro_1: col_item})
    col_1['nefro_score'] = 1

    col_05 = df_nefro[[col_nefro_05]].dropna().rename(columns={col_nefro_05: col_item})
    col_05['nefro_score'] = 0.5

    df_nefro_long = pd.concat([col_1, col_05], ignore_index=True)

    df_nefro_long[col_item] = (
        df_nefro_long[col_item]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df_konsumsi[col_item] = (
        df_konsumsi[col_item]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df_merge = df_konsumsi.merge(
        df_nefro_long,
        on=col_item,
        how='left'
    )

    result = (
        df_merge
        .groupby(col_mr)['nefro_score']
        .max()
        .reset_index()
    )

    result['nefro_score'] = result['nefro_score'].fillna(0)

    result = result.rename(columns={'nefro_score': 'nefrotoksik'})

    return result

def hitung_nefrotoksik_akumulasi(
    df_konsumsi,
    df_nefro,
    col_mr='MR No. / Vendor Code',
    col_item='Item Name',
    col_nefro_1=1,
    col_nefro_05='0,5'
):
    """
    Menghasilkan dataframe berisi MR No dan skor nefrotoksik (akumulasi)
    contoh:
    - 2 obat skor 1 + 1 obat skor 0.5 → total 2.5
    """

    col_1 = df_nefro[[col_nefro_1]].dropna().rename(columns={col_nefro_1: col_item})
    col_1['nefro_score'] = 1

    col_05 = df_nefro[[col_nefro_05]].dropna().rename(columns={col_nefro_05: col_item})
    col_05['nefro_score'] = 0.5

    df_nefro_long = pd.concat([col_1, col_05], ignore_index=True)

    df_nefro_long[col_item] = (
        df_nefro_long[col_item]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df_konsumsi[col_item] = (
        df_konsumsi[col_item]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df_merge = df_konsumsi.merge(
        df_nefro_long,
        on=col_item,
        how='left'
    )

    result = (
        df_merge
        .groupby(col_mr)['nefro_score']
        .sum()
        .reset_index()
    )

    result['nefro_score'] = result['nefro_score'].fillna(0)

    result = result.rename(columns={'nefro_score': 'nefrotoksik_akumulasi'})

    return result

def hitung_nefrotoksik_akumulasi_unique(
    df_konsumsi,
    df_nefro,
    col_mr='MR No. / Vendor Code',
    col_item='Item Name',
    col_nefro_1=1,
    col_nefro_05='0,5'
):
    """
    Menghasilkan skor nefrotoksik berbasis UNIQUE OBAT per pasien
    contoh:
    - pasien minum obat A (1) 10x → tetap dihitung 1
    - pasien minum A (1) + B (1) + C (0.5) → total 2.5
    """

    col_1 = df_nefro[[col_nefro_1]].dropna().rename(columns={col_nefro_1: col_item})
    col_1['nefro_score'] = 1

    col_05 = df_nefro[[col_nefro_05]].dropna().rename(columns={col_nefro_05: col_item})
    col_05['nefro_score'] = 0.5

    df_nefro_long = pd.concat([col_1, col_05], ignore_index=True)

    df_nefro_long[col_item] = (
        df_nefro_long[col_item]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df_konsumsi[col_item] = (
        df_konsumsi[col_item]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df_merge = df_konsumsi.merge(
        df_nefro_long,
        on=col_item,
        how='left'
    )

    df_unique = df_merge.drop_duplicates(subset=[col_mr, col_item])

    result = (
        df_unique
        .groupby(col_mr)['nefro_score']
        .sum()
        .reset_index()
    )

    result['nefro_score'] = result['nefro_score'].fillna(0)

    result = result.rename(columns={'nefro_score': 'nefrotoksik_akumulasi'})

    return result

if __name__ == "__main__":
    if 0: #Menggabungkan SC berdasarkan MR diagnosis
        df_hasil = extract_and_merge_mr(
        file_master=r"/Users/lathifasyakira/Desktop/SKRIPSI/RMDiuretik.xlsx",
        folder_path=r"/Users/lathifasyakira/Desktop/SKRIPSI/StockCard",
        output_path=r"/Users/lathifasyakira/Desktop/SKRIPSI/konsumsiobatpasiendiuretik.xlsx"
        )
        df_hasil = df_hasil.drop_duplicates(subset=["Item Name"])

        print ("bismillah")

    if 1: #Menghilangkan BMHP dan menyatukan obat
        df_items = extract_unique_items(
        input_path=r"/Users/lathifasyakira/Desktop/SKRIPSI/konsumsiobatpasiendiuretik.xlsx",
        output_path=r"/Users/lathifasyakira/Desktop/SKRIPSI/konsumsiobatpasiendiuretik_filtered.xlsx"
        )

        print ("bismillah")

    if 0: #Agregasi maksimalisasi
        df_konsumsi = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/konsumsiobatpasiendiuretik.xlsx")
        df_nefro = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/Nefrotoksik.xlsx")

        df_result = hitung_nefrotoksik(df_konsumsi, df_nefro)

        print(df_result.head())

    if 0: #Agregasi akumulasi
        df_konsumsi = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/konsumsiobatpasiendiuretik.xlsx")
        df_nefro = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/Nefrotoksik.xlsx")
        
        df_result_sum = hitung_nefrotoksik_akumulasi_unique(df_konsumsi, df_nefro)
        
        print(df_result_sum.head())