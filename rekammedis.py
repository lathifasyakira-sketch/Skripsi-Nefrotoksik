import pandas as pd
import re

def build_pattern(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    
    text = text.replace('\n', ' | ')
    
    items = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)|\|', text)
    
    items = [i.strip().strip('"') for i in items if i.strip()]
    
    return items

def split_diagnosis(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
        
    #tokens = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)|\||;|/', text)
    tokens = re.split(r'\||;|\n', text)
    
    tokens = [t.strip().strip('"') for t in tokens if t.strip()]

    if "obese" in text:
        print ("cape deh")
    
    return tokens

def match_patterns(text_series, patterns):
    patterns = [p.lower() for p in patterns]

    return text_series.apply(
        lambda x: any(
            any(p == token for p in patterns)
            for token in split_diagnosis(x)
        )
    )

def create_diagnosis_features(df_rm, df_dict, weight_map=None, mode='merge_weighted'):
    """
    mode:
    - 'merge_weighted'   → 1 kolom, isi weight (0–1)
    - 'separate_weighted'→ kolom terpisah per prefix
    
    weight_map contoh:
    {'sus':0.5, 'ec':0.6, 'dd':0.7}
    """
    
    if weight_map is None:
        weight_map = {'sus': 0.5, 'ec': 0.6, 'dd': 0.7}
    
    # PREPARE DATA
    df = df_rm[['Medical Record No.', 'billing_awal', 'billing_akhir', 'Diagnosa', 'Diagnosa.1']].copy()
    df.columns = ['MRN', 'billing_awal', 'billing_akhir', 'diag_awal', 'diag_akhir']
    
    df['diag_awal'] = df['diag_awal'].astype(str).str.lower()
    df['diag_akhir'] = df['diag_akhir'].astype(str).str.lower()
    
    # BUILD STRUCTURE
    grouped = {}
    
    for _, row in df_dict.iterrows():
        raw_key = str(row['Dict']).strip()
        raw_key = raw_key.replace('"','')
        key_lower = raw_key.lower()
        
        prefix_found = None
        for prefix in weight_map:
            if key_lower.startswith(prefix):
                prefix_found = prefix
                base_key = raw_key[len(prefix):]
                break
        
        if prefix_found is None:
            base_key = raw_key
            prefix_found = 'base'
            weight = 1.0
        else:
            weight = weight_map[prefix_found]
        
        base_key = base_key.strip()
        
 
            
        patterns = list(set(
            build_pattern(row['Diagnosa 1']) +
            build_pattern(row['Diagnosa 2'])
        ))
        
        if base_key not in grouped:
            grouped[base_key] = []
        
        grouped[base_key].append({
            'prefix': prefix_found,
            'patterns': [p for p in patterns if p],
            'weight': weight
        })
    
    # APPLY
    if mode == 'merge_weighted':
        for base_key, entries in grouped.items():
            df[base_key] = 0.0

            if base_key == "DMT2":
                print("lelah")

            for entry in entries:
                patterns = entry['patterns']
                weight = entry['weight']
                
                if not patterns:
                    continue
                
                
                match = (
                    match_patterns(df['diag_awal'], patterns) |
                    match_patterns(df['diag_akhir'], patterns)
                    )
                
                # ambil max weight
                df.loc[match, base_key] = df.loc[match, base_key].clip(lower=weight)
    
    
    elif mode == 'separate_weighted':
        
        for base_key, entries in grouped.items():
            
            for entry in entries:
                prefix = entry['prefix']
                patterns = entry['patterns']
                weight = entry['weight']
                
                if prefix == 'base':
                    col_name = base_key
                else:
                    col_name = prefix.capitalize() + base_key
                
                if not patterns:
                    df[col_name] = 0
                    continue
                
                match = (
                    match_patterns(df['diag_awal'], patterns) |
                    match_patterns(df['diag_akhir'], patterns)
                )
                
                # isi weight (bukan cuma 1)
                df[col_name] = match.astype(float) 
    
    else:
        raise ValueError("mode tidak valid")
    
    # CLEANUP
    df = df.drop(columns=['diag_awal', 'diag_akhir'])
    
    return df

# Tambah kolom usia
def add_usia_from_dob(
    file_merge,
    file_ref,
    output_file=None,
    col_merge_mrn="MRN",
    col_ref_mrn="MR No. / Vendor Code",
    col_dob="Date of Birth",
    ref_date="2025-12-31",
    drop_dob=False
):
    df_merge = pd.read_excel(file_merge)
    df_ref = pd.read_excel(file_ref)

    df_ref = df_ref.rename(columns={
        col_ref_mrn: col_merge_mrn,
        col_dob: "DOB"
    })

    df_merge[col_merge_mrn] = df_merge[col_merge_mrn].astype(str)
    df_ref[col_merge_mrn] = df_ref[col_merge_mrn].astype(str)

    df_ref["DOB"] = pd.to_datetime(df_ref["DOB"], errors="raise", dayfirst=True)

    df_ref['is_valid_dob'] = df_ref['DOB'].notna()

    df_ref = (
        df_ref
        .sort_values(by=[col_merge_mrn, 'is_valid_dob'], ascending=[True, False])
        .drop_duplicates(subset=col_merge_mrn, keep='first')
        .drop(columns='is_valid_dob')
        )
    df = df_merge.merge(
        df_ref[[col_merge_mrn, "DOB"]],
        on=col_merge_mrn,
        how="left"
    )

    ref_date = pd.to_datetime(ref_date)

    df["usia_tahun"] = ref_date.year - df["DOB"].dt.year
    df["usia_tahun"] -= (
        (ref_date.month < df["DOB"].dt.month) |
        ((ref_date.month == df["DOB"].dt.month) & (ref_date.day < df["DOB"].dt.day))
    )

    total_bulan = (
        (ref_date.year - df["DOB"].dt.year) * 12 +
        (ref_date.month - df["DOB"].dt.month)
    )

    # kalau tanggal belum lewat → kurang 1 bulan
    kurang_bulan = (ref_date.day < df["DOB"].dt.day).astype(int)
    total_bulan = total_bulan - kurang_bulan

    df["usia_bulan"] = total_bulan % 12
    df.fillna(0)
    #df["usia"] = (
        #df["usia_tahun"].astype("Int64")
    #)

    if drop_dob:
        df = df.drop(columns=["DOB"])

    if output_file:
        df.to_excel(output_file, index=False)

    return df

if __name__ == "__main__":
    if 0: 
        df_rm = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/variabelX/RMdiuretik.xlsx")
        df_dict = pd.read_excel("/Users/lathifasyakira/Desktop/SKRIPSI/Diagnosis_Diuretik.xlsx")


        cols = ['Diagnosa', 'Diagnosa.1']

        df_rm[cols] = df_rm[cols].replace(r'[\r\n]+', '|', regex=True)
        df_dict = df_dict.replace(r'[\r\n]+', '|', regex=True)
    
        weight_config = {'sus': 0.5, 'ec': 0.6, 'dd': 0.7}

        # 1. DIGABUNG (1 kolom per penyakit)
        df_merge = create_diagnosis_features(df_rm, df_dict, weight_map=weight_config, mode='merge_weighted')

        # 2. DIPISAH (multi kolom)
        df_sep = create_diagnosis_features(df_rm, df_dict, weight_map=weight_config, mode='separate_weighted')

        df_merge.to_excel('merge_s05e06d07.xlsx', index=False)
        df_sep.to_excel('separate.xlsx', index=False)   

        print ("selesai")  

    if 1: #Tambah kolom usia
        df = add_usia_from_dob(
            file_merge="/Users/lathifasyakira/Desktop/SKRIPSI/merge_s05e06d07.xlsx",
            file_ref="/Users/lathifasyakira/Desktop/SKRIPSI/variabelX/diuretikfinal3bulan.xlsx",
            output_file="/Users/lathifasyakira/Desktop/SKRIPSI/variabelX/merge_with_usia.xlsx"             
            )
        
        print ("selesai")         
                    