"""
Moduł ETL: Parser Excela GUS (F-01) - Wersja Dedykowana
Dostosowana do struktury: Nagłówki kodów w wierszu 13, dane od 14.
"""

import pandas as pd
import re
from pathlib import Path

# Konfiguracja ścieżek
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / 'data'
OUTPUT_FILE = DATA_DIR / 'gus_clean.csv'

def find_gus_excel():
    for file in DATA_DIR.glob('*.xlsx'):
        if not file.name.startswith('~$'):
            return file
    return None

def clean_number(val):
    """Konwersja '1 234,56' -> 1234.56"""
    if pd.isna(val) or val == '' or str(val).strip() in ['.', '-', '#']:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).replace(' ', '').replace('\xa0', '').replace(',', '.')
    try:
        return float(val_str)
    except:
        return 0.0

def process_raw_files():
    print("🕵️  ETL START: Pobieranie danych z Tablicy 2...")
    
    excel_file = find_gus_excel()
    if not excel_file:
        print(f"❌ BŁĄD: Brak pliku .xlsx w {DATA_DIR}")
        return

    print(f"   Plik: {excel_file.name}")
    
    try:
        # Wczytujemy arkusz "Tablica 2." bez nagłówków, żeby operować na indeksach
        df = pd.read_excel(excel_file, sheet_name="Tablica 2.", header=None)
    except Exception as e:
        print(f"❌ BŁĄD: Nie można wczytać arkusza 'Tablica 2.'. {e}")
        return

    # --- KONFIGURACJA KOLUMN (Na podstawie Twojej analizy) ---
    # Wiersz z danymi zaczyna się od indeksu 13 (czyli 14. wiersz w Excelu)
    START_ROW = 13 
    
    # Indeksy kolumn (A=0, B=1, ...):
    COL_PKD = 1       # Kolumna B: Dział_PKD (np. "01")
    COL_FIRMY = 6     # Kolumna G: Liczba przedsiębiorstw
    COL_PRZYCHODY = 11 # Kolumna L: Przychody netto...
    COL_ROS = 193     # Kolumna GH: Wskaźnik rentowności obrotu netto
    
    clean_rows = []
    print(f"   Rozpoczynam ekstrakcję od wiersza {START_ROW}...")

    for i in range(START_ROW, len(df)):
        row = df.iloc[i]
        
        # Pobieramy kod PKD
        pkd_val = str(row[COL_PKD]).strip()
        
        # Filtracja: Interesują nas tylko kody 2-cyfrowe (Działy), pomijamy sekcje (litery) i grupy (3 cyfry)
        # Regex: Musi być dokładnie 2 cyfry
        if re.match(r'^\d{2}$', pkd_val):
            
            # Pomijamy sumy działów (często mają '00' w kodzie grupy, ale my chcemy wiersze z działami)
            # W tym pliku: Kol 1 to Dział. Jeśli jest wypełniony i ma 2 cyfry, to bierzemy.
            # Wyjątek: '00' to zazwyczaj suma sekcji, można pominąć lub zostawić.
            if pkd_val == '00': continue

            firmy = int(clean_number(row[COL_FIRMY]))
            przychody = clean_number(row[COL_PRZYCHODY])
            ros = clean_number(row[COL_ROS])
            
            # Zabezpieczenie przed pustymi wierszami
            if firmy == 0 and przychody == 0: continue

            clean_rows.append({
                'pkd': pkd_val,
                'firmy': firmy,
                'przychody': przychody,
                'ros': ros,
                'wynik_netto': 0 # Nieistotne dla silnika, jeśli mamy ROS
            })

    if not clean_rows:
        print("❌ BŁĄD: Nie wyekstrahowano danych. Sprawdź indeksy kolumn.")
        return

    # Zapis
    df_out = pd.DataFrame(clean_rows)
    
    # Usuwamy duplikaty (czasem ten sam dział występuje w różnych grupach, bierzemy unikalne po kodzie)
    # W strukturze GUS wiersz z Działem jest nadrzędny. 
    # Jeśli są powtórzenia, weźmiemy pierwszy (nadrzędny) lub zsumujemy.
    # Tutaj zakładam, że wiersz "01" występuje raz jako podsumowanie działu.
    df_out = df_out.drop_duplicates(subset=['pkd'])
    
    df_out.to_csv(OUTPUT_FILE, sep=';', index=False)
    
    print(f"✅ SUKCES: Wygenerowano 'gus_clean.csv'")
    print(f"   -> Liczba branż: {len(df_out)}")
    print(f"   -> Przykładowy wiersz: PKD {df_out.iloc[0]['pkd']} | Firmy: {df_out.iloc[0]['firmy']} | ROS: {df_out.iloc[0]['ros']}%")

if __name__ == "__main__":
    process_raw_files()