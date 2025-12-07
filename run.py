"""
Główny skrypt uruchomieniowy.
Łączy ETL (Excel->CSV) z Engine (CSV->JSON).
"""
import sys
import os

sys.path.append(os.getcwd())

from backend.data_aggregator import IABEngine
import import_gus_data

def main():
    print("🚀 START: System Oceny Ryzyka IAB (GUS F-01)")
    print("-" * 50)
    
    # 1. ETL: Przetwarzanie Excela
    try:
        import_gus_data.process_raw_files()
    except Exception as e:
        print(f"❌ BŁĄD ETL: {e}")
        # Jeśli nie masz openpyxl, skrypt się tu zatrzyma z jasnym komunikatem
        print("   Wskazówka: Upewnij się, że masz zainstalowane: pip install pandas openpyxl")
        return

    # 2. ENGINE: Obliczenia
    print("-" * 50)
    engine = IABEngine()
    engine.init_storage()
    
    engine.fetch_bankruptcy_data() 
    engine.fetch_gus_clean_data()  # Czyta to, co wygenerował krok 1
    engine.fetch_market_data()     
    
    engine.calculate_and_save()
    print("\n🏁 KONIEC. Możesz uruchomić serwer: python -m http.server")

if __name__ == "__main__":
    main()