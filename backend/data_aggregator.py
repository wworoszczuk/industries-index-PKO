import pandas as pd
import yfinance as yf
import numpy as np
import csv
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

BACKEND_DIR = Path(__file__).parent.resolve()
ROOT_DIR = BACKEND_DIR.parent
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'output'

# --- PROFILE RYZYKA ---
AI_PROFILES = {
    'CREATOR': ['26', '62', '63', '72'],
    'SUPPORTED': [
        '10', '11', '20', '21', '27', '28', '29', '30', 
        '64', '65', '66', '69', '70', '71', '73', '74', '86'
    ],
    'THREATENED': [
        '45', '46', '47', '49', '50', '51', '52', '53', '55', '56', '77', '78', '80', '81', '82'
    ]
}

MACRO_RISKS = {
    'CO2_INTENSIVE': ['05', '06', '07', '08', '09', '19', '20', '23', '24', '35', '36', '37'],
    'ESG_NEGATIVE': ['12', '92'],
    'INTEREST_SENSITIVE': ['41', '42', '43', '64', '65', '66', '68'],
    'COMMODITY_EXPOSED': ['01', '02', '03']
}

ETF_MAP = {
    '01': 'DBA', '02': 'DBA', '05': 'XME', '06': 'XLE', '07': 'XME',
    '10': 'XLP', '11': 'XLP', '19': 'XLE', '20': 'XLB', '24': 'XME', 
    '26': 'SOXX', '27': 'XLI', '28': 'XLI', '29': 'XLI', '35': 'XLU', 
    '41': 'ITB', '42': 'ITB', '43': 'ITB', '45': 'XRT', '46': 'XRT', 
    '47': 'XRT', '51': 'JETS', '62': 'IGV', '63': '^IXIC', '64': 'XLF', 
    '65': 'KIE', '66': 'XLF', '68': 'XLRE', '86': 'XLV'
}

class IABEngine:
    def __init__(self):
        self.data_gus = {}
        self.data_failures = {}
        self.meta = {}
        self.macro_current = {}
        self.macro_forecast = {}
        self.etf_trends = {}
        
    def init_storage(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        pkd_file = DATA_DIR / 'PKD.csv'
        if pkd_file.exists():
            with open(pkd_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    code = str(row.get('Division Code', '')).strip().zfill(2)
                    self.meta[code] = {'nazwa': row.get('Division Name'), 'sekcja': row.get('Section Code')}
        else: self.meta = {}

    def fetch_bankruptcy_data(self):
        csv_path = DATA_DIR / 'upadlosci.csv'
        if not csv_path.exists(): return
        try:
            df = pd.read_csv(csv_path, sep=';', dtype={'pkd': str})
            df.columns = [c.lower() for c in df.columns]
            max_year = df['rok'].max()
            df = df[df['rok'] == max_year].copy()
            df['dzial'] = df['pkd'].str[:2].str.zfill(2)
            self.data_failures = df.groupby('dzial')['liczba_upadlosci'].sum().to_dict()
            print(f"✅ Upadłości: Załadowano {len(self.data_failures)} branż.")
        except: pass

    def fetch_gus_clean_data(self):
        csv_path = DATA_DIR / 'gus_clean.csv'
        if not csv_path.exists(): return
        try:
            df = pd.read_csv(csv_path, sep=';', dtype={'pkd': str})
            for _, row in df.iterrows():
                pkd = str(row['pkd']).zfill(2)
                self.data_gus[pkd] = {
                    'firmy': int(row['firmy']),
                    'przychody': float(row['przychody']),
                    'ros': float(row['ros'])
                }
            print(f"✅ Dane finansowe: Załadowano {len(self.data_gus)} branż.")
        except: pass

    def fetch_market_data(self):
        print("🌍 Pobieranie danych rynkowych...")
        tickers = {'AI_TREND': 'BOTZ', 'CO2_PRICE': 'KRBN', 'RATES': '^TNX'}
        all_tickers = list(tickers.values()) + list(set(ETF_MAP.values()))
        
        try:
            data = yf.download(all_tickers, period="6mo", progress=False)['Close']
            if data.empty: raise Exception("Brak danych")

            for key, ticker in tickers.items():
                if ticker in data.columns:
                    s = data[ticker].dropna()
                    if not s.empty:
                        change = (s.iloc[-1] - s.iloc[0]) / s.iloc[0]
                        self.macro_current[key] = change
                        self.macro_forecast[key] = change * 1.5 # Projekcja trendu
                    else:
                        self.macro_current[key] = 0.0
                        self.macro_forecast[key] = 0.0
            
            for code, ticker in ETF_MAP.items():
                if ticker in data.columns:
                    s = data[ticker].dropna()
                    self.etf_trends[ticker] = (s.iloc[-1] - s.iloc[0]) / s.iloc[0] if not s.empty else 0.0

            print(f"✅ Rynek Live: AI={self.macro_current.get('AI_TREND',0):.1%}")

        except Exception as e:
            print(f"⚠️ Yahoo Finance niedostępne ({e}). Używam SYMULACJI.")
            self.macro_current = {'AI_TREND': 0.15, 'CO2_PRICE': 0.05, 'RATES': -0.05}
            self.macro_forecast = {'AI_TREND': 0.25, 'CO2_PRICE': 0.10, 'RATES': -0.08}
            self.etf_trends = {k: 0.0 for k in ETF_MAP.values()}

    def _calculate_components_and_score(self, pkd, gus, failures, macro):
        """Oblicza wynik metodą średniej ważonej (0-100)"""
        
        # --- 1. WAGI ---
        W = {
            'SP': 0.30, # Rentowność
            'SH': 0.25, # Zdrowie
            'SR': 0.15, # Odporność
            'RR': 0.10, # Ryzyko (Waga ujemna we wzorze)
            'AI': 0.10, # Innowacje
            'MC': 0.05, # Makro
            'GT': 0.05  # Global
        }

        # --- 2. OBLICZANIE KOMPONENTÓW (Skala 0-100) ---
        
        # SP (Rentowność) - Skalowanie: -5% -> 0 pkt, 15% -> 100 pkt
        ros = gus['ros']
        score_sp = 50 + (ros * 3.33) 
        score_sp = max(0, min(100, score_sp))
        
        # SH (Zdrowie) - Upadłości na 1000 firm
        fail_ratio = (failures / gus['firmy']) * 1000
        # 0 upadłości = 100 pkt, 20 upadłości = 0 pkt
        score_sh = max(0, min(100, 100 - (fail_ratio * 5)))
        
        # RR (Ryzyko) - To samo co SH tylko odwrotnie (wysokie RR = wysokie ryzyko)
        # 20 upadłości = 100 pkt (Full Risk), 0 upadłości = 0 pkt
        score_rr = min(100, fail_ratio * 5)
        
        # SR (Odporność) - Proxy: Zysk + Stabilność
        score_sr = (score_sp * 0.6) + (score_sh * 0.4)
        
        # AI (Innowacje) - Baza 50 + Trend
        ai_trend = macro.get('AI_TREND', 0)
        ai_base = 50
        if pkd in AI_PROFILES['CREATOR']: ai_base = 85
        elif pkd in AI_PROFILES['SUPPORTED']: ai_base = 65
        elif pkd in AI_PROFILES['THREATENED']: ai_base = 35
        
        # Trend wpływa na bazę (+/- 20 pkt)
        score_ai = ai_base * (1.0 + ai_trend)
        score_ai = max(0, min(100, score_ai))
        
        # MC (Makro) - Baza 100 - Kary
        mc_penalty = 0
        co2 = macro.get('CO2_PRICE', 0)
        rates = macro.get('RATES', 0)
        
        if pkd in MACRO_RISKS['CO2_INTENSIVE'] and co2 > 0: mc_penalty += (co2 * 100 + 10)
        if pkd in MACRO_RISKS['INTEREST_SENSITIVE'] and rates > 0: mc_penalty += (rates * 100 + 10)
        if pkd in MACRO_RISKS['ESG_NEGATIVE']: mc_penalty += 30
        
        score_mc = max(0, 100 - mc_penalty)
        
        # GT (Global)
        etf = ETF_MAP.get(pkd)
        trend = self.etf_trends.get(etf, 0)
        score_gt = 50 + (trend * 100)
        score_gt = max(0, min(100, score_gt))

        # --- 3. WZÓR GŁÓWNY ---
        # IAB = Suma(Waga * Wynik) - (WagaRR * WynikRR)
        
        weighted_sum = (
            score_sp * W['SP'] +
            score_sh * W['SH'] +
            score_sr * W['SR'] +
            score_ai * W['AI'] +
            score_mc * W['MC'] +
            score_gt * W['GT']
        )
        
        risk_penalty = score_rr * W['RR']
        
        final_score = weighted_sum - risk_penalty
        
        # Normalizacja do skali 0-100 (teoretyczne max to 90, min -10)
        final_score += 5 
        final_score = max(0, min(100, final_score))
        
        return final_score, {
            'SH': round(score_sh,1), 'SP': round(score_sp,1), 'SR': round(score_sr,1),
            'RR': round(score_rr,1), 'AI': round(score_ai,1), 'MC': round(score_mc,1), 'GT': round(score_gt,1)
        }

    def calculate_and_save(self):
        print("\n🧮 Obliczanie IAB (Weighted Methodology)...")
        valid_pkds = set(self.data_gus.keys())
        results = []
        
        for pkd in valid_pkds:
            info = self.meta.get(pkd, {'nazwa': f'Dział {pkd}', 'sekcja': '?'})
            gus = self.data_gus[pkd]
            failures = self.data_failures.get(pkd, 0)
            
            if gus['firmy'] == 0: continue
            
            # Obliczenia
            score_now, comps = self._calculate_components_and_score(pkd, gus, failures, self.macro_current)
            score_future, _ = self._calculate_components_and_score(pkd, gus, failures, self.macro_forecast)
            
            # Outlook
            delta = score_future - score_now
            if delta >= 0.5: outlook = "Pozytywna ↗"
            elif delta <= -0.5: outlook = "Negatywna ↘"
            else: outlook = "Stabilna ➡"
            
            if score_now < 40: risk_cat = 'Wysokie'
            elif score_now > 70: risk_cat = 'Niskie'
            else: risk_cat = 'Średnie'

            results.append({
                'pkd': pkd, 'nazwa': info['nazwa'], 'sekcja': info['sekcja'],
                'iab': round(score_now, 1),
                'iab_forecast': round(score_future, 1),
                'risk': risk_cat,
                'outlook': outlook,
                'firmy': gus['firmy'], 'upadlosci': failures, 'finanse_ros': round(gus['ros'], 2),
                'components': comps, 'data_quality': 'FINAL_WEIGHTED'
            })

        results.sort(key=lambda x: x['iab'], reverse=True)
        
        with open(OUTPUT_DIR / 'iab_data.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Raport wygenerowany: {OUTPUT_DIR / 'iab_data.json'}")