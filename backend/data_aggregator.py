import pandas as pd
import yfinance as yf
import numpy as np
import csv
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

W = {'SP': 0.30, 'SH': 0.25, 'SR': 0.15, 'RR': 0.10, 'AI': 0.10, 'MC': 0.05, 'GT': 0.05}

BACKEND_DIR = Path(__file__).parent.resolve()
ROOT_DIR = BACKEND_DIR.parent
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'output'

AI_PROFILES = {
    'CREATOR': ['26', '62', '63', '72'],
    'SUPPORTED': ['10', '11', '20', '21', '27', '28', '29', '30', '64', '65', '66', '69', '70', '71', '73', '74', '86'],
    'THREATENED': ['45', '46', '47', '49', '50', '51', '52', '53', '55', '56', '77', '78', '80', '81', '82']
}

MACRO_RISKS = {
    'CO2_INTENSIVE': ['05', '06', '07', '08', '09', '19', '20', '23', '24', '35', '36', '37'],
    'ENERGY_SENSITIVE': ['10', '20', '22', '25', '29', '30'],
    'INTEREST_SENSITIVE': ['41', '42', '43', '64', '65', '66', '68'],
    'COMMODITY_EXPOSED': ['01', '02', '10', '13', '14', '15', '16', '17']
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
        self.data_gus = {}; self.data_failures = {}; self.meta = {}
        self.macro_current = {}; self.macro_forecast = {}; self.etf_trends = {}
        
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
                self.data_gus[pkd] = {'firmy': int(row['firmy']), 'przychody': float(row['przychody']), 'ros': float(row['ros'])}
            print(f"✅ Dane finansowe: Załadowano {len(self.data_gus)} branż.")
        except: pass

    def calculate_trend_projection(self, series, multiplier=1.5):
        if series.empty or series.iloc[0] == 0: return 0.0, 0.0
        change = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
        return change, change * multiplier

    def fetch_market_data(self):
        print("🌍 Pobieranie danych rynkowych...")
        tickers = {'AI_TREND': 'BOTZ', 'CO2_PRICE': 'KRBN', 'RATES': '^TNX', 'OIL_PRICE': 'CL=F', 'GAS_PRICE': 'NG=F'}
        all_tickers = list(tickers.values()) + list(set(ETF_MAP.values()))
        
        try:
            data = yf.download(all_tickers, period="6mo", progress=False)['Close']
            if data.empty: raise Exception("Brak danych")

            for key, ticker in tickers.items():
                if ticker in data.columns:
                    s = data[ticker].dropna()
                    if not s.empty:
                        curr, fore = self.calculate_trend_projection(s)
                        self.macro_current[key] = curr
                        self.macro_forecast[key] = fore
                    else:
                        self.macro_current[key] = 0.0
                        self.macro_forecast[key] = 0.0
            
            for code, ticker in ETF_MAP.items():
                if ticker in data.columns:
                    s = data[ticker].dropna()
                    self.etf_trends[ticker] = (s.iloc[-1] - s.iloc[0]) / s.iloc[0] if not s.empty else 0.0

            print(f"✅ Rynek Live: AI={self.macro_current.get('AI_TREND',0):.2%}, Stopy={self.macro_current.get('RATES',0):.2%}")

        except Exception as e:
            print(f"⚠️ Yahoo Finance Error: {e}. Używam SYMULACJI (Demo Mode).")
            self.macro_current = {'AI_TREND': 0.15, 'CO2_PRICE': 0.05, 'RATES': -0.04, 'OIL_PRICE': 0.10, 'GAS_PRICE': -0.15}
            self.macro_forecast = {'AI_TREND': 0.25, 'CO2_PRICE': 0.10, 'RATES': -0.06, 'OIL_PRICE': 0.15, 'GAS_PRICE': -0.20}
            self.etf_trends = {k: 0.0 for k in ETF_MAP.values()}

    def _normalize(self, value, min_val, max_val):
        if max_val == min_val: return 50
        norm = (value - min_val) / (max_val - min_val) * 100
        return max(0, min(100, norm))

    def _get_raw_components(self, pkd, gus, failures, macro):
        """Oblicza SUROWE wartości komponentów (RAW) przed standaryzacją."""
        
        L_i = failures  
        N_active = gus['firmy']
        
        if pkd in MACRO_RISKS['CO2_INTENSIVE'] or pkd in MACRO_RISKS['ENERGY_SENSITIVE']: sigma_pi = 0.7 
        else: sigma_pi = 0.3
            
        if pkd in AI_PROFILES['THREATENED']: ar_i = 1.0 
        elif pkd in AI_PROFILES['CREATOR']: ar_i = 0.2
        else: ar_i = 0.5 
        
        P_R = gus['ros'] / 100 
        deltaR_Rbar_proxy = macro.get('AI_TREND', 0) 
        VA_E_proxy = 0.15 
        SP_RAW = (0.5 * P_R + 0.3 * abs(deltaR_Rbar_proxy) + 0.2 * VA_E_proxy)

        survival_rate = N_active / (N_active + failures + L_i) if N_active > 0 else 0
        deltaN_Nbar_proxy = macro.get('AI_TREND', 0) 
        deltaE_Ebar_proxy = gus['ros'] / 100
        SH_RAW = (0.4 * abs(deltaN_Nbar_proxy) + 0.3 * survival_rate + 0.3 * deltaE_Ebar_proxy)

        term1_closure_rate = (failures + L_i) / N_active
        RR_RAW = (0.5 * term1_closure_rate * 100 + 0.3 * sigma_pi * 100 + 0.2 * ar_i * 100)
        
        delta_ai = macro.get('AI_TREND', 0)
        raw_ai = 0.0
        if pkd in AI_PROFILES['CREATOR']: raw_ai = 3.0 * delta_ai
        elif pkd in AI_PROFILES['SUPPORTED']: raw_ai = 1.5 * delta_ai
        elif pkd in AI_PROFILES['THREATENED']: raw_ai = -1.5 * abs(delta_ai)
        AI_RAW = raw_ai
        
        mc_penalty = 0 
        co2 = macro.get('CO2_PRICE', 0); oil = macro.get('OIL_PRICE', 0); gas = macro.get('GAS_PRICE', 0); rates = macro.get('RATES', 0)
        if pkd in MACRO_RISKS['CO2_INTENSIVE'] and co2 > 0: mc_penalty += (co2 * 150)
        if pkd in MACRO_RISKS['ENERGY_SENSITIVE'] and (oil > 0 or gas > 0):
            avg_adverse_trend = (max(0, oil) + max(0, gas)) / 2
            mc_penalty += (avg_adverse_trend * 100)
        if pkd in MACRO_RISKS['INTEREST_SENSITIVE'] and rates > 0: mc_penalty += (rates * 150)
        if pkd in MACRO_RISKS['COMMODITY_EXPOSED'] and oil > 0: mc_penalty += (oil * 50)
        MC_RAW = 0.0 - mc_penalty
        
        etf = ETF_MAP.get(pkd)
        trend = self.etf_trends.get(etf, 0)
        GT_RAW = 100 * trend
        GT_RAW_CLIPPED = max(-10, min(10, GT_RAW))
        
        return {'SH': SH_RAW, 'SP': SP_RAW, 'RR': RR_RAW, 'AI': AI_RAW, 'MC': MC_RAW, 'GT': GT_RAW_CLIPPED}

    def _compute_final_score(self, df_components):
        component_names = ['SH', 'SP', 'RR', 'AI', 'MC', 'GT']
        df_scored = df_components.copy()
        
        for col in component_names:
            if df_scored[col].std() == 0:
                df_scored[f'Z_{col}'] = 0
            else:
                df_scored[f'Z_{col}'] = (df_scored[col] - df_scored[col].mean()) / df_scored[col].std()
            
        def normalize_zscore(z):
            return np.clip(50 + (z * 50 / 3), 0, 100)
            
        for col in component_names:
            df_scored[f'N_{col}'] = df_scored[f'Z_{col}'].apply(normalize_zscore)
            
        df_scored['N_SR'] = (df_scored['N_SP'] * 0.6) + (df_scored['N_SH'] * 0.4)
        
        
        df_scored['IAB'] = (
            df_scored['N_SH'] * W['SH'] +
            df_scored['N_SP'] * W['SP'] +
            df_scored['N_SR'] * W['SR'] +  
            df_scored['N_AI'] * W['AI'] +
            df_scored['N_MC'] * W['MC'] +
            df_scored['N_GT'] * W['GT']
        ) - (df_scored['N_RR'] * W['RR'])
        
        df_scored['IAB'] = df_scored['IAB'] * 1.1 + 5 
        df_scored['IAB'] = df_scored['IAB'].clip(0, 100)
        
        return df_scored

    def calculate_and_save(self):
        W_local = W 
        
        print("\n🧮 Obliczanie IAB (Z-Score Standaryzacja)...")
        
        raw_results = []
        
        for pkd in self.data_gus.keys():
            info = self.meta.get(pkd, {'nazwa': f'Dział {pkd}', 'sekcja': '?'})
            gus = self.data_gus[pkd]
            failures = self.data_failures.get(pkd, 0)
            
            if gus['firmy'] == 0: continue
            
            raw_comps_current = self._get_raw_components(pkd, gus, failures, self.macro_current)
            raw_comps_current['pkd'] = pkd
            raw_comps_current['ros'] = gus['ros']
            raw_comps_current['nazwa'] = info['nazwa']
            raw_results.append(raw_comps_current)

        df_raw = pd.DataFrame(raw_results)
        df_final = self._compute_final_score(df_raw.drop(columns=['ros', 'nazwa']).set_index(df_raw['pkd']))
        
        results = []
        for pkd, row in df_final.iterrows():
            
            raw_comps_future = self._get_raw_components(pkd, self.data_gus[pkd], self.data_failures.get(pkd, 0), self.macro_forecast)
            
            delta_raw = 0
            for comp in ['AI', 'MC', 'GT']:
                raw_change = raw_comps_future[comp] - row[comp] 
                
                delta_score_scaled = raw_change * W_local[comp] * 100
                
                delta_raw += delta_score_scaled
            
            score_future = row['IAB'] + delta_raw
            
            delta = score_future - row['IAB']
            
            if delta >= 1.5: outlook = "Pozytywna ↗" 
            elif delta <= -1.5: outlook = "Negatywna ↘"
            else: outlook = "Stabilna ➡"
            
            current_iab = row['IAB']
            risk_cat = 'Wysokie' if current_iab < 40 else ('Niskie' if current_iab > 70 else 'Średnie')
            
            comps_data = {c: round(row[f'N_{c}'], 1) for c in ['SH', 'SP', 'RR', 'AI', 'MC', 'GT']}
            comps_data['SR'] = round(row['N_SR'], 1)

            results.append({
                'pkd': pkd, 'nazwa': self.meta.get(pkd, {}).get('nazwa', f'Dział {pkd}'), 
                'sekcja': self.meta.get(pkd, {}).get('sekcja', '?'),
                'iab': round(current_iab, 1),
                'iab_forecast': round(score_future, 1),
                'risk': risk_cat,
                'outlook': outlook,
                'firmy': self.data_gus[pkd]['firmy'], 
                'upadlosci': self.data_failures.get(pkd, 0), 
                'finanse_ros': self.data_gus[pkd]['ros'],
                'components': comps_data,
                'data_quality': 'COMPLIANT_V12'
            })

        results.sort(key=lambda x: x['iab'], reverse=True)
        
        json_path = OUTPUT_DIR / 'iab_data.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Raport wygenerowany: {json_path}")