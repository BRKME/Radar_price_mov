import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import ccxt
import yfinance as yf
import pandas as pd
import ta
import requests
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATE_FILE = 'state.json'

class MarketIntelligence:
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.channel_id = os.getenv('TELEGRAM_CHANNEL_ID')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not all([self.telegram_token, self.channel_id, self.openai_api_key]):
            logger.error("Missing required environment variables")
            sys.exit(1)
        
        self.exchange = ccxt.kraken()
        self.openai = OpenAI(api_key=self.openai_api_key)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"State file corrupted, using defaults: {e}")
        return {'last_regime': None, 'last_publish': None}
    
    def _save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError)),
        reraise=True
    )
    def fetch_market_data(self) -> Dict:
        try:
            btc_ticker = self.exchange.fetch_ticker('BTC/USD')
            eth_ticker = self.exchange.fetch_ticker('ETH/USD')
            
            btc_ohlcv = self.exchange.fetch_ohlcv('BTC/USD', '1h', limit=48)
            eth_ohlcv = self.exchange.fetch_ohlcv('ETH/USD', '1h', limit=48)
            
            btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            eth_df = pd.DataFrame(eth_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            spx = yf.download('^GSPC', period='5d', progress=False)
            if len(spx) >= 2:
                spx_price = float(spx['Close'].iloc[-1])
                spx_change = ((spx_price / float(spx['Close'].iloc[-2])) - 1) * 100
            else:
                spx_price = None
                spx_change = 0
            
            return {
                'btc': {
                    'price': btc_ticker.get('last', btc_ticker.get('close', 0)),
                    'change_24h': btc_ticker.get('percentage', 0),
                    'volume': btc_ticker.get('quoteVolume', btc_ticker.get('baseVolume', 0)),
                    'df': btc_df
                },
                'eth': {
                    'price': eth_ticker.get('last', eth_ticker.get('close', 0)),
                    'change_24h': eth_ticker.get('percentage', 0),
                    'volume': eth_ticker.get('quoteVolume', eth_ticker.get('baseVolume', 0)),
                    'df': eth_df
                },
                'spx': {
                    'price': spx_price,
                    'change': spx_change
                }
            }
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            raise
    
    def calculate_volume_ma(self, df: pd.DataFrame, period: int = 20) -> float:
        return df['volume'].tail(period).mean()
    
    def check_triggers(self, data: Dict) -> Tuple[bool, str]:
        triggers = []
        
        btc_change = abs(data['btc']['change_24h'])
        eth_change = abs(data['eth']['change_24h'])
        
        if btc_change > 2.0:
            triggers.append(f"BTC move {data['btc']['change_24h']:+.1f}%")
        
        if eth_change > 2.0:
            triggers.append(f"ETH move {data['eth']['change_24h']:+.1f}%")
        
        btc_vol_ma = self.calculate_volume_ma(data['btc']['df'])
        if btc_vol_ma > 0 and data['btc']['volume'] > btc_vol_ma * 2:
            vol_ratio = data['btc']['volume'] / btc_vol_ma
            triggers.append(f"BTC volume spike {vol_ratio:.1f}x MA")
        
        eth_vol_ma = self.calculate_volume_ma(data['eth']['df'])
        if eth_vol_ma > 0 and data['eth']['volume'] > eth_vol_ma * 2:
            vol_ratio = data['eth']['volume'] / eth_vol_ma
            triggers.append(f"ETH volume spike {vol_ratio:.1f}x MA")
        
        return (len(triggers) > 0, ' | '.join(triggers))
    
    def classify_regime(self, data: Dict) -> str:
        btc_df = data['btc']['df'].copy()
        btc_df['close'] = pd.to_numeric(btc_df['close'])
        
        price = btc_df['close'].iloc[-1]
        
        rsi_series = ta.momentum.RSIIndicator(close=btc_df['close'], window=14).rsi()
        rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0
        
        ema20_series = ta.trend.EMAIndicator(close=btc_df['close'], window=20).ema_indicator()
        ema20 = ema20_series.iloc[-1]
        if pd.isna(ema20):
            ema20 = price
        
        ema50_series = ta.trend.EMAIndicator(close=btc_df['close'], window=48).ema_indicator()
        ema50 = ema50_series.iloc[-1]
        if pd.isna(ema50):
            ema50 = ema20
        
        btc_change = data['btc']['change_24h']
        
        if price > ema20 > ema50 and btc_change > 5 and rsi > 70:
            return "Bull Trend (Late-stage)"
        elif price > ema20 > ema50 and btc_change > 2:
            return "Bull Trend (Impulse)"
        elif price < ema20 < ema50 and btc_change < -5 and rsi < 30:
            return "Bear Trend (Capitulation)"
        elif price < ema20 < ema50 and btc_change < -2:
            return "Bear Trend (Distribution Breakdown)"
        elif abs(btc_change) < 1 and 45 < rsi < 55:
            return "Transitional Consolidation"
        elif abs(btc_change) < 2 and rsi > 60 and price > ema20:
            return "Late-range Distribution"
        elif abs(btc_change) < 2 and rsi < 40 and price < ema20:
            return "Early Range Accumulation"
        else:
            return "Range-to-Trend Transition"
    
    def generate_intelligence(self, data: Dict, regime: str, trigger_reason: str) -> str:
        spx_status = 'Market closed' if data['spx']['price'] is None else f"{data['spx']['change']:+.1f}% (daily)"
        
        # Calculate volume anomaly ratio
        btc_vol_ma = self.calculate_volume_ma(data['btc']['df'])
        btc_vol_ratio = data['btc']['volume'] / btc_vol_ma if btc_vol_ma > 0 else 1.0
        
        eth_vol_ma = self.calculate_volume_ma(data['eth']['df'])
        eth_vol_ratio = data['eth']['volume'] / eth_vol_ma if eth_vol_ma > 0 else 1.0
        
        # Calculate ATR (Average True Range) for volatility
        btc_df = data['btc']['df'].copy()
        btc_df['tr'] = btc_df[['high', 'low']].apply(lambda x: x['high'] - x['low'], axis=1)
        atr = btc_df['tr'].tail(14).mean()
        
        context = f"""MARKET DATA:

BTC: ${data['btc']['price']:,.0f} | 24h: {data['btc']['change_24h']:+.1f}% | Volume anomaly: {btc_vol_ratio:.1f}x MA
ETH: ${data['eth']['price']:,.0f} | 24h: {data['eth']['change_24h']:+.1f}%
ETH Volume anomaly: {eth_vol_ratio:.1f}x MA

SPX: {spx_status}
ATR (14): ${atr:,.0f}

Classified Regime: {regime}
Trigger: {trigger_reason}
"""

        system_prompt = """You are an institutional-grade crypto market intelligence engine.
Produce concise, quant-driven market briefs with actionable signals.

OUTPUT FORMAT (STRICT):

1) Market Regime Classification
Market Regime: <regime from user data>
Drivers: <specific quant indicators from data - volume ratio, price action, ATR>

2) Price & Liquidity Snapshot
Include only quant-relevant metrics. Format: 2-3 concise sentences with specific numbers.
- BTC/ETH price deltas
- Volume vs MA ratios
- ATR / volatility context
- Liquidity conditions (SPX status, session type)

3) Hard Signal Interpretation
Replace generic macro with concrete hypotheses based on available data.
Use probabilistic language. 3-4 sentences.
Since you don't have OI/funding data, focus on:
- Volume spike patterns (accumulation vs distribution)
- Cross-asset correlation implications (SPX closed/open)
- Price-volume divergences
- Regime transition signals

4) Institutional Alpha Take
Format:
Base Case: <most likely scenario with price levels>
Alt Scenario: <alternative path with trigger condition>
Positioning Bias: <conditional bias - "Neutral-to-Long on X" or "Risk-Off pending Y">

5) Confidence & Risk Flags
Confidence: <0.50-0.85 based on signal strength>
Risk Flags: <2-4 specific risks from: leverage proxy, low liquidity, weekend gap, macro uncertainty, correlation spike>

STYLE RULES:
- No emojis, no hype language
- Hedge fund research tone
- Quantified statements (ratios, deltas, ranges)
- Avoid predictions without conditional triggers
- No generic AI disclaimers

Keep total output under 900 characters."""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze:\n\n{context}"}
                ],
                temperature=0.7,
                max_tokens=700,
                timeout=30.0
            )
            
            if not response.choices or len(response.choices) == 0:
                raise ValueError("OpenAI returned empty choices")
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def publish_telegram(self, message: str):
        try:
            if len(message) > 4096:
                logger.warning(f"Message too long ({len(message)} chars), truncating")
                message = message[:4090] + "\n..."
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.channel_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Published to Telegram")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            raise
    
    def run(self):
        try:
            logger.info("Starting Market Intelligence scan")
            
            data = self.fetch_market_data()
            regime = self.classify_regime(data)
            should_publish, trigger_reason = self.check_triggers(data)
            
            regime_changed = regime != self.state.get('last_regime')
            
            if regime_changed:
                logger.info(f"Regime shift: {self.state.get('last_regime')} -> {regime}")
                should_publish = True
                if trigger_reason:
                    trigger_reason += " | Regime shift"
                else:
                    trigger_reason = "Regime shift"
            
            if should_publish:
                logger.info(f"Publishing: {trigger_reason}")
                
                intelligence = self.generate_intelligence(data, regime, trigger_reason)
                
                # Calculate volume anomaly for header
                btc_vol_ma = self.calculate_volume_ma(data['btc']['df'])
                btc_vol_ratio = data['btc']['volume'] / btc_vol_ma if btc_vol_ma > 0 else 1.0
                
                timestamp = datetime.utcnow().strftime('%d %b %Y %H:%M UTC')
                message = f"""<b>CRYPTO MARKET INTELLIGENCE</b>

BTC: ${data['btc']['price']:,.0f} | 24h: {data['btc']['change_24h']:+.1f}% | Volume anomaly: {btc_vol_ratio:.1f}x MA

{intelligence}

<i>Radar | {timestamp}</i>"""
                
                self.publish_telegram(message)
                
                self.state['last_regime'] = regime
                self.state['last_publish'] = datetime.utcnow().isoformat()
                
                try:
                    self._save_state()
                    logger.info("State saved successfully")
                except Exception as e:
                    logger.error(f"CRITICAL: Failed to save state: {e}")
                    raise
                
                logger.info("Scan complete - Published")
            else:
                logger.info(f"No significant triggers. Regime: {regime}")
                
                self.state['last_regime'] = regime
                
                try:
                    self._save_state()
                except Exception as e:
                    logger.warning(f"Failed to save state (non-critical): {e}")
                
                logger.info("Scan complete - No publish")
        
        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    bot = MarketIntelligence()
    bot.run()
