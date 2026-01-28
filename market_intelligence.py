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
            
            # P0 Fix: Validate minimum data length
            if len(btc_ohlcv) < 20:
                raise ValueError(f"Insufficient BTC data: {len(btc_ohlcv)} candles, need ≥20")
            if len(eth_ohlcv) < 20:
                raise ValueError(f"Insufficient ETH data: {len(eth_ohlcv)} candles, need ≥20")
            
            btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            eth_df = pd.DataFrame(eth_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            spx = None
            try:
                spx = yf.download('^GSPC', period='5d', progress=False)
            except Exception as e:
                logger.warning(f"SPX download failed: {e}")
            
            if spx is not None and len(spx) >= 2:
                spx_price = float(spx['Close'].iloc[-1])
                spx_change = ((spx_price / float(spx['Close'].iloc[-2])) - 1) * 100
            else:
                spx_price = None
                spx_change = 0
            
            # P0 Fix: Handle None percentage from Kraken
            return {
                'btc': {
                    'price': btc_ticker.get('last', btc_ticker.get('close', 0)),
                    'change_24h': btc_ticker.get('percentage') or 0.0,
                    'volume': float(btc_df['volume'].iloc[-1]),
                    'df': btc_df
                },
                'eth': {
                    'price': eth_ticker.get('last', eth_ticker.get('close', 0)),
                    'change_24h': eth_ticker.get('percentage') or 0.0,
                    'volume': float(eth_df['volume'].iloc[-1]),
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
    
    def enrich_data_with_indicators(self, data: Dict) -> Dict:
        """Add technical indicators to data"""
        btc_df = data['btc']['df'].copy()
        
        if len(btc_df) < 14:
            logger.warning(f"Insufficient data for RSI: {len(btc_df)} candles")
            data['btc']['rsi'] = 50.0  # Default fallback
        else:
            btc_df['close'] = pd.to_numeric(btc_df['close'])
            rsi_series = ta.momentum.RSIIndicator(close=btc_df['close'], window=14).rsi()
            data['btc']['rsi'] = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0
        
        return data
    
    def check_triggers(self, data: Dict) -> Tuple[bool, str]:
        triggers = []
        
        btc_change = abs(data['btc']['change_24h'])
        eth_change = abs(data['eth']['change_24h'])
        
        if btc_change > 3.0:
            triggers.append(f"BTC move {data['btc']['change_24h']:+.1f}%")
        
        if eth_change > 3.0:
            triggers.append(f"ETH move {data['eth']['change_24h']:+.1f}%")
        
        return (len(triggers) > 0, ' | '.join(triggers))
    
    def classify_regime(self, data: Dict) -> str:
        btc_df = data['btc']['df'].copy()
        
        # P0 Fix: Validate data length before processing
        if len(btc_df) < 14:
            raise ValueError(f"Insufficient data for RSI: {len(btc_df)} candles, need ≥14")
        
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
        spx_status = 'closed' if data['spx']['price'] is None else 'open'
        
        # Calculate volume regime
        btc_vol_ma = self.calculate_volume_ma(data['btc']['df'])
        # P0 Fix: Better handling of zero volume MA
        if btc_vol_ma > 0:
            btc_vol_ratio = data['btc']['volume'] / btc_vol_ma
        else:
            logger.warning("BTC volume MA is zero, using volume directly")
            btc_vol_ratio = 1.0  # Fallback for display only
        
        eth_vol_ma = self.calculate_volume_ma(data['eth']['df'])
        if eth_vol_ma > 0:
            eth_vol_ratio = data['eth']['volume'] / eth_vol_ma
        else:
            logger.warning("ETH volume MA is zero, using volume directly")
            eth_vol_ratio = 1.0  # Fallback for display only
        
        # Determine volume regime
        if btc_vol_ratio < 2:
            vol_regime = "Normal"
        elif btc_vol_ratio < 5:
            vol_regime = "Elevated"
        elif btc_vol_ratio < 20:
            vol_regime = "High"
        else:
            vol_regime = "Extreme"
        
        # Calculate ATR
        btc_df = data['btc']['df'].copy()
        btc_df['tr'] = btc_df[['high', 'low']].apply(lambda x: x['high'] - x['low'], axis=1)
        atr = btc_df['tr'].tail(14).mean()
        
        context = f"""DATA:

BTC: ${data['btc']['price']:,.0f} | 24h: {data['btc']['change_24h']:+.1f}%
ETH: ${data['eth']['price']:,.0f} | 24h: {data['eth']['change_24h']:+.1f}%
BTC vol ratio: {btc_vol_ratio:.1f}x MA
ETH vol ratio: {eth_vol_ratio:.1f}x MA
Vol regime: {vol_regime}
ATR(14): ${atr:,.0f}
SPX: {spx_status}
RSI: {data['btc']['rsi']:.0f}

Classified regime: {regime}
Trigger: {trigger_reason}
"""

        system_prompt = """CRITICAL FORMAT RULES (FOLLOW EXACTLY):
- NO numbered lists (1), 2), 3))
- NO paragraphs or sentences - ONLY bullets (•)
- NO "Confidence:" score
- Terminal wire style - SHORT bullets only

OUTPUT STRUCTURE:

Market Regime: [from data]
Vol regime: [Normal/Elevated/High/Extreme]

Liquidity Snapshot
• BTC: $XX,XXX (±X.X%)
• ETH: $X,XXX (±X.X%)
• Vol: X.Xx MA (BTC), X.Xx MA (ETH)
• ATR: $XXX
• SPX: open/closed
• RSI: XX

Hard Signals
• Trend: [range/bull/bear with levels]
• Momentum: [UP/DOWN/FLAT]
• Volume: [accumulation/distribution/neutral]
• Volatility: [expansion/compression]

Alpha Take
Base: [primary scenario + levels]
Alt: [alternative + trigger]
Bias: [Neutral/Long/Short on condition]

Risk Flags
[List 2-4 specific risks]

STYLE:
- ONLY bullets (•), never numbered lists
- No paragraphs, no full sentences after bullets
- Short phrases: "BTC range 88k-92k" not "BTC is trading in a range between..."
- No "suggests", "likely", "potentially"
- Concrete numbers only"""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}"}
                ],
                temperature=0.6,
                max_tokens=600,
                timeout=30.0
            )
            
            if not response.choices or len(response.choices) == 0:
                raise ValueError("OpenAI returned empty choices")
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        reraise=True
    )
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
            logger.info("Published to Telegram successfully")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            raise
    
    def run(self):
        try:
            logger.info("Starting Market Intelligence scan")
            
            data = self.fetch_market_data()
            data = self.enrich_data_with_indicators(data)  # Add RSI and other indicators
            regime = self.classify_regime(data)
            should_publish, trigger_reason = self.check_triggers(data)
            
            regime_changed = regime != self.state.get('last_regime')
            
            # P2 Fix: Add cooldown for regime shifts (4 hours)
            if regime_changed:
                last_regime_publish = self.state.get('last_regime_publish')
                regime_cooldown = 4 * 3600  # 4 hours in seconds
                
                cooldown_active = False
                if last_regime_publish:
                    try:
                        last_time = datetime.fromisoformat(last_regime_publish)
                        elapsed = (datetime.utcnow() - last_time).total_seconds()
                        if elapsed < regime_cooldown:
                            cooldown_active = True
                            logger.info(f"Regime shift detected but cooldown active ({elapsed/3600:.1f}h elapsed, need 4h)")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse last_regime_publish: {e}")
                
                if not cooldown_active:
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
                
                # P1 Fix: Publish first, then update state only on success
                self.publish_telegram(message)
                logger.info("Telegram publish confirmed")
                
                # Update state AFTER successful publish
                self.state['last_regime'] = regime
                self.state['last_publish'] = datetime.utcnow().isoformat()
                if regime_changed:
                    self.state['last_regime_publish'] = datetime.utcnow().isoformat()
                
                try:
                    self._save_state()
                    logger.info("State saved successfully")
                except Exception as e:
                    logger.error(f"CRITICAL: Failed to save state: {e}")
                    raise
                
                logger.info("Scan complete - Published")
            else:
                logger.info(f"No significant triggers. Regime: {regime}")
                
                # Update regime even when not publishing
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
