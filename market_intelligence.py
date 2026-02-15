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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATE_FILE = 'state.json'

class MarketIntelligence:
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.channel_id = os.getenv('TELEGRAM_CHANNEL_ID')
        
        if not all([self.telegram_token, self.channel_id]):
            logger.error("Missing required environment variables")
            sys.exit(1)
        
        self.exchange = ccxt.kraken()
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
            
            # Hourly data for short-term context (48h)
            btc_ohlcv_1h = self.exchange.fetch_ohlcv('BTC/USD', '1h', limit=48)
            eth_ohlcv_1h = self.exchange.fetch_ohlcv('ETH/USD', '1h', limit=48)
            
            # DAILY data for regime classification and RSI
            btc_ohlcv_1d = self.exchange.fetch_ohlcv('BTC/USD', '1d', limit=60)
            eth_ohlcv_1d = self.exchange.fetch_ohlcv('ETH/USD', '1d', limit=60)
            
            # Validate minimum data length
            if len(btc_ohlcv_1h) < 20:
                raise ValueError(f"Insufficient BTC 1h data: {len(btc_ohlcv_1h)} candles, need ≥20")
            if len(btc_ohlcv_1d) < 30:
                raise ValueError(f"Insufficient BTC 1d data: {len(btc_ohlcv_1d)} candles, need ≥30")
            
            btc_df_1h = pd.DataFrame(btc_ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            eth_df_1h = pd.DataFrame(eth_ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            btc_df_1d = pd.DataFrame(btc_ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            eth_df_1d = pd.DataFrame(eth_ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
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
            
            # Calculate 7-day change from daily data
            btc_price = btc_ticker.get('last', btc_ticker.get('close', 0))
            btc_7d_ago = float(btc_df_1d['close'].iloc[-7]) if len(btc_df_1d) >= 7 else btc_price
            btc_change_7d = ((btc_price / btc_7d_ago) - 1) * 100
            
            eth_price = eth_ticker.get('last', eth_ticker.get('close', 0))
            eth_7d_ago = float(eth_df_1d['close'].iloc[-7]) if len(eth_df_1d) >= 7 else eth_price
            eth_change_7d = ((eth_price / eth_7d_ago) - 1) * 100
            
            return {
                'btc': {
                    'price': btc_price,
                    'change_24h': btc_ticker.get('percentage') or 0.0,
                    'change_7d': btc_change_7d,
                    'volume': float(btc_df_1h['volume'].iloc[-1]),
                    'df_1h': btc_df_1h,
                    'df_1d': btc_df_1d
                },
                'eth': {
                    'price': eth_price,
                    'change_24h': eth_ticker.get('percentage') or 0.0,
                    'change_7d': eth_change_7d,
                    'volume': float(eth_df_1h['volume'].iloc[-1]),
                    'df_1h': eth_df_1h,
                    'df_1d': eth_df_1d
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
    
    def get_round_level(self, price: float) -> int:
        """Get nearest 5k round level"""
        return round(price / 5000) * 5000
    
    def enrich_data_with_indicators(self, data: Dict) -> Dict:
        """Add technical indicators using DAILY data for stability"""
        btc_df = data['btc']['df_1d'].copy()
        
        if len(btc_df) < 14:
            logger.warning(f"Insufficient daily data for RSI: {len(btc_df)} candles")
            data['btc']['rsi'] = 50.0
            data['btc']['rsi_context'] = "neutral"
        else:
            btc_df['close'] = pd.to_numeric(btc_df['close'])
            rsi_series = ta.momentum.RSIIndicator(close=btc_df['close'], window=14).rsi()
            rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0
            data['btc']['rsi'] = rsi
            
            # RSI context for message
            if rsi > 70:
                data['btc']['rsi_context'] = "overbought"
            elif rsi > 60:
                data['btc']['rsi_context'] = "elevated"
            elif rsi < 30:
                data['btc']['rsi_context'] = "oversold"
            elif rsi < 40:
                data['btc']['rsi_context'] = "depressed"
            else:
                data['btc']['rsi_context'] = "neutral"
        
        # Add EMA positions from daily data
        if len(btc_df) >= 50:
            price = float(btc_df['close'].iloc[-1])
            ema20 = ta.trend.EMAIndicator(close=btc_df['close'], window=20).ema_indicator().iloc[-1]
            ema50 = ta.trend.EMAIndicator(close=btc_df['close'], window=50).ema_indicator().iloc[-1]
            
            data['btc']['above_ema20'] = price > ema20 if not pd.isna(ema20) else True
            data['btc']['above_ema50'] = price > ema50 if not pd.isna(ema50) else True
            data['btc']['ema20'] = ema20 if not pd.isna(ema20) else price
            data['btc']['ema50'] = ema50 if not pd.isna(ema50) else price
        else:
            data['btc']['above_ema20'] = True
            data['btc']['above_ema50'] = True
        
        return data
    
    def check_triggers(self, data: Dict) -> Tuple[bool, str]:
        """Check for significant market events that warrant publication"""
        triggers = []
        
        # 24h moves >5% are significant (raised from 3%)
        btc_change = abs(data['btc']['change_24h'])
        eth_change = abs(data['eth']['change_24h'])
        
        if btc_change > 5.0:
            triggers.append(f"BTC 24h: {data['btc']['change_24h']:+.1f}%")
        
        if eth_change > 5.0:
            triggers.append(f"ETH 24h: {data['eth']['change_24h']:+.1f}%")
        
        # 7d moves >10% are significant
        btc_7d = abs(data['btc'].get('change_7d', 0))
        if btc_7d > 10.0:
            triggers.append(f"BTC 7d: {data['btc']['change_7d']:+.1f}%")
        
        return (len(triggers) > 0, ' | '.join(triggers))
    
    def classify_regime(self, data: Dict) -> Dict:
        """
        Classify market regime using DAILY data.
        
        Returns dict with:
        - regime: BULL/BEAR/TRANSITION with qualifier
        - confidence: LOW/MODERATE/HIGH with percentage
        - tail_risk: INACTIVE/ELEVATED/ACTIVE
        - bias: directional statement
        """
        btc_df = data['btc']['df_1d'].copy()
        
        if len(btc_df) < 30:
            raise ValueError(f"Insufficient daily data: {len(btc_df)} candles, need ≥30")
        
        btc_df['close'] = pd.to_numeric(btc_df['close'])
        price = float(btc_df['close'].iloc[-1])
        
        # Daily RSI
        rsi = data['btc'].get('rsi', 50.0)
        
        # Daily EMAs
        ema20 = data['btc'].get('ema20', price)
        ema50 = data['btc'].get('ema50', price)
        above_ema20 = data['btc'].get('above_ema20', True)
        above_ema50 = data['btc'].get('above_ema50', True)
        
        # 7-day change (more meaningful than 24h for regime)
        change_7d = data['btc'].get('change_7d', 0)
        change_24h = data['btc'].get('change_24h', 0)
        
        # === REGIME SCORING ===
        score = 0
        
        # EMA structure (+2/-2)
        if above_ema20 and above_ema50:
            score += 2
        elif not above_ema20 and not above_ema50:
            score -= 2
        elif above_ema20:
            score += 1
        else:
            score -= 1
        
        # 7d momentum (+2/-2)
        if change_7d > 8:
            score += 2
        elif change_7d > 3:
            score += 1
        elif change_7d < -8:
            score -= 2
        elif change_7d < -3:
            score -= 1
        
        # RSI context (+1/-1)
        if rsi > 60:
            score += 1
        elif rsi < 40:
            score -= 1
        
        # === DETERMINE REGIME ===
        if score >= 4:
            regime = "BULL"
            regime_qualifier = None
        elif score >= 2:
            regime = "BULL"
            regime_qualifier = "early"
        elif score <= -4:
            regime = "BEAR"
            regime_qualifier = None
        elif score <= -2:
            regime = "BEAR"
            regime_qualifier = "early"
        else:
            regime = "TRANSITION"
            regime_qualifier = None
        
        # === CONFIDENCE ===
        base_confidence = min(abs(score) * 12, 50)
        
        # Boost for EMA alignment
        if (above_ema20 and above_ema50) or (not above_ema20 and not above_ema50):
            base_confidence += 15
        
        # Boost for RSI confirmation
        if (regime == "BULL" and rsi > 55) or (regime == "BEAR" and rsi < 45):
            base_confidence += 10
        
        confidence_pct = min(base_confidence, 80)
        
        if confidence_pct >= 60:
            confidence_label = "HIGH"
        elif confidence_pct >= 35:
            confidence_label = "MODERATE"
        else:
            confidence_label = "LOW"
        
        # === TAIL RISK ===
        tail_risk = "INACTIVE"
        tail_direction = None
        
        if regime == "BULL" or regime_qualifier == "early" and score > 0:
            if rsi > 75:
                tail_risk = "ACTIVE"
                tail_direction = "↓"
            elif rsi > 68:
                tail_risk = "ELEVATED"
                tail_direction = "↓"
        elif regime == "BEAR":
            if rsi < 25:
                tail_risk = "ACTIVE"
                tail_direction = "↓"
            elif rsi < 32:
                tail_risk = "ELEVATED"
                tail_direction = "↓"
        
        # === BIAS ===
        if regime == "BULL":
            if tail_risk == "ACTIVE":
                bias = "Upside exhaustion risk elevated"
            else:
                bias = "Directional upside favored"
        elif regime == "BEAR":
            if tail_risk == "ACTIVE":
                bias = "Capitulation risk present"
            else:
                bias = "Directional downside pressure"
        else:
            bias = "No clear directional edge"
        
        # Format regime string
        if regime_qualifier:
            regime_str = f"{regime} ({regime_qualifier})"
        else:
            regime_str = regime
        
        return {
            'regime': regime_str,
            'regime_base': regime,
            'confidence': confidence_pct,
            'confidence_label': confidence_label,
            'tail_risk': tail_risk,
            'tail_direction': tail_direction,
            'bias': bias,
            'score': score
        }
    
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
            
            # Log message for debugging
            logger.info(f"Telegram message ({len(message)} chars):\n{message}")
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.channel_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Published to Telegram successfully")
        except requests.exceptions.HTTPError as e:
            # Log response for 400 errors
            if e.response.status_code == 400:
                logger.error(f"Telegram 400 error. Response: {e.response.text}")
            logger.error(f"Telegram error: {e}")
            raise
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            raise
    
    def run(self):
        try:
            logger.info("Starting Market Intelligence scan")
            
            data = self.fetch_market_data()
            data = self.enrich_data_with_indicators(data)
            regime_data = self.classify_regime(data)
            regime = regime_data['regime']
            should_publish, trigger_reason = self.check_triggers(data)
            
            # Check for regime change
            last_regime = self.state.get('last_regime')
            regime_changed = regime != last_regime
            
            if regime_changed and last_regime is not None:
                logger.info(f"Regime shift: {last_regime} -> {regime}")
                if not should_publish:
                    should_publish = True
                    trigger_reason = f"Regime: {last_regime} → {regime}"
                else:
                    trigger_reason += f" | Regime shift"
            
            # Check round level cross (70k, 75k, 80k, 85k...)
            current_level = self.get_round_level(data['btc']['price'])
            last_level = self.state.get('last_round_level')
            
            if last_level and current_level != last_level:
                direction = "above" if current_level > last_level else "below"
                round_trigger = f"BTC crossed {direction} ${current_level:,.0f}"
                
                # Apply 4h cooldown for round level triggers
                last_round_publish = self.state.get('last_round_publish')
                round_cooldown = 4 * 3600  # 4 hours
                
                round_cooldown_active = False
                if last_round_publish:
                    try:
                        last_time = datetime.fromisoformat(last_round_publish)
                        elapsed = (datetime.utcnow() - last_time).total_seconds()
                        if elapsed < round_cooldown:
                            round_cooldown_active = True
                            logger.info(f"Round level trigger ({round_trigger}) but cooldown active: {elapsed/3600:.1f}h elapsed")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse last_round_publish: {e}")
                
                if not round_cooldown_active:
                    logger.info(f"Round level crossed: ${last_level:,.0f} -> ${current_level:,.0f}")
                    if not should_publish:
                        should_publish = True
                        trigger_reason = round_trigger
                    else:
                        trigger_reason += f" | {round_trigger}"
                    
                    round_trigger_fired = True
                else:
                    round_trigger_fired = False
            else:
                round_trigger_fired = False
            
            # Global cooldown: 8 hours between ANY publications (increased from 6)
            if should_publish:
                last_publish = self.state.get('last_publish')
                cooldown = 8 * 3600  # 8 hours in seconds
                
                cooldown_active = False
                if last_publish:
                    try:
                        last_time = datetime.fromisoformat(last_publish)
                        elapsed = (datetime.utcnow() - last_time).total_seconds()
                        if elapsed < cooldown:
                            cooldown_active = True
                            hours_elapsed = elapsed / 3600
                            hours_remaining = (cooldown - elapsed) / 3600
                            logger.info(f"Trigger detected ({trigger_reason}) but cooldown active: {hours_elapsed:.1f}h elapsed, {hours_remaining:.1f}h remaining")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse last_publish: {e}")
                
                should_publish = not cooldown_active
            
            if should_publish:
                logger.info(f"Publishing: {trigger_reason}")
                
                # Calculate volume anomaly
                btc_vol_ma = self.calculate_volume_ma(data['btc']['df_1h'])
                btc_vol_ratio = data['btc']['volume'] / btc_vol_ma if btc_vol_ma > 0 else 1.0
                
                # Regime emoji
                if "BULL" in regime:
                    regime_emoji = "🟢"
                elif "BEAR" in regime:
                    regime_emoji = "🔴"
                else:
                    regime_emoji = "🟡"
                
                # Tail risk display
                tail_display = regime_data['tail_risk']
                if regime_data['tail_direction']:
                    tail_display += f" {regime_data['tail_direction']}"
                
                timestamp = datetime.utcnow().strftime('%d %b %Y %H:%M UTC')
                
                # New regime-based message format
                message = f"""<b>CRYPTO · MARKET STATE</b>

{regime_emoji} Regime: {regime}
Confidence: {regime_data['confidence_label']} ({regime_data['confidence']}%)
Tail risk: {tail_display}

<b>Prices</b>
• BTC: ${data['btc']['price']:,.0f} ({data['btc']['change_24h']:+.1f}% 24h | {data['btc']['change_7d']:+.1f}% 7d)
• ETH: ${data['eth']['price']:,.0f} ({data['eth']['change_24h']:+.1f}% 24h)

<b>Context</b>
• RSI (daily): {data['btc']['rsi']:.0f} ({data['btc']['rsi_context']})
• Volume: {btc_vol_ratio:.1f}x average

<b>Bias</b>
{regime_data['bias']}

<i>Trigger: {trigger_reason}</i>
<i>Radar | {timestamp}</i>"""
                
                # Publish first, then update state only on success
                self.publish_telegram(message)
                logger.info("Telegram publish confirmed")
                
                # Update state AFTER successful publish
                self.state['last_regime'] = regime
                self.state['last_publish'] = datetime.utcnow().isoformat()
                self.state['last_round_level'] = current_level
                if round_trigger_fired:
                    self.state['last_round_publish'] = datetime.utcnow().isoformat()
                
                try:
                    self._save_state()
                    logger.info("State saved successfully")
                except Exception as e:
                    logger.error(f"CRITICAL: Failed to save state: {e}")
                    raise
                
                logger.info("Scan complete - Published")
            else:
                logger.info(f"No significant triggers. Regime: {regime}")
                
                # Update regime and round level even when not publishing
                self.state['last_regime'] = regime
                self.state['last_round_level'] = current_level
                
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
