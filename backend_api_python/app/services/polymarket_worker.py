"""
Polymarket后台任务
每30分钟更新一次市场数据，并批量分析市场机会
"""
import re
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from app.utils.logger import get_logger
from app.utils.db import get_db_connection
from app.data_sources.polymarket import PolymarketDataSource
from app.services.polymarket_batch_analyzer import PolymarketBatchAnalyzer

logger = get_logger(__name__)

# ── Crypto关键词扩展列表 ──────────────────────────────────────────
# Polymarket 的 category 标签不完整，很多加密货币相关市场被标为
# other / politics / tech / finance。此列表用于二次扫描，将这些
# 市场也归入 crypto 类别，供 Smart Trading 消费。
CRYPTO_KEYWORDS: List[str] = [
    # 主流币
    'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol',
    'binance', 'bnb', 'ripple', 'xrp', 'cardano', 'ada',
    'dogecoin', 'doge', 'polkadot', 'dot', 'avalanche', 'avax',
    'polygon', 'matic', 'chainlink', 'link', 'litecoin', 'ltc',
    'uniswap', 'uni', 'pepe', 'pepe',
    # 平台 / 机构
    'coinbase', 'coinbase', 'kraken', 'bybit', 'okx', 'bitget',
    ' tether', 'usdt', 'usdc', 'circle',
    # 监管 / 政策（SEC相关加密事件）
    'sec ', 'spot etf', 'bitcoin etf', 'eth etf', 'crypto etf',
    'crypto regulation', 'crypto ban', 'crypto tax',
    # 概念 / 垂类
    'cryptocurrency', 'crypto ', 'cryptocurrencies',
    'memecoin', 'meme coin', 'meme token', 'meme coins',
    'defi', 'decentralized finance', 'nft', 'nfts',
    'airdrop', 'airdrops', 'staking', 'staking',
    'blockchain', 'web3', 'token', 'tokens',
    'ico', 'ieo', 'ido', 'presale',
    'layer 1', 'layer1', 'layer 2', 'layer2', 'l2', 'rollup',
    'dapp', 'dapps', 'smart contract', 'smart contracts',
    'gas fee', 'gas fees', 'hash rate', 'hashrate',
    'mining', 'miner', 'miners',
    # DeFi 协议
    'aave', 'compound', 'makerdao', 'curve finance', 'lido',
    'sushi', 'sushiswap', 'pancake', 'pancakeswap',
    'megaeth', 'megaeth',
    # NFT / GameFi
    'opensea', 'nft marketplace', 'play-to-earn', 'gamefi',
    # 稳定币
    'stablecoin', 'stablecoins',
    # 钱包 / 基础设施
    'metamask', 'ledger', 'trezor', 'phantom',
    # 关键人物（加密相关）
    'satoshi', 'vitalik', 'cz ',
    # 常见项目
    'arbitrum', 'arb ', 'optimism', 'op ', 'zkSync', 'starknet',
    'base chain', 'base network',
    'ton ', 'toncoin',
]

# 编译为一条正则，忽略大小写
_CRYPTO_PATTERN = re.compile(
    r'(?:' + '|'.join(re.escape(kw) for kw in CRYPTO_KEYWORDS) + r')',
    re.IGNORECASE,
)


class PolymarketWorker:
    """Polymarket数据更新和分析后台任务"""
    
    def __init__(self, update_interval_minutes: int = 30, analysis_cache_minutes: int = 1440):  # 24小时缓存
        """
        初始化后台任务
        
        Args:
            update_interval_minutes: 市场数据更新间隔（分钟）
            analysis_cache_minutes: AI分析结果缓存时间（分钟）
        """
        self.update_interval_minutes = update_interval_minutes
        self.analysis_cache_minutes = analysis_cache_minutes
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.polymarket_source = PolymarketDataSource()
        self.batch_analyzer = PolymarketBatchAnalyzer()
        self._last_update_ts = 0.0
        
    def start(self) -> bool:
        """启动后台任务"""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="PolymarketWorker", daemon=True)
            self._thread.start()
            logger.info(f"PolymarketWorker started (update_interval={self.update_interval_minutes}min, cache={self.analysis_cache_minutes}min)")
            return True
    
    def stop(self, timeout_sec: float = 5.0) -> None:
        """停止后台任务"""
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                return
            self._stop_event.set()
            self._thread.join(timeout=timeout_sec)
            if self._thread.is_alive():
                logger.warning("PolymarketWorker thread did not stop within timeout")
            else:
                logger.info("PolymarketWorker stopped")
    
    def _run_loop(self) -> None:
        """主循环"""
        logger.info("PolymarketWorker loop started")
        
        # 启动时立即执行一次
        self._update_markets_and_analyze()
        
        while not self._stop_event.is_set():
            try:
                # 等待指定时间间隔
                wait_seconds = self.update_interval_minutes * 60
                if self._stop_event.wait(wait_seconds):
                    break  # 如果收到停止信号，退出循环
                
                # 执行更新和分析
                self._update_markets_and_analyze()
                
            except Exception as e:
                logger.error(f"PolymarketWorker loop error: {e}", exc_info=True)
                # 出错后等待1分钟再重试
                self._stop_event.wait(60)
        
        logger.info("PolymarketWorker loop stopped")
    
    @staticmethod
    def _is_crypto_market(market: Dict) -> bool:
        """检查市场是否与加密货币相关（基于关键词）"""
        question = (market.get('question') or '').lower()
        slug = (market.get('slug') or '').lower()
        description = (market.get('description') or '').lower()
        text = f"{question} {slug} {description}"
        return bool(_CRYPTO_PATTERN.search(text))
    
    def _reclassify_crypto_markets(self, unique_markets: Dict[str, Dict]) -> Dict[str, int]:
        """
        二次扫描：跨所有类别检测含 crypto 关键词的市场，
        将其 category 更新为 'crypto' 并保留原始类别为 original_category。
        
        Returns:
            reclassified_count: 被重新分类的市场数量（不含原本已是 crypto 的）
        """
        reclassified_count = 0
        reclassified_from: Dict[str, int] = {}  # 原类别 -> 被重分类数量
        
        for market_id, market in unique_markets.items():
            original_cat = market.get('category', 'other')
            
            # 如果已经是 crypto，跳过
            if original_cat == 'crypto':
                continue
            
            # 关键词匹配
            if self._is_crypto_market(market):
                # 保留原始类别
                market['original_category'] = original_cat
                market['secondary_category'] = original_cat  # 兼容下游查询
                market['category'] = 'crypto'
                reclassified_count += 1
                reclassified_from[original_cat] = reclassified_from.get(original_cat, 0) + 1
                
                logger.debug(
                    f"Reclassified market [{original_cat} -> crypto]: "
                    f"{market.get('question', '')[:80]}"
                )
        
        if reclassified_count > 0:
            logger.info(
                f"Crypto keyword scan: reclassified {reclassified_count} markets "
                f"(from: {reclassified_from})"
            )
        
        return reclassified_count
    
    def _update_markets_and_analyze(self) -> None:
        """更新市场数据并分析"""
        try:
            logger.info("Starting Polymarket data update and analysis...")
            start_time = time.time()
            
            # ── Step 1: 采集市场 ──────────────────────────────────
            # Gamma API /events has no category param — fetch ALL once, categorize locally.
            all_markets = self.polymarket_source.get_trending_markets(category="all", limit=500)
            logger.info(f"Fetched {len(all_markets)} markets from Gamma API (single request)")
            
            # ── Step 2: 去重 ──────────────────────────────────────
            unique_markets: Dict[str, Dict] = {}
            cat_counts_before: Dict[str, int] = {}
            for market in all_markets:
                market_id = market.get('market_id')
                if market_id:
                    unique_markets[market_id] = market
                    cat = market.get('category', 'other')
                    cat_counts_before[cat] = cat_counts_before.get(cat, 0) + 1
            
            logger.info(
                f"Total unique markets: {len(unique_markets)}, "
                f"categories BEFORE crypto scan: {cat_counts_before}"
            )
            
            # ── Step 3: Crypto 关键词二次扫描 ─────────────────────
            # Polymarket 的 category 标签不完整，很多加密货币市场被标为
            # other/politics/tech/finance。通过关键词扫描将这些市场也归入 crypto。
            reclassified_count = self._reclassify_crypto_markets(unique_markets)
            
            # 统计重分类后的类别分布
            cat_counts_after: Dict[str, int] = {}
            for market in unique_markets.values():
                cat = market.get('category', 'other')
                cat_counts_after[cat] = cat_counts_after.get(cat, 0) + 1
            
            crypto_count = cat_counts_after.get('crypto', 0)
            logger.info(
                f"Categories AFTER crypto scan: {cat_counts_after} "
                f"(crypto: {crypto_count}, +{reclassified_count} reclassified)"
            )
            
            # ── Step 4: 保存更新后的市场数据到数据库 ──────────────
            markets_list = list(unique_markets.values())
            try:
                self.polymarket_source._save_markets_to_db(markets_list)
                logger.info(f"Saved {len(markets_list)} markets to DB (with updated categories)")
            except Exception as save_err:
                logger.warning(f"Failed to save markets to DB: {save_err}")
            
            # ── Step 5: 规则筛选 + LLM 分析 ───────────────────────
            # 优化策略：先用规则筛选，只对高价值机会调用LLM
            # 这样可以大幅减少LLM调用次数，节省token
            
            # 5a. 规则筛选：高交易量 + 明显概率偏差
            rule_based_opportunities = []
            for market in markets_list:
                prob = market.get('current_probability', 50.0)
                volume = market.get('volume_24h', 0)
                divergence = abs(prob - 50.0)
                
                # 规则筛选：高交易量 + 明显概率偏差
                if volume > 5000 and divergence > 8:
                    rule_based_opportunities.append(market)
            
            # 5b. 只对规则筛选出的机会调用LLM（最多30个，节省token）
            if rule_based_opportunities:
                logger.info(f"Rule-based filtering: {len(rule_based_opportunities)} opportunities, analyzing top 30 with LLM")
                # 按交易量和概率偏差排序，取前30个
                rule_based_opportunities.sort(
                    key=lambda x: (x.get('volume_24h', 0) * abs(x.get('current_probability', 50) - 50)),
                    reverse=True
                )
                top_opportunities = rule_based_opportunities[:30]
                
                analyzed_markets = self.batch_analyzer.batch_analyze_markets(
                    top_opportunities,
                    max_opportunities=30  # 只分析30个最有价值的机会
                )
            else:
                logger.info("No rule-based opportunities found, skipping LLM analysis")
                analyzed_markets = []
            
            # ── Step 6: 保存 AI 分析结果 ──────────────────────────
            if analyzed_markets:
                self.batch_analyzer.save_batch_analysis(analyzed_markets)
                analyzed_count = len(analyzed_markets)
            else:
                analyzed_count = 0
            
            elapsed = time.time() - start_time
            logger.info(
                f"Polymarket update completed: {len(unique_markets)} markets, "
                f"{crypto_count} crypto (incl. {reclassified_count} keyword-matched), "
                f"{analyzed_count} AI-identified opportunities in {elapsed:.1f}s"
            )
            self._last_update_ts = time.time()
            
        except Exception as e:
            logger.error(f"Failed to update markets and analyze: {e}", exc_info=True)
    
    
    def force_update(self) -> None:
        """强制立即更新（用于手动触发）"""
        logger.info("Force update triggered")
        self._update_markets_and_analyze()


# 全局单例
_polymarket_worker: Optional[PolymarketWorker] = None
_worker_lock = threading.Lock()


def get_polymarket_worker() -> PolymarketWorker:
    """获取PolymarketWorker单例"""
    global _polymarket_worker
    with _worker_lock:
        if _polymarket_worker is None:
            update_interval = int(os.getenv("POLYMARKET_UPDATE_INTERVAL_MIN", "30"))
            cache_minutes = int(os.getenv("POLYMARKET_ANALYSIS_CACHE_MIN", "30"))
            _polymarket_worker = PolymarketWorker(
                update_interval_minutes=update_interval,
                analysis_cache_minutes=cache_minutes
            )
        return _polymarket_worker


# 需要导入os
import os
