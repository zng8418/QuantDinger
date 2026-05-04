"""
Polymarket批量分析器
一次性分析多个市场，由AI筛选出有交易机会的市场
"""
import json
import re
from typing import List, Dict, Optional
from app.utils.logger import get_logger
from app.utils.db import get_db_connection
from app.services.llm import LLMService
from app.data_sources.polymarket import PolymarketDataSource

logger = get_logger(__name__)


class PolymarketBatchAnalyzer:
    """批量分析预测市场，由AI筛选交易机会"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.polymarket_source = PolymarketDataSource()
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[dict]:
        """
        从LLM返回的文本中提取JSON对象。
        按优先级尝试：
          1. 直接解析整个字符串
          2. 去除 markdown 代码围栏 (```json ... ```)
          3. 用正则匹配第一个完整的 {...}
        """
        if not text:
            return None

        # 1) 直接解析
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # 2) 去除 markdown 围栏
        cleaned = text.strip()
        # Match ```json ... ``` or ``` ... ```
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # 3) 正则提取第一个完整的 JSON 对象 {...}
        #    使用非贪婪匹配，找到最外层的 { ... }
        brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass

        # 4) Broader attempt: find first { and last } and try parse
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def batch_analyze_markets(self, markets: List[Dict], max_opportunities: int = 20) -> List[Dict]:
        """
        批量分析市场，由AI筛选出有交易机会的市场
        
        Args:
            markets: 市场列表
            max_opportunities: 最多返回多少个交易机会
            
        Returns:
            筛选后的市场列表（包含AI分析结果）
        """
        if not markets:
            return []
        
        try:
            # 1. 构建批量分析的prompt
            markets_summary = self._build_markets_summary(markets)
            
            prompt = f"""Analyze the following prediction markets and identify the best trading opportunities.

Market List:
{markets_summary}

Evaluate each market based on:
1. Market Activity: Is trading volume and liquidity sufficient?
2. Probability Deviation: Does current market probability deviate from reasonable expectations? (Further from 50% = more opportunity)
3. Event Importance: How significant is the event's market impact?
4. Time Window: Is the settlement timeframe appropriate? (Too close or too far is bad)
5. Information Edge: Is there obvious information asymmetry or market mispricing?

IMPORTANT: You MUST respond with ONLY a valid JSON object, no additional text, no explanation, no markdown formatting.
Return ONLY the JSON below:

{{
    "opportunities": [
        {{
            "market_id": "<the market ID from the list>",
            "opportunity_score": 85,
            "reason": "Brief reason why this market has a trading opportunity",
            "recommendation": "YES",
            "confidence": 75,
            "key_factors": ["factor1", "factor2"],
            "predicted_probability": 65.0
        }}
    ]
}}

Rules:
- Return at most {max_opportunities} opportunities
- Only include opportunities with score >= 60
- Prioritize: high volume + clear probability deviation + high confidence
- Keep reasons concise
- predicted_probability: YOUR estimated true probability (0.0-100.0), which may differ from market probability
- recommendation must be YES, NO, or HOLD"""
            
            # 2. 调用LLM进行批量分析
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional prediction market analyst. "
                        "You MUST respond with ONLY valid JSON. No explanatory text, no markdown, no comments inside JSON. "
                        "Output raw JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            logger.info(f"Batch analyzing {len(markets)} markets, requesting {max_opportunities} opportunities")
            result_text = self.llm_service.call_llm_api(
                messages=messages,
                use_json_mode=True,
                temperature=0.3
            )
            
            # 3. 解析结果 — 使用健壮的 JSON 提取
            result = None
            if isinstance(result_text, dict):
                result = result_text
            elif isinstance(result_text, str):
                result = self._extract_json_from_text(result_text)
            
            if result is None:
                logger.error(
                    f"Failed to extract JSON from LLM response. "
                    f"Raw text (first 500 chars): {str(result_text)[:500]}"
                )
                return self._fallback_analysis(markets, max_opportunities)
            
            opportunities = result.get('opportunities', [])
            if not opportunities:
                logger.warning("LLM returned no opportunities, using fallback")
                return self._fallback_analysis(markets, max_opportunities)
            
            # 4. 将AI分析结果合并到市场数据中
            opportunities_map = {opp.get('market_id'): opp for opp in opportunities}
            analyzed_markets = []
            
            for market in markets:
                market_id = market.get('market_id')
                if not market_id:
                    continue
                
                opp = opportunities_map.get(market_id)
                if opp:
                    # 获取AI预测的概率
                    predicted_prob = float(opp.get('predicted_probability', market.get('current_probability', 50.0)))
                    market_prob = market.get('current_probability', 50.0)
                    divergence = predicted_prob - market_prob
                    
                    # 合并AI分析结果
                    market['ai_analysis'] = {
                        'predicted_probability': predicted_prob,  # 使用AI预测的概率
                        'recommendation': opp.get('recommendation', 'HOLD'),
                        'confidence_score': float(opp.get('confidence', 0)),
                        'opportunity_score': float(opp.get('opportunity_score', 0)),
                        'divergence': divergence,  # AI预测概率 - 市场概率
                        'reasoning': opp.get('reason', ''),
                        'key_factors': opp.get('key_factors', [])
                    }
                    analyzed_markets.append(market)
            
            # 5. 按机会评分排序
            analyzed_markets.sort(
                key=lambda x: x.get('ai_analysis', {}).get('opportunity_score', 0),
                reverse=True
            )
            
            logger.info(f"Batch analysis completed: {len(analyzed_markets)} opportunities identified")
            return analyzed_markets
            
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages for common API errors
            if "403" in error_msg or "Forbidden" in error_msg:
                logger.error(
                    f"Batch analysis failed: LLM API 403 Forbidden. "
                    f"Please check: 1) API key is correctly configured 2) API key is valid 3) Account balance is sufficient. "
                    f"Detail: {error_msg}"
                )
            elif "401" in error_msg or "Unauthorized" in error_msg:
                logger.error(
                    f"Batch analysis failed: LLM API 401 Unauthorized. "
                    f"API key is invalid or expired. Check backend_api_python/.env configuration. "
                    f"Detail: {error_msg}"
                )
            else:
                logger.error(f"Batch analysis failed: {error_msg}", exc_info=True)
            return self._fallback_analysis(markets, max_opportunities)
    
    def _build_markets_summary(self, markets: List[Dict]) -> str:
        """构建市场摘要，用于批量分析"""
        summary_lines = []
        
        for i, market in enumerate(markets[:50], 1):  # 限制最多50个，避免prompt过长
            market_id = market.get('market_id', '')
            question = market.get('question', '')[:100]  # 限制长度
            prob = market.get('current_probability', 50.0)
            volume = market.get('volume_24h', 0)
            category = market.get('category', 'other')
            
            summary_lines.append(
                f"{i}. ID: {market_id}\n"
                f"   问题: {question}\n"
                f"   当前概率: {prob:.1f}%\n"
                f"   24h交易量: ${volume:,.0f}\n"
                f"   分类: {category}"
            )
        
        return "\n\n".join(summary_lines)
    
    def _fallback_analysis(self, markets: List[Dict], max_opportunities: int) -> List[Dict]:
        """回退分析：基于简单规则筛选"""
        opportunities = []
        
        for market in markets:
            prob = market.get('current_probability', 50.0)
            volume = market.get('volume_24h', 0)
            
            # 简单规则：交易量大 + 概率偏离50%
            if volume > 10000 and abs(prob - 50.0) > 10:
                opportunity_score = min(60 + abs(prob - 50.0) * 0.5, 90)
                
                market['ai_analysis'] = {
                    'predicted_probability': prob,
                    'recommendation': 'YES' if prob > 50 else 'NO',
                    'confidence_score': 60.0,
                    'opportunity_score': opportunity_score,
                    'divergence': 0,
                    'reasoning': f'高交易量({volume:,.0f}) + 明显概率偏差({prob:.1f}%)',
                    'key_factors': ['高交易量', '概率偏差']
                }
                opportunities.append(market)
        
        # 按机会评分排序
        opportunities.sort(
            key=lambda x: x.get('ai_analysis', {}).get('opportunity_score', 0),
            reverse=True
        )
        
        return opportunities[:max_opportunities]
    
    def save_batch_analysis(self, markets: List[Dict]):
        """保存批量分析结果到数据库"""
        try:
            with get_db_connection() as db:
                cur = db.cursor()
                
                for market in markets:
                    market_id = market.get('market_id')
                    ai_analysis = market.get('ai_analysis')
                    
                    if not market_id or not ai_analysis:
                        continue
                    
                    try:
                        # 先删除该市场的旧分析记录（user_id为NULL的通用分析）
                        cur.execute("""
                            DELETE FROM qd_polymarket_ai_analysis
                            WHERE market_id = %s AND user_id IS NULL
                        """, (market_id,))
                        
                        # 插入新的分析记录
                        cur.execute("""
                            INSERT INTO qd_polymarket_ai_analysis
                            (market_id, user_id, ai_predicted_probability, market_probability,
                             divergence, recommendation, confidence_score, opportunity_score,
                             reasoning, key_factors, related_assets, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """, (
                            market_id,
                            None,  # 通用分析
                            float(ai_analysis.get('predicted_probability', market.get('current_probability', 50.0))),
                            market.get('current_probability', 50.0),
                            float(ai_analysis.get('divergence', 0)),
                            ai_analysis.get('recommendation', 'HOLD'),
                            ai_analysis.get('confidence_score', 0),
                            ai_analysis.get('opportunity_score', 0),
                            ai_analysis.get('reasoning', ''),
                            json.dumps(ai_analysis.get('key_factors', [])),
                            []
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to save analysis for market {market_id}: {e}")
                        continue
                
                db.commit()
                cur.close()
                logger.info(f"Saved batch analysis for {len(markets)} markets")
                
        except Exception as e:
            logger.error(f"Failed to save batch analysis: {e}", exc_info=True)
