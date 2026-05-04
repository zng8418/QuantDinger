"""
内部信号桥接路由 — 供 Smart Trading V10 调用
认证方式：X-Internal-Key header = INTERNAL_API_KEY
"""
import os
from flask import Blueprint, request, jsonify
from app.utils.logger import get_logger

logger = get_logger(__name__)
internal_bp = Blueprint('internal', __name__)


def _check_internal_key():
    """验证内部API Key"""
    try:
        from app.utils.config_loader import get_internal_api_key
        expected = get_internal_api_key() or ''
    except Exception:
        expected = os.getenv('INTERNAL_API_KEY', '').strip()
    provided = request.headers.get('X-Internal-Key', '').strip()
    if not expected or provided != expected:
        return False
    return True


@internal_bp.route('/api/internal/polymarket/crypto-signals', methods=['GET'])
def polymarket_crypto_signals():
    """
    获取Polymarket加密货币相关AI分析信号
    
    GET /api/internal/polymarket/crypto-signals
    Headers: X-Internal-Key: <INTERNAL_API_KEY>
    Query: ?limit=10&min_score=60&direction=long|short|all
    
    返回: {
        code: 0,
        data: {
            timestamp: "...",
            signals: [
                {
                    market_id, question, category,
                    market_prob, ai_prob, divergence,
                    recommendation, confidence, opportunity_score,
                    reasoning, related_assets
                }
            ],
            summary: {
                total_markets, bullish_count, bearish_count, neutral_count,
                avg_confidence, top_signal
            }
        }
    }
    """
    if not _check_internal_key():
        return jsonify({'code': 401, 'msg': 'Invalid internal key'}), 401

    try:
        from app.utils.db import get_db_connection
        limit = min(int(request.args.get('limit', 10)), 50)
        min_score = float(request.args.get('min_score', 50))
        direction = request.args.get('direction', 'all')

        with get_db_connection() as conn:
            cur = conn.cursor()
            # 查询crypto类别 + 有AI分析的市场
            sql = """
                SELECT 
                    m.market_id,
                    m.question,
                    m.category,
                    m.current_probability as market_prob,
                    m.volume_24h,
                    m.liquidity,
                    a.ai_predicted_probability as ai_prob,
                    a.divergence,
                    a.recommendation,
                    a.confidence_score as confidence,
                    a.opportunity_score,
                    a.reasoning,
                    a.related_assets,
                    a.key_factors,
                    m.updated_at
                FROM qd_polymarket_markets m
                INNER JOIN qd_polymarket_ai_analysis a ON m.market_id = a.market_id
                WHERE m.category = 'crypto'
                  AND m.status = 'active'
                  AND a.opportunity_score >= %s
            """
            params = [min_score]

            # 方向过滤
            if direction == 'long':
                sql += " AND a.recommendation = 'YES'"
            elif direction == 'short':
                sql += " AND a.recommendation = 'NO'"

            # 排除过期市场
            sql += " AND (m.end_date_iso IS NULL OR m.end_date_iso > NOW())"

            # 去重：每个market只取最新分析
            sql += " ORDER BY a.opportunity_score DESC, a.created_at DESC"

            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()

        # 去重（同一market可能有多次分析）
        seen = set()
        signals = []
        for d in rows:
            mid = d['market_id']
            if mid in seen:
                continue
            seen.add(mid)

            # 格式化概率
            mp = float(d['market_prob'] or 0)
            ap = float(d['ai_prob'] or 0)
            div = float(d['divergence'] or 0)
            conf = float(d['confidence'] or 0)
            opp = float(d['opportunity_score'] or 0)

            # 推断方向性信号
            rec = d['recommendation'] or 'HOLD'

            # 提取关联资产
            related = d['related_assets'] or []
            if isinstance(related, str):
                related = [r.strip() for r in related.split(',') if r.strip()]

            signals.append({
                'market_id': mid,
                'question': d['question'] or '',
                'category': d['category'] or 'crypto',
                'market_prob': round(mp, 2),
                'ai_prob': round(ap, 2),
                'divergence': round(div, 2),
                'recommendation': rec,
                'direction': 'bullish' if rec == 'YES' else ('bearish' if rec == 'NO' else 'neutral'),
                'confidence': round(conf, 2),
                'opportunity_score': round(opp, 2),
                'volume_24h': float(d['volume_24h'] or 0),
                'liquidity': float(d['liquidity'] or 0),
                'reasoning': d['reasoning'] or '',
                'related_assets': related,
                'key_factors': d.get('key_factors'),
                'updated_at': str(d.get('updated_at', '')),
            })

            if len(signals) >= limit:
                break

        # 生成摘要
        bullish = sum(1 for s in signals if s['direction'] == 'bullish')
        bearish = sum(1 for s in signals if s['direction'] == 'bearish')
        neutral = len(signals) - bullish - bearish
        avg_conf = sum(s['confidence'] for s in signals) / max(len(signals), 1)
        top = signals[0] if signals else None

        # 生成综合方向性信号
        if bullish > bearish + neutral:
            overall = 'bullish'
        elif bearish > bullish + neutral:
            overall = 'bearish'
        else:
            overall = 'neutral'

        return jsonify({
            'code': 0,
            'data': {
                'timestamp': signals[0]['updated_at'] if signals else None,
                'total_analyzed': len(signals),
                'overall_signal': overall,
                'signals': signals[:limit],
                'summary': {
                    'total_markets': len(signals),
                    'bullish_count': bullish,
                    'bearish_count': bearish,
                    'neutral_count': neutral,
                    'avg_confidence': round(avg_conf, 2),
                    'top_signal': top,
                }
            }
        })

    except Exception as e:
        logger.error(f"Internal polymarket signals error: {e}")
        return jsonify({'code': 500, 'msg': str(e)}), 500


@internal_bp.route('/api/internal/fast-analysis', methods=['POST'])
def fast_analysis():
    """
    快速AI分析 — 供Smart Trading调用
    
    POST /api/internal/fast-analysis
    Headers: X-Internal-Key: <INTERNAL_API_KEY>
    Body: { "symbol": "BTC_USDT", "market": "crypto", "language": "zh-CN" }
    """
    if not _check_internal_key():
        return jsonify({'code': 401, 'msg': 'Invalid internal key'}), 401

    try:
        from app.services.fast_analysis import FastAnalysisService
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTC_USDT')
        market = data.get('market', 'crypto')
        language = data.get('language', 'zh-CN')

        service = FastAnalysisService()
        result = service.analyze(symbol=symbol, market=market, language=language)

        return jsonify({
            'code': 0,
            'data': result
        })

    except Exception as e:
        logger.error(f"Internal fast analysis error: {e}")
        return jsonify({'code': 500, 'msg': str(e)}), 500
