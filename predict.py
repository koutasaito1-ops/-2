"""
競馬予想メインスクリプト

使い方:
  python predict.py demo          # サンプルデータでデモ
  python predict.py nakagyo       # 中京12R (スマート出走表データ込み)
  python predict.py weights       # 現在の重み設定を表示
  python predict.py predict --race-id 202405050811 --data race_data.csv
  python predict.py collect --year 2024 --place 05 --save race_data.csv
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from model import (
    CourseStats,
    CourseTraitScore,
    RaceTrend,
    TendencyRule,
    RaceTendencyRules,
    _safe_int,
    _safe_str,
    HorseRacingPredictor,
    ScoreWeights,
    print_prediction,
    print_bet_suggestions,
)
from scraper import NetkeibaScaper, collect_race_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# サンプル学習データ (馬・騎手の過去成績フォールバック用)
# ---------------------------------------------------------------------------

SAMPLE_HISTORY = pd.DataFrame({
    "race_id":      ["r001"]*8 + ["r002"]*8,
    "finish_order": [1,2,3,4,5,6,7,8]*2,
    "horse_name": [
        "ディープインパクト","オルフェーヴル","ゴールドシップ","キズナ",
        "ジェンティルドンナ","ウオッカ","ダイワスカーレット","ブエナビスタ",
        "コントレイル","グランアレグリア","アーモンドアイ","フィエールマン",
        "クロノジェネシス","リスグラシュー","スワーヴリチャード","キセキ",
    ],
    "jockey": [
        "武豊","池添謙一","内田博幸","佐藤哲三",
        "岩田康誠","四位洋文","安藤勝己","横山典弘",
        "福永祐一","ルメール","ルメール","池添謙一",
        "北村友一","武豊","デムーロ","横山典弘",
    ],
    "burden_weight": [57.0]*16,
    "odds": [2.1,3.5,5.0,8.0,12.0,18.0,25.0,40.0, 1.8,2.5,4.0,6.5,10.0,15.0,22.0,35.0],
    "popularity": [1,2,3,4,5,6,7,8]*2,
    "last_3f": [34.5]*16,
    "weight_kg": [480.0]*16,
    "weight_diff": [0.0]*16,
})

_extra = []
for horse, orders in {
    "ディープインパクト": [1,1,2,1,1,3],
    "オルフェーヴル":     [2,1,1,2,4,1],
    "コントレイル":       [1,1,1,2,1,1],
    "グランアレグリア":   [1,2,1,1,3,1],
    "アーモンドアイ":     [1,1,2,1,1,2],
}.items():
    for i, o in enumerate(orders):
        _extra.append({
            "race_id": f"extra_{horse}_{i}", "finish_order": o,
            "horse_name": horse, "jockey": "武豊",
            "burden_weight": 57.0, "odds": 2.0+o,
            "popularity": o, "last_3f": 34.0,
            "weight_kg": 480.0, "weight_diff": 0.0,
        })
SAMPLE_HISTORY = pd.concat([SAMPLE_HISTORY, pd.DataFrame(_extra)], ignore_index=True)

SAMPLE_RACE = pd.DataFrame({
    "horse_number": list(range(1,9)),
    "horse_name": [
        "コントレイル","グランアレグリア","アーモンドアイ","ディープインパクト",
        "フィエールマン","クロノジェネシス","キセキ","スワーヴリチャード",
    ],
    "jockey": ["福永祐一","ルメール","ルメール","武豊","池添謙一","北村友一","横山典弘","デムーロ"],
    "burden_weight": [57.0,55.0,55.0,57.0,57.0,55.0,57.0,57.0],
    "odds": [3.2,4.5,5.0,6.0,8.5,10.0,18.0,25.0],
    "weight_diff": [0,+2,-4,0,+6,-2,0,+4],
})


# ---------------------------------------------------------------------------
# 中京12R  3歳上1勝クラス  芝1400m  晴/良  2026-03-30
# smartrc.jp より (集計期間 2021-03-31〜2026-03-23)
#
# course_stats の構造:
#   {horse_name: {category: CourseStats(name, category, total, wins, top2, top3, win_return%, place_return%)}}
#
# category 一覧:
#   "父"          父馬コース
#   "母父"        母父コース
#   "騎手"        騎手コース
#   "調教師"      調教師コース
#   "父小系統"    父の小系統コース
#   "母父小系統"  母父の小系統コース
#   "父国×母父国" 父国×母父国コース
# ---------------------------------------------------------------------------

def _cs(name, cat, total, wins, top2, top3, win_ret, place_ret) -> CourseStats:
    """CourseStats 生成ショートハンド"""
    return CourseStats(
        name=name, category=cat,
        total_races=total, wins=wins, top2=top2, top3=top3,
        win_return=win_ret, place_return=place_ret,
    )


# ---------------------------------------------------------------------------
# 中京芝1400m コース特性補正 (重賞ナビ・過去統計より)
#
# 参考: 2021〜2025年 中京芝1400m 集計
#   人気別: 1番人気 単回68% (オーバーベット), 5番人気 単回125% (バリュー)
#   脚質別: 差し優勢 (複回収+), 逃げ割引
#   枠番別: 4〜6枠好成績, 8枠割引
#   馬体重: 480kg以上好成績
#   前走距離: 同距離→1400m 好成績, 1200m→1400m 割引
# ---------------------------------------------------------------------------

CHUKYO_SHIBA_1400_TRAIT = CourseTraitScore(
    course_name="中京芝1400m",
    popularity_adj={
        1:  0.90,   # 1番人気: オーバーベット気味 (単回68%)
        2:  0.97,
        3:  1.03,
        4:  1.07,
        5:  1.15,   # 5番人気: バリューゾーン (単回125%)
        6:  1.10,
        7:  1.05,
        8:  1.00,
        9:  0.98,
        10: 0.95,
        # 11番人気以降はデフォルト 1.0
    },
    running_style_adj={
        "逃げ":  0.70,   # 差し馬場 → 逃げ不利
        "先行":  0.95,
        "差し":  1.20,   # 差し優勢
        "追込":  0.90,
    },
    gate_adj={
        1: 1.00,
        2: 1.02,
        3: 1.05,
        4: 1.15,   # 4〜6枠優秀
        5: 1.15,
        6: 1.15,
        7: 1.00,
        8: 0.85,   # 8枠割引
    },
    weight_band_adj={
        "~439":    0.90,
        "440~479": 1.00,
        "480~519": 1.10,   # 480kg以上好成績
        "520+":    1.05,
    },
    prev_distance_adj={
        "同距離": 1.15,   # 1400m→1400m 好成績
        "短縮":   0.90,   # 1600m以上→1400m やや割引
        "延長":   0.95,   # 1200m→1400m 割引
        "1200→1400": 0.85,  # 1200m特定
    },
)


# ---------------------------------------------------------------------------
# チャーチルダウンズC 傾向ルール (重賞ナビ データまとめ)
#
# ── プラスデータ ──
#   母父サンデー系 / 欧州型ノーザンダンサー系 / ナスルーラ系
#   3代目に米国血統 + 4代目までにナスルーラ系持ち
#   差し馬 (前走4角6番手以下)
#   前走が重賞だった1番人気馬
#   前走でルメール騎手が騎乗していた馬
#   2月生まれ
# ── マイナスデータ ──
#   8枠
#   前走オープン特別以下で4角5番手以内 (前走2番人気以内を除く)
#   当日5番人気以内 + 当日プラス体重
#   関東馬
#   キャリア2戦以下
#   前走1勝クラス以下で4番人気以下 (1勝クラスは+前走3着以下)
# ---------------------------------------------------------------------------

# 母父系統判定用セット
_SUNDAY_LINES      = {"サンデーサイレンス系", "サンデー系", "ディープインパクト系", "ハーツクライ系", "ステイゴールド系"}
_EU_ND_LINES       = {"欧州型ノーザンダンサー系", "ノーザンダンサー系", "サドラーズウェルズ系", "ガリレオ系", "ダンジグ系", "ニジンスキー系"}
_NASRULLAH_LINES   = {"ナスルーラ系", "ロベルト系", "ブラッシンググルーム系", "グレイソヴリン系"}

_PLUS_BLOOD_LINES  = _SUNDAY_LINES | _EU_ND_LINES | _NASRULLAH_LINES


def _is_plus_bloodline(row) -> bool:
    bml = _safe_str(row, "broodmare_sire_line")
    return bml in _PLUS_BLOOD_LINES


def _is_sashi(row) -> bool:
    pos = _safe_int(row, "prev_4corner_pos", 0)
    return pos >= 6


def _is_prev_graded_fav(row) -> bool:
    pcls = _safe_str(row, "prev_race_class")
    ppop = _safe_int(row, "prev_popularity", 99)
    return pcls in {"G1", "G2", "G3"} and ppop == 1


def _had_lemaire(row) -> bool:
    return _safe_str(row, "prev_jockey") == "ルメール"


def _born_february(row) -> bool:
    return _safe_int(row, "birth_month", 0) == 2


def _is_gate8(row) -> bool:
    hn = _safe_int(row, "horse_number", 0)
    return CourseTraitScore._gate_from_horse_number(hn) == 8


def _minus_prev_op_senko(row) -> bool:
    """前走OP特別以下で4角5番手以内 (前走2番人気以内は除外)"""
    pcls = _safe_str(row, "prev_race_class")
    p4c  = _safe_int(row, "prev_4corner_pos", 99)
    ppop = _safe_int(row, "prev_popularity", 99)
    is_below_op = pcls in {"OP", "3勝", "2勝", "1勝", "未勝利"}
    is_senko = p4c is not None and p4c <= 5
    is_fav   = ppop is not None and ppop <= 2
    return is_below_op and is_senko and not is_fav


def _minus_fav_plus_weight(row) -> bool:
    """当日5番人気以内 + 当日プラス体重"""
    pop  = _safe_int(row, "popularity", 99)
    wdif = _safe_int(row, "weight_diff", 0)
    return pop is not None and pop <= 5 and wdif is not None and wdif > 0


def _is_kanto_horse(row) -> bool:
    return _safe_str(row, "training_region") == "関東"


def _low_career(row) -> bool:
    return (_safe_int(row, "career_races", 99) or 99) <= 2


def _minus_prev_1sho_outsider(row) -> bool:
    """前走1勝クラス以下で4番人気以下 (1勝クラスは+前走3着以下)"""
    pcls = _safe_str(row, "prev_race_class")
    ppop = _safe_int(row, "prev_popularity", 99)
    porder = _safe_int(row, "prev_finish_order", 0)
    if pcls == "1勝":
        return ppop >= 4 and (porder is None or porder >= 4)
    elif pcls in {"未勝利"}:
        return ppop is not None and ppop >= 4
    return False


CHURCHILLDOWNS_C_RULES = RaceTendencyRules(
    "チャーチルダウンズC",
    [
        # ── プラスデータ ────────────────────────────────────
        TendencyRule("母父血統優秀",      1.20, _is_plus_bloodline,      is_plus=True,
                     description="母父サンデー系/欧州ND系/ナスルーラ系"),
        TendencyRule("差し馬",           1.18, _is_sashi,               is_plus=True,
                     description="前走4角6番手以下"),
        TendencyRule("前走重賞1番人気",   1.15, _is_prev_graded_fav,    is_plus=True,
                     description="前走が重賞かつ1番人気"),
        TendencyRule("前走ルメール騎乗",  1.12, _had_lemaire,           is_plus=True,
                     description="前走でルメール騎手が騎乗"),
        TendencyRule("2月生まれ",         1.10, _born_february,         is_plus=True,
                     description="誕生月が2月"),
        # ── マイナスデータ ──────────────────────────────────
        TendencyRule("8枠",              0.80, _is_gate8,               is_plus=False,
                     description="枠番8枠"),
        TendencyRule("前走OP以下先行",    0.82, _minus_prev_op_senko,   is_plus=False,
                     description="前走OP以下で4角5番手以内(2番人気以内除く)"),
        TendencyRule("人気馬プラス体重",  0.83, _minus_fav_plus_weight, is_plus=False,
                     description="当日5番人気以内+当日プラス体重"),
        TendencyRule("関東馬",           0.85, _is_kanto_horse,        is_plus=False,
                     description="関東所属馬"),
        TendencyRule("キャリア浅い",      0.82, _low_career,            is_plus=False,
                     description="キャリア2戦以下"),
        TendencyRule("前走1勝穴馬",       0.83, _minus_prev_1sho_outsider, is_plus=False,
                     description="前走1勝クラス以下で4番人気以下"),
    ]
)


NAKAGYO_12R_RACE = pd.DataFrame({
    "horse_number": list(range(1, 19)),
    "horse_name": [
        "ジャンヌローサ", "ナムライリス", "イリフィ", "レイザリオ",
        "ディスタントスカイ", "エクストラバック", "スピリットライズ", "チムグクル",
        "バンディート", "ノボリリア", "ワイルデンウーリー", "メイショウタマユラ",
        "レオンバローズ", "ショウナンラウール", "ピアストイヤーズ", "ヴァージル",
        "コモンスナイプ", "ヒルノセビリア",
    ],
    "sex_age": [
        "牝4","牝4","牝3","牡3","牝4","牝4","騙3","牡3",
        "牡5","牡3","牝3","牝3","牝6","牝4","牡3","牡3","牡3","牝4",
    ],
    "popularity": [10,18,3,12,9,5,4,2,6,14,7,16,13,15,8,1,11,17],
    "odds": [
        15.8, 105.5, 11.5, 29.3, 15.0, 9.6, 24.0, 4.8,
        9.2,  43.3,  16.3, 51.2, 36.5, 19.3, 32.1, 3.3,
        53.5, 247.4,
    ],
    # ── 展開系データ (判明分のみ; 不明は None のまま) ──────────────────────
    # ten_pattern / agari_pattern: 15=先行/末脚優秀, 30=中位, 50=後方/末脚凡庸
    # prev_margin: 前走着差(秒) 正=着けた差, 負=着けられた差
    # race_interval_weeks: 前走からの経過週数
    "ten_pattern":         [None,None, 30,None,None, 15,None, 15, 30,None,None,None,None,None, 50, 15,None,None],
    "agari_pattern":       [None,None, 15,None,None, 30,None, 30, 30,None, 15,None,None,None, 30, 15,None,None],
    "prev_margin":         [None,-0.5,-0.2,-0.8,None,-0.1,None,+0.3,-0.4,None,-0.3,None,-0.6,None,-0.5,+0.2,None,None],
    "race_interval_weeks": [   4,   4,   5,   4,   5,   4,   3,   4,   5,   4,   4,   8,   4,   4,   4,   4,   5,   4],
    # ── コース特性補正用 追加列 (判明分のみ; 不明は None) ──────────────────────
    # running_style: 脚質 ("逃げ"/"先行"/"差し"/"追込")
    # prev_distance_cat: 前走距離カテゴリ ("同距離"/"短縮"/"延長")
    # horse_weight: 馬体重 kg (出走当日)
    "running_style":    [None,None,"差し",None,None,"先行",None,"先行","先行",None,"差し",None,None,None,"差し","先行",None,None],
    "prev_distance_cat":[None,None,"同距離",None,None,"同距離",None,"同距離","延長",None,"同距離",None,"延長",None,"同距離","同距離",None,None],
    "horse_weight":     [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None],
    # ── 重賞ナビ傾向ルール用追加列 ─────────────────────────────────────────────
    # prev_4corner_pos: 前走4角通過順位
    # prev_race_class: 前走クラス ("G1"/"G2"/"G3"/"OP"/"3勝"/"2勝"/"1勝"/"未勝利")
    # prev_popularity: 前走人気
    # prev_jockey: 前走騎手
    # broodmare_sire_line: 母父系統
    # birth_month: 生まれ月 (1〜12)
    # training_region: 所属 ("関東"/"関西")
    # career_races: キャリア戦数
    # prev_finish_order: 前走着順
    "prev_4corner_pos": [None, 8,   5,  10, None,  3, None,  2,   6, None,  7, None,  8, None,  9,  2, None, None],
    "prev_race_class":  [None,"1勝","1勝","1勝",None,"1勝",None,"1勝","OP",None,"1勝",None,"2勝",None,"1勝","1勝",None,None],
    "prev_popularity":  [None,  3,   2,   5, None,  1, None,  2,   3, None,  4, None,  2, None,  5,  3, None, None],
    "prev_jockey":      [None,None,None,None,None,"ルメール",None,None,None,None,None,None,None,None,None,"川田将雅",None,None],
    "broodmare_sire_line": [None,None,"サンデー系",None,None,None,None,"ノーザンダンサー系",None,None,"サンデー系",None,None,None,"サンデー系","サンデー系",None,None],
    "birth_month":      [None, None, None, 2, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    "training_region":  ["関西","関西","関西","関西","関東","関西","関西","関西","関西","関西","関西","関西","関西","関西","関西","関西","関東","関西"],
    "career_races":     [  20,   15,   5,   6,  18,   10,   3,    8,   12,   4,    9,   16,   22,   18,   7,    8,   5,   14],
    "prev_finish_order":[None,   2,   3,   5, None,   1, None,   1,    4, None,   3, None,   1, None,   4,  1, None, None],
    "weight_diff":      [   0,   0,  -2,  +4,   0,   0,   0,   0,   -2,   0,  +2,   0,   0,   0,  +2,  0,   0,   0],
})


# ── コースデータ (スマート出走表 集計項目) ───────────────────────────────────
#
# 現在入力済み: "父" (画像から読み取り済み)
# 追加予定: "母父" / "騎手" / "調教師" / "父小系統" / "母父小系統" / "父国×母父国"
#   → スマート出走表で各行を確認してデータを追記してください
#
# 未入力カテゴリはリーグ平均値として処理されます (減点なし)
# ---------------------------------------------------------------------------

NAKAGYO_12R_COURSE: dict[str, dict[str, CourseStats]] = {
    "ジャンヌローサ": {
        "父": _cs("ベーカバド",           "父",   4,  0,  0,  1,   0.0,  53.0),
        # "母父":    _cs("...", "母父", ...) ← スマート出走表で確認して追記
        # "騎手":    _cs("...", "騎手", ...)
        # "調教師":  _cs("...", "調教師", ...)
        # "父小系統": _cs("...", "父小系統", ...)
        # "母父小系統": _cs("...", "母父小系統", ...)
        # "父国×母父国": _cs("...", "父国×母父国", ...)
    },
    "ナムライリス": {
        "父": _cs("クロフネ",             "父",  55,  4,  7,  8,  56.0,  39.0),
    },
    "イリフィ": {
        "父": _cs("Invincible Spirit",   "父",  10,  1,  2,  2,  73.0,  42.0),
    },
    "レイザリオ": {
        "父": _cs("Tapit",               "父",  18,  2,  4,  4,  57.0,  49.0),
    },
    "ディスタントスカイ": {
        "父": _cs("Smart Strike",        "父",   6,  0,  0,  0,   0.0,   0.0),
    },
    "エクストラバック": {
        "父": _cs("Frankel",             "父",  10,  1,  1,  2,  96.0,  58.0),
    },
    "スピリットライズ": {
        "父": _cs("High Yield",          "父",   2,  0,  0,  0,   0.0,   0.0),
    },
    "チムグクル": {
        "父": _cs("ディープインパクト",  "父", 152, 19, 31, 47, 181.0, 143.0),
    },
    "バンディート": {
        "父": _cs("Sea The Stars",       "父",  12,  0,  1,  1,   0.0,  37.0),
    },
    "ノボリリア": {
        "父": _cs("ディープインパクト",  "父", 152, 19, 31, 47, 181.0, 143.0),
    },
    "ワイルデンウーリー": {
        "父": _cs("More Than Ready",     "父",   5,  0,  0,  1,   0.0, 106.0),
    },
    "メイショウタマユラ": {
        "父": _cs("ヨハネスブルグ",      "父",   9,  0,  1,  1,   0.0,  21.0),
    },
    "レオンバローズ": {
        "父": _cs("ゼンノロブロイ",      "父",  44,  0,  3,  8,   0.0, 111.0),
    },
    "ショウナンラウール": {
        "父": _cs("クロフネ",            "父",  55,  4,  7,  8,  56.0,  39.0),
    },
    "ピアストイヤーズ": {
        "父": _cs("ディープインパクト",  "父", 152, 19, 31, 47, 181.0, 143.0),
    },
    "ヴァージル": {
        "父": _cs("ダンスインザダーク",  "父",  44,  4,  9, 11, 207.0,  95.0),
    },
    "コモンスナイプ": {
        "父": _cs("Dark Angel",          "父",   4,  1,  1,  1,  55.0,  30.0),
    },
    "ヒルノセビリア": {
        "父": _cs("マンハッタンカフェ",  "父",  32,  1,  3,  7,  24.0, 138.0),
    },
}


# ---------------------------------------------------------------------------
# コマンド実装
# ---------------------------------------------------------------------------

def _make_predictor(mw=0.40, sw=0.60, weights=None) -> HorseRacingPredictor:
    p = HorseRacingPredictor(market_weight=mw, stat_weight=sw, weights=weights)
    p.fit(SAMPLE_HISTORY)
    return p


def run_demo() -> None:
    print("\n" + "="*60)
    print("  競馬予想デモ  (サンプルデータ)")
    print("="*60)
    p = _make_predictor()
    results = p.predict(SAMPLE_RACE, surface="芝", distance=2000)
    print_prediction(results)
    print_bet_suggestions(results)


def run_demo_nakagyo(show_breakdown: bool = False, no_trait: bool = False) -> None:
    print("\n" + "="*85)
    print("  中京 12R  3歳上1勝クラス  芝1400m  晴/良  (2026-03-30)")
    print("  データ: smartrc.jp (2021-03-31〜2026-03-23)  ※父馬コース入力済み")
    trait_label = "コース特性補正: なし" if no_trait else "コース特性補正: 中京芝1400m (重賞ナビ準拠)"
    print(f"  {trait_label}")
    print("="*85)

    p = _make_predictor()
    # 重みを表示
    p.weights.show()

    results = p.predict(
        NAKAGYO_12R_RACE,
        course_stats=NAKAGYO_12R_COURSE,
        surface="芝",
        distance=1400,
        course_trait=None if no_trait else CHUKYO_SHIBA_1400_TRAIT,
    )
    print_prediction(results, show_breakdown=show_breakdown)
    print_bet_suggestions(results)


def run_tendency_demo() -> None:
    """重賞ナビ方式プラス/マイナスルールの適用デモ (中京12Rで例示)"""
    print("\n" + "="*85)
    print("  重賞傾向ルール分析デモ  (チャーチルダウンズC ルールを中京12Rに試適用)")
    print("  ※ 本来は重賞レースのデータで使うルールです。構造の確認用デモです。")
    print("="*85)

    # ルール一覧を表示
    print("\n【プラスデータ】")
    for r in CHURCHILLDOWNS_C_RULES.rules:
        if r.is_plus:
            print(f"  +{r.adjustment:.2f}x  {r.name}  ({r.description})")
    print("\n【マイナスデータ】")
    for r in CHURCHILLDOWNS_C_RULES.rules:
        if not r.is_plus:
            print(f"  {r.adjustment:.2f}x  {r.name}  ({r.description})")

    # 各馬へのルール適用結果
    CHURCHILLDOWNS_C_RULES.print_analysis(NAKAGYO_12R_RACE)

    # tendency_rules を組み込んだ予想
    p = _make_predictor()
    results = p.predict(
        NAKAGYO_12R_RACE,
        course_stats=NAKAGYO_12R_COURSE,
        surface="芝",
        distance=1400,
        course_trait=CHUKYO_SHIBA_1400_TRAIT,
        tendency_rules=CHURCHILLDOWNS_C_RULES,
    )
    print_prediction(results)
    print_bet_suggestions(results)


def run_show_weights() -> None:
    p = _make_predictor()
    p.weights.show()


def run_predict(race_id: str, data_path: str, surface: str, distance: int) -> None:
    logger.info("学習データ読み込み: %s", data_path)
    try:
        hist_df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("ファイルが見つかりません: %s", data_path)
        sys.exit(1)

    p = HorseRacingPredictor()
    p.fit(hist_df)

    scraper = NetkeibaScaper(interval=2.0)
    race_df = scraper.get_shutuba(race_id)
    if race_df is None or race_df.empty:
        logger.error("出走表の取得に失敗しました")
        sys.exit(1)

    results = p.predict(race_df, surface=surface, distance=distance)
    print_prediction(results)
    print_bet_suggestions(results)


def run_collect(year: int, place: str, save_path: str, interval: float) -> None:
    logger.info("データ収集開始: %d年 場所=%s", year, place)
    collect_race_data(years=[year], place_codes=[place], interval=interval, save_path=save_path)


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="競馬予想システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo",    help="サンプルデータでデモ")
    n = sub.add_parser("nakagyo", help="中京12R 父馬コースデータ込みで予想")
    n.add_argument("--breakdown", action="store_true", help="スコア内訳も表示")
    n.add_argument("--no-trait", action="store_true", help="コース特性補正を無効化")
    sub.add_parser("weights",   help="現在の重み設定を表示")
    sub.add_parser("tendency",  help="重賞ナビ傾向ルール適用デモ (チャーチルダウンズC)")

    p_pred = sub.add_parser("predict", help="指定レースを予想")
    p_pred.add_argument("--race-id",  required=True)
    p_pred.add_argument("--data",     required=True)
    p_pred.add_argument("--surface",  default="芝", choices=["芝","ダート"])
    p_pred.add_argument("--distance", type=int, default=2000)

    p_col = sub.add_parser("collect", help="netkeibaからデータ収集")
    p_col.add_argument("--year",     type=int, required=True)
    p_col.add_argument("--place",    default="05")
    p_col.add_argument("--save",     default="race_data.csv")
    p_col.add_argument("--interval", type=float, default=3.0)

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "nakagyo":
        run_demo_nakagyo(
            show_breakdown=getattr(args, "breakdown", False),
            no_trait=getattr(args, "no_trait", False),
        )
    elif args.command == "weights":
        run_show_weights()
    elif args.command == "tendency":
        run_tendency_demo()
    elif args.command == "predict":
        run_predict(args.race_id, args.data, args.surface, args.distance)
    elif args.command == "collect":
        run_collect(args.year, args.place, args.save, args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
