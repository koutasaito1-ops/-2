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


def run_demo_nakagyo(show_breakdown: bool = False) -> None:
    print("\n" + "="*85)
    print("  中京 12R  3歳上1勝クラス  芝1400m  晴/良  (2026-03-30)")
    print("  データ: smartrc.jp (2021-03-31〜2026-03-23)  ※父馬コース入力済み")
    print("="*85)

    p = _make_predictor()
    # 重みを表示
    p.weights.show()

    results = p.predict(
        NAKAGYO_12R_RACE,
        course_stats=NAKAGYO_12R_COURSE,
        surface="芝",
        distance=1400,
    )
    print_prediction(results, show_breakdown=show_breakdown)
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
    sub.add_parser("weights", help="現在の重み設定を表示")

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
        run_demo_nakagyo(show_breakdown=getattr(args, "breakdown", False))
    elif args.command == "weights":
        run_show_weights()
    elif args.command == "predict":
        run_predict(args.race_id, args.data, args.surface, args.distance)
    elif args.command == "collect":
        run_collect(args.year, args.place, args.save, args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
