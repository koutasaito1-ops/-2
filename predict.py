"""
競馬予想メインスクリプト

使い方:
  # 1. 過去データをCSVから読み込んでモデル学習 → 指定レースを予想
  python predict.py --race-id 202405050811 --data race_data.csv

  # 2. サンプルデータでデモ実行
  python predict.py --demo

  # 3. データ収集 (netkeiba)
  python predict.py --collect --year 2024 --place 05 --save race_data.csv
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from model import HorseRacingPredictor, print_prediction
from scraper import NetkeibaScaper, collect_race_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# デモ用サンプルデータ
# ---------------------------------------------------------------------------

SAMPLE_HISTORY = pd.DataFrame(
    {
        "race_id": ["r001"] * 8 + ["r002"] * 8,
        "finish_order": [1, 2, 3, 4, 5, 6, 7, 8] * 2,
        "horse_name": [
            "ディープインパクト", "オルフェーヴル", "ゴールドシップ", "キズナ",
            "ジェンティルドンナ", "ウオッカ", "ダイワスカーレット", "ブエナビスタ",
            "コントレイル", "グランアレグリア", "アーモンドアイ", "フィエールマン",
            "クロノジェネシス", "リスグラシュー", "スワーヴリチャード", "キセキ",
        ],
        "jockey": [
            "武豊", "池添謙一", "内田博幸", "佐藤哲三",
            "岩田康誠", "四位洋文", "安藤勝己", "横山典弘",
            "福永祐一", "ルメール", "ルメール", "池添謙一",
            "北村友一", "武豊", "デムーロ", "横山典弘",
        ],
        "burden_weight": [57.0] * 16,
        "odds": [
            2.1, 3.5, 5.0, 8.0, 12.0, 18.0, 25.0, 40.0,
            1.8, 2.5, 4.0, 6.5, 10.0, 15.0, 22.0, 35.0,
        ],
        "popularity": [1, 2, 3, 4, 5, 6, 7, 8] * 2,
        "last_3f": [34.5] * 16,
        "weight_kg": [480.0] * 16,
        "weight_diff": [0.0] * 16,
    }
)

# 追加過去成績 (同じ馬が複数レース出走)
_extra_rows = []
horse_results = {
    "ディープインパクト": [1, 1, 2, 1, 1, 3],
    "オルフェーヴル":     [2, 1, 1, 2, 4, 1],
    "コントレイル":       [1, 1, 1, 2, 1, 1],
    "グランアレグリア":   [1, 2, 1, 1, 3, 1],
    "アーモンドアイ":     [1, 1, 2, 1, 1, 2],
}
for horse, orders in horse_results.items():
    for i, order in enumerate(orders):
        _extra_rows.append({
            "race_id": f"extra_{horse}_{i}",
            "finish_order": order,
            "horse_name": horse,
            "jockey": "武豊",
            "burden_weight": 57.0,
            "odds": 2.0 + order,
            "popularity": order,
            "last_3f": 34.0,
            "weight_kg": 480.0,
            "weight_diff": 0.0,
        })

SAMPLE_HISTORY = pd.concat(
    [SAMPLE_HISTORY, pd.DataFrame(_extra_rows)], ignore_index=True
)


SAMPLE_RACE = pd.DataFrame(
    {
        "horse_number": list(range(1, 9)),
        "horse_name": [
            "コントレイル", "グランアレグリア", "アーモンドアイ", "ディープインパクト",
            "フィエールマン", "クロノジェネシス", "キセキ", "スワーヴリチャード",
        ],
        "jockey": [
            "福永祐一", "ルメール", "ルメール", "武豊",
            "池添謙一", "北村友一", "横山典弘", "デムーロ",
        ],
        "burden_weight": [57.0, 55.0, 55.0, 57.0, 57.0, 55.0, 57.0, 57.0],
        "odds": [3.2, 4.5, 5.0, 6.0, 8.5, 10.0, 18.0, 25.0],
        "weight_diff": [0, +2, -4, 0, +6, -2, 0, +4],
    }
)


# ---------------------------------------------------------------------------
# コマンド実装
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """サンプルデータで予想デモを実行する"""
    print("\n" + "=" * 60)
    print("  競馬予想デモ  (サンプルデータ使用)")
    print("  レース条件: 芝 2000m")
    print("=" * 60)

    predictor = HorseRacingPredictor(market_weight=0.45, stat_weight=0.55)
    predictor.fit(SAMPLE_HISTORY)

    results = predictor.predict(SAMPLE_RACE, surface="芝", distance=2000)
    print_prediction(results)

    # 馬連ボックス推奨 (上位3頭)
    top3 = [r.horse_name for r in results[:3]]
    print(f"馬連ボックス推奨: {' - '.join(top3)}")

    # 3連複推奨
    print(f"3連複フォーメーション本命軸: {results[0].horse_name}")
    print(f"  相手: {' / '.join(r.horse_name for r in results[1:4])}")
    print()


def run_predict(race_id: str, data_path: str) -> None:
    """CSVデータを読み込みレースを予想する"""
    logger.info("学習データ読み込み: %s", data_path)
    try:
        hist_df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("ファイルが見つかりません: %s", data_path)
        sys.exit(1)

    predictor = HorseRacingPredictor(market_weight=0.45, stat_weight=0.55)
    predictor.fit(hist_df)

    logger.info("出走表取得: race_id=%s", race_id)
    scraper = NetkeibaScaper(interval=2.0)
    race_df = scraper.get_shutuba(race_id)

    if race_df is None or race_df.empty:
        logger.error("出走表の取得に失敗しました")
        sys.exit(1)

    logger.info("出走馬数: %d", len(race_df))
    results = predictor.predict(race_df, surface="芝", distance=2000)
    print_prediction(results)


def run_collect(year: int, place: str, save_path: str) -> None:
    """netkeibaからデータを収集してCSVに保存する"""
    logger.info("データ収集開始: %d年 場所=%s", year, place)
    collect_race_data(
        years=[year],
        place_codes=[place],
        interval=3.0,
        save_path=save_path,
    )
    logger.info("収集完了: %s", save_path)


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="競馬予想システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # demo
    subparsers.add_parser("demo", help="サンプルデータでデモ実行")

    # predict
    p_pred = subparsers.add_parser("predict", help="指定レースを予想")
    p_pred.add_argument("--race-id", required=True, help="netkeibaのレースID")
    p_pred.add_argument("--data", required=True, help="学習用CSVパス")
    p_pred.add_argument("--surface", default="芝", choices=["芝", "ダート"], help="馬場種別")
    p_pred.add_argument("--distance", type=int, default=2000, help="距離(m)")

    # collect
    p_col = subparsers.add_parser("collect", help="netkeibaからデータ収集")
    p_col.add_argument("--year", type=int, required=True, help="収集年")
    p_col.add_argument(
        "--place", default="05",
        help="競馬場コード: 05=東京 06=中山 08=京都 09=阪神 等"
    )
    p_col.add_argument("--save", default="race_data.csv", help="保存先CSVパス")
    p_col.add_argument("--interval", type=float, default=3.0, help="リクエスト間隔(秒)")

    # 後方互換: --demo / --race-id を直接渡す旧形式もサポート
    parser.add_argument("--demo", action="store_true", help="(旧形式) デモ実行")
    parser.add_argument("--race-id", help="(旧形式) レースID")
    parser.add_argument("--data", help="(旧形式) 学習CSVパス")
    parser.add_argument("--collect", action="store_true", help="(旧形式) データ収集")
    parser.add_argument("--year", type=int, help="(旧形式) 収集年")
    parser.add_argument("--place", default="05", help="(旧形式) 競馬場コード")
    parser.add_argument("--save", default="race_data.csv", help="(旧形式) 保存先")

    args = parser.parse_args()

    if args.command == "demo" or getattr(args, "demo", False):
        run_demo()
    elif args.command == "predict":
        run_predict(args.race_id, args.data)
    elif args.command == "collect":
        run_collect(args.year, args.place, args.save)
    elif getattr(args, "race_id", None) and getattr(args, "data", None):
        run_predict(args.race_id, args.data)
    elif getattr(args, "collect", False):
        if not args.year:
            parser.error("--year が必要です")
        run_collect(args.year, args.place, args.save)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
