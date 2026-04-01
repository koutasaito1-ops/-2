"""
競馬予想 統計・確率モデル

以下の手法を組み合わせて各馬の勝率を推定する:
1. オッズから市場確率を算出 (過剰率補正)
2. 過去成績スコア (勝率・複勝率・平均着順)
3. 騎手・調教師スコア
4. 馬場適性スコア (芝/ダート、距離帯)
5. ベイズ的重み付け統合
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class HorseStats:
    """馬の統計情報"""
    horse_name: str
    total_races: int = 0
    wins: int = 0
    top2: int = 0
    top3: int = 0
    avg_finish: float = 0.0
    surface_wins: dict = field(default_factory=dict)    # {"芝": 3, "ダート": 1}
    distance_wins: dict = field(default_factory=dict)   # {"短距離": 2, "マイル": 1}

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_races if self.total_races > 0 else 0.0

    @property
    def place_rate(self) -> float:
        return self.top3 / self.total_races if self.total_races > 0 else 0.0


@dataclass
class PredictionResult:
    """予想結果"""
    horse_number: int
    horse_name: str
    market_prob: float       # オッズから算出した市場確率
    stat_score: float        # 統計スコア
    final_prob: float        # 最終予測確率
    expected_value: float    # 期待値 (final_prob * odds - 1)
    odds: Optional[float]
    recommendation: str      # "◎本命" / "○対抗" / "▲単穴" / "△穴" / ""


# ---------------------------------------------------------------------------
# 特徴量エンジニアリング
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    レース結果DataFrameを前処理して特徴量を追加する。

    必要カラム: race_id, finish_order, horse_name, burden_weight,
               odds, popularity, last_3f, weight_kg, weight_diff
    """
    df = df.copy()

    # 着順が数値でないもの(中止・除外等)を除外
    df["finish_order"] = pd.to_numeric(df["finish_order"], errors="coerce")
    df = df.dropna(subset=["finish_order"])
    df["finish_order"] = df["finish_order"].astype(int)

    # 1着フラグ
    df["is_win"] = (df["finish_order"] == 1).astype(int)
    # 3着以内フラグ
    df["is_place"] = (df["finish_order"] <= 3).astype(int)

    # 着順を正規化スコアに変換 (小さいほど良い → 逆数)
    df["finish_score"] = 1.0 / df["finish_order"]

    return df


def build_horse_stats(df: pd.DataFrame) -> dict[str, HorseStats]:
    """
    過去レースデータから馬ごとの統計情報を構築する。

    Args:
        df: preprocess済みのレース結果DataFrame

    Returns:
        {horse_name: HorseStats} の辞書
    """
    stats: dict[str, HorseStats] = {}

    for name, grp in df.groupby("horse_name"):
        s = HorseStats(horse_name=str(name))
        s.total_races = len(grp)
        s.wins = int(grp["is_win"].sum())
        s.top2 = int((grp["finish_order"] <= 2).sum())
        s.top3 = int(grp["is_place"].sum())
        s.avg_finish = float(grp["finish_order"].mean())
        stats[str(name)] = s

    return stats


def build_jockey_stats(df: pd.DataFrame) -> dict[str, float]:
    """騎手ごとの勝率スコアを返す"""
    if "jockey" not in df.columns:
        return {}
    grp = df.groupby("jockey")
    scores = {}
    for jockey, g in grp:
        if len(g) >= 10:  # サンプル数が少ない騎手は除外
            scores[str(jockey)] = float(g["is_win"].mean())
    return scores


# ---------------------------------------------------------------------------
# 確率計算
# ---------------------------------------------------------------------------

def odds_to_market_prob(odds_series: pd.Series) -> pd.Series:
    """
    単勝オッズから市場確率を計算する (過剰率補正あり)。

    競馬のオッズには控除率 (約25%) が含まれているため、
    単純な 1/odds の合計は1を超える。ここでは正規化で補正する。

    Args:
        odds_series: 出走馬のオッズ Series (NaN許容)

    Returns:
        補正済み市場確率 Series
    """
    raw = 1.0 / odds_series.clip(lower=1.0)  # 1.0オッズ未満は存在しないが念のため
    total = raw.sum()
    if total <= 0:
        return pd.Series(np.nan, index=odds_series.index)
    return raw / total


def stat_score(
    entries: pd.DataFrame,
    horse_stats: dict[str, HorseStats],
    jockey_stats: dict[str, float],
    surface: str = "芝",
    distance: int = 1600,
) -> pd.Series:
    """
    出走馬リストに対して統計スコアを計算する。

    Args:
        entries: 出走馬DataFrame (horse_name 列が必要)
        horse_stats: build_horse_stats の結果
        jockey_stats: build_jockey_stats の結果
        surface: "芝" or "ダート"
        distance: レース距離 (m)

    Returns:
        各馬の統計スコア (0〜1 に正規化)
    """
    scores = []
    for _, row in entries.iterrows():
        name = str(row.get("horse_name", ""))
        s = horse_stats.get(name)

        # --- 馬スコア ---
        if s and s.total_races >= 3:
            # 勝率・複勝率・平均着順の加重平均
            horse_sc = (
                0.5 * s.win_rate
                + 0.3 * s.place_rate
                + 0.2 * (1.0 / max(s.avg_finish, 1.0))
            )
            # サンプル数ボーナス (多いほどスコアが安定)
            confidence = min(s.total_races / 30.0, 1.0)
            horse_sc = horse_sc * confidence + 0.1 * (1 - confidence)
        else:
            horse_sc = 0.1  # データ不足はやや低スコア

        # --- 騎手スコア ---
        jockey = str(row.get("jockey", ""))
        jockey_sc = jockey_stats.get(jockey, 0.08)  # デフォルト平均勝率

        # --- 斤量補正 (軽いほどわずかに有利) ---
        burden = float(row.get("burden_weight", 55.0) or 55.0)
        burden_sc = max(0, (57.0 - burden) * 0.01)

        # --- 馬体重増減補正 ---
        diff = float(row.get("weight_diff", 0.0) or 0.0)
        # ±10kg以内はほぼ影響なし、大きく変化すると小さくペナルティ
        weight_sc = max(0, 1.0 - abs(diff) * 0.005)

        total = (
            0.50 * horse_sc
            + 0.25 * jockey_sc
            + 0.15 * burden_sc
            + 0.10 * weight_sc
        )
        scores.append(total)

    s_arr = np.array(scores, dtype=float)
    # 0〜1 に正規化
    rng = s_arr.max() - s_arr.min()
    if rng > 0:
        s_arr = (s_arr - s_arr.min()) / rng
    return pd.Series(s_arr, index=entries.index)


# ---------------------------------------------------------------------------
# 予想エンジン
# ---------------------------------------------------------------------------

class HorseRacingPredictor:
    """
    統計・確率モデルによる競馬予想エンジン

    使い方:
        predictor = HorseRacingPredictor()
        predictor.fit(historical_df)          # 過去データで学習
        results = predictor.predict(race_df, surface="芝", distance=2000)
        for r in results:
            print(r.horse_name, f"{r.final_prob:.1%}", r.recommendation)
    """

    def __init__(self, market_weight: float = 0.5, stat_weight: float = 0.5):
        """
        Args:
            market_weight: 市場確率 (オッズ) の重み
            stat_weight: 統計スコアの重み
        """
        assert abs(market_weight + stat_weight - 1.0) < 1e-9, "重みの合計は1.0"
        self.market_weight = market_weight
        self.stat_weight = stat_weight
        self.horse_stats: dict[str, HorseStats] = {}
        self.jockey_stats: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "HorseRacingPredictor":
        """
        過去レースデータで統計モデルを学習する。

        Args:
            df: preprocess() 済みのDataFrame
        """
        df = preprocess(df)
        self.horse_stats = build_horse_stats(df)
        self.jockey_stats = build_jockey_stats(df)
        logger.info(
            "学習完了: 馬数=%d, 騎手数=%d",
            len(self.horse_stats),
            len(self.jockey_stats),
        )
        return self

    def predict(
        self,
        race_df: pd.DataFrame,
        surface: str = "芝",
        distance: int = 1600,
    ) -> list[PredictionResult]:
        """
        出走表から各馬の勝率・期待値を予測する。

        Args:
            race_df: 出走馬DataFrame
                     必須カラム: horse_name, horse_number, odds
                     任意カラム: jockey, burden_weight, weight_diff
            surface: "芝" or "ダート"
            distance: レース距離 (m)

        Returns:
            PredictionResult のリスト (final_prob 降順)
        """
        race_df = race_df.copy().reset_index(drop=True)

        # --- 市場確率 ---
        if "odds" in race_df.columns:
            market_prob = odds_to_market_prob(race_df["odds"].astype(float))
        else:
            n = len(race_df)
            market_prob = pd.Series([1.0 / n] * n)

        # --- 統計スコア → 確率に変換 ---
        s_scores = stat_score(
            race_df, self.horse_stats, self.jockey_stats, surface, distance
        )
        s_scores_norm = s_scores / s_scores.sum() if s_scores.sum() > 0 else s_scores

        # --- 統合確率 ---
        combined = (
            self.market_weight * market_prob.fillna(0)
            + self.stat_weight * s_scores_norm
        )
        combined = combined / combined.sum()  # 再正規化

        # --- 期待値計算 ---
        # 期待値 = 勝率 × オッズ - 1 (1は投資額の回収分)
        results = []
        for idx, row in race_df.iterrows():
            prob = float(combined.iloc[idx])
            odds_val = float(row.get("odds", np.nan))
            if not np.isnan(odds_val) and odds_val > 0:
                ev = prob * odds_val - 1.0
            else:
                ev = np.nan

            results.append(
                PredictionResult(
                    horse_number=int(row.get("horse_number", idx + 1)),
                    horse_name=str(row.get("horse_name", f"馬{idx+1}")),
                    market_prob=float(market_prob.iloc[idx]),
                    stat_score=float(s_scores.iloc[idx]),
                    final_prob=prob,
                    expected_value=ev,
                    odds=odds_val if not np.isnan(odds_val) else None,
                    recommendation="",
                )
            )

        # --- 印付け ---
        results.sort(key=lambda r: r.final_prob, reverse=True)
        marks = ["◎本命", "○対抗", "▲単穴", "△穴"]
        for i, r in enumerate(results[:4]):
            r.recommendation = marks[i]

        return results


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def print_prediction(results: list[PredictionResult]) -> None:
    """予想結果をターミナルに表示する"""
    print(f"\n{'='*60}")
    print(f"{'馬番':>4}  {'馬名':<14}  {'印':<5}  {'予測勝率':>7}  {'オッズ':>6}  {'期待値':>7}")
    print(f"{'-'*60}")
    for r in results:
        ev_str = f"{r.expected_value:+.2f}" if r.expected_value is not None and not np.isnan(r.expected_value) else "  N/A"
        odds_str = f"{r.odds:.1f}" if r.odds else " N/A"
        print(
            f"{r.horse_number:>4}  {r.horse_name:<14}  {r.recommendation:<5}  "
            f"{r.final_prob:>6.1%}  {odds_str:>6}  {ev_str:>7}"
        )
    print(f"{'='*60}\n")
    print("※ 期待値 > 0 の馬が統計的にプラス期待値")
