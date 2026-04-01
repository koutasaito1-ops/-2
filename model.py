"""
競馬予想 統計・確率モデル

スコア構成:
  ┌─ 血統系 ──── 父馬コース / 母父コース / 父小系統コース / 母父小系統コース / 父国×母父国コース
  ├─ 人 ─────── 騎手コース / 調教師コース
  ├─ 馬自身 ──── 過去成績 (勝率・複勝率・平均着順)
  ├─ 展開系 ──── テンP/上がりP / 前走着差 / レース間隔
  └─ フィジカル── 斤量 / 馬体重増減

ScoreWeights で各要素の重みを自由に調整可能。
重みは正規化されるため合計が1.0でなくても動作する。
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

# smartrc テンP/上がりP の値 → スコア変換
_PATTERN_SCORE = {15: 1.0, 30: 0.6, 50: 0.3}
_PATTERN_DEFAULT = 0.5  # パターン未記録 (中間)

# コース別リーグ平均 (過剰率補正後)
_LEAGUE = dict(win_rate=0.067, place_rate=0.33, win_return=0.75, place_return=0.75)

# レース間隔スコアテーブル (週数 → スコア)
_INTERVAL_SCORE = {
    1: 0.70,   # 連闘
    2: 0.82,
    3: 0.88,
    4: 1.00,   # 最適 (4〜5週)
    5: 1.00,
    6: 0.92,
    7: 0.85,
    8: 0.80,
}
_INTERVAL_DEFAULT = 0.72   # 9週以上 (長期休養)


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class CourseStats:
    """
    コース別成績統計 — スマート出馬表の「集計項目」行1つに対応。

    category の値:
        "父"        父馬コース
        "母父"      母父コース
        "騎手"      騎手コース
        "調教師"    調教師コース
        "父小系統"  父の小系統コース
        "母父小系統" 母父の小系統コース
        "父国×母父国" 父国 × 母父国コース
    """
    name: str
    category: str
    total_races: int = 0
    wins: int = 0
    top2: int = 0
    top3: int = 0
    win_return: float = 0.0    # 単回収率 (%)
    place_return: float = 0.0  # 複回収率 (%)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_races if self.total_races > 0 else 0.0

    @property
    def place_rate(self) -> float:
        return self.top3 / self.total_races if self.total_races > 0 else 0.0

    @property
    def win_return_rate(self) -> float:
        return self.win_return / 100.0

    @property
    def place_return_rate(self) -> float:
        return self.place_return / 100.0


# 後方互換: SireStats → CourseStats("父") ラッパー
def SireStats(
    sire_name: str,
    total_races: int = 0,
    wins: int = 0,
    top2: int = 0,
    top3: int = 0,
    win_return: float = 0.0,
    place_return: float = 0.0,
) -> CourseStats:
    return CourseStats(
        name=sire_name, category="父",
        total_races=total_races, wins=wins,
        top2=top2, top3=top3,
        win_return=win_return, place_return=place_return,
    )


@dataclass
class ScoreWeights:
    """
    stat_score 内の各要素の重み。

    合計が 1.0 でなくても predict() 内で自動正規化される。
    予想を重ねながら各値を調整してください。

    例:
        w = ScoreWeights()
        w.sire = 0.20        # 父馬を重視
        w.jockey = 0.05      # 騎手の影響を下げる
    """
    # ── 血統系 ──────────────────────────────────
    sire: float          = 0.15   # 父馬コース
    broodmare_sire: float = 0.10  # 母父コース
    sire_line: float     = 0.08   # 父小系統コース
    broodmare_line: float = 0.07  # 母父小系統コース
    nationality: float   = 0.06   # 父国×母父国コース
    # ── 人 ──────────────────────────────────────
    jockey: float        = 0.12   # 騎手コース
    trainer: float       = 0.10   # 調教師コース
    # ── 馬自身 ──────────────────────────────────
    horse: float         = 0.15   # 馬の過去成績
    # ── 展開系 ──────────────────────────────────
    pace: float          = 0.09   # テンP / 上がりP
    form: float          = 0.06   # 前走着差・レース間隔
    # ── フィジカル ──────────────────────────────
    burden: float        = 0.01   # 斤量
    weight_change: float = 0.01   # 馬体重増減

    def total(self) -> float:
        return sum(asdict(self).values())

    def normalized(self) -> "ScoreWeights":
        """合計が 1.0 になるように正規化したコピーを返す"""
        t = self.total()
        if t <= 0:
            raise ValueError("重みの合計が0以下です")
        d = asdict(self)
        return ScoreWeights(**{k: v / t for k, v in d.items()})

    def show(self) -> None:
        """現在の重み設定を表示する"""
        print(f"\n{'─'*45}")
        print(f"  ScoreWeights (合計: {self.total():.3f})")
        print(f"{'─'*45}")
        cats = {
            "血統系": ["sire","broodmare_sire","sire_line","broodmare_line","nationality"],
            "人":     ["jockey","trainer"],
            "馬自身": ["horse"],
            "展開系": ["pace","form"],
            "フィジカル": ["burden","weight_change"],
        }
        labels = {
            "sire": "父馬コース",
            "broodmare_sire": "母父コース",
            "sire_line": "父小系統コース",
            "broodmare_line": "母父小系統コース",
            "nationality": "父国×母父国コース",
            "jockey": "騎手コース",
            "trainer": "調教師コース",
            "horse": "馬の過去成績",
            "pace": "テンP/上がりP",
            "form": "前走着差・間隔",
            "burden": "斤量",
            "weight_change": "馬体重増減",
        }
        d = asdict(self)
        for cat, keys in cats.items():
            print(f"  [{cat}]")
            for k in keys:
                bar = "█" * int(d[k] * 100)
                print(f"    {labels[k]:<16} {d[k]:.3f}  {bar}")
        print(f"{'─'*45}\n")


@dataclass
class HorseStats:
    """馬の過去成績統計"""
    horse_name: str
    total_races: int = 0
    wins: int = 0
    top2: int = 0
    top3: int = 0
    avg_finish: float = 0.0

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
    market_prob: float
    final_prob: float
    expected_value: float        # 単勝期待値
    odds: Optional[float]
    recommendation: str          # ◎○▲△
    # スコア内訳 (デバッグ・チューニング用)
    score_breakdown: dict = field(default_factory=dict)
    # 父馬コース (表示用サマリ)
    sire_name: str = "-"
    sire_place_rate: float = 0.0
    sire_place_return: float = 0.0


# ---------------------------------------------------------------------------
# 特徴量エンジニアリング
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["finish_order"] = pd.to_numeric(df["finish_order"], errors="coerce")
    df = df.dropna(subset=["finish_order"])
    df["finish_order"] = df["finish_order"].astype(int)
    df["is_win"] = (df["finish_order"] == 1).astype(int)
    df["is_place"] = (df["finish_order"] <= 3).astype(int)
    df["finish_score"] = 1.0 / df["finish_order"]
    return df


def build_horse_stats(df: pd.DataFrame) -> dict[str, HorseStats]:
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


def build_jockey_history(df: pd.DataFrame) -> dict[str, float]:
    """過去データから騎手の勝率を算出 (コースデータがない場合のフォールバック)"""
    if "jockey" not in df.columns:
        return {}
    scores = {}
    for jockey, g in df.groupby("jockey"):
        if len(g) >= 10:
            scores[str(jockey)] = float(g["is_win"].mean())
    return scores


# ---------------------------------------------------------------------------
# コーススコア計算
# ---------------------------------------------------------------------------

def _course_score(stats: Optional[CourseStats], min_races: int = 5) -> float:
    """
    CourseStats → 0〜1+ の生スコアを返す。

    計算式:
        スコア = 複勝率×0.40 + 複回収率×0.40 + 単回収率×0.20
        ※ サンプル数が min_races 未満の場合はリーグ平均へベイズ縮小
    """
    L = _LEAGUE
    if stats is None or stats.total_races == 0:
        return L["place_rate"]*0.40 + L["place_return"]*0.40 + L["win_return"]*0.20

    conf = min(stats.total_races / 30.0, 1.0)  # 30レースで完全信頼

    pr  = conf * stats.place_rate       + (1-conf) * L["place_rate"]
    plr = conf * stats.place_return_rate + (1-conf) * L["place_return"]
    wr  = conf * stats.win_return_rate  + (1-conf) * L["win_return"]

    return pr*0.40 + plr*0.40 + wr*0.20


def _horse_score(s: Optional[HorseStats]) -> float:
    if s is None or s.total_races < 3:
        return 0.10
    sc = 0.5*s.win_rate + 0.3*s.place_rate + 0.2*(1.0/max(s.avg_finish, 1.0))
    conf = min(s.total_races / 30.0, 1.0)
    return sc*conf + 0.10*(1-conf)


def _pace_score(ten_pattern, agari_pattern) -> float:
    """テンP・上がりP から展開スコアを返す (0〜1)"""
    ten_sc   = _PATTERN_SCORE.get(ten_pattern,   _PATTERN_DEFAULT)
    agari_sc = _PATTERN_SCORE.get(agari_pattern, _PATTERN_DEFAULT)
    # 先行力と末脚を均等評価
    return (ten_sc + agari_sc) / 2.0


def _form_score(prev_margin, race_interval_weeks) -> float:
    """前走着差 + レース間隔スコア"""
    # 前走着差スコア
    if prev_margin is None or pd.isna(prev_margin):
        margin_sc = 0.50
    elif prev_margin >= 0:
        # 勝利 or 同着
        margin_sc = min(1.0, 0.70 + prev_margin * 0.05)
    else:
        # 負け (着差はマイナス値)
        margin_sc = max(0.05, 0.65 + prev_margin * 0.15)

    # レース間隔スコア
    weeks = int(race_interval_weeks) if race_interval_weeks and not pd.isna(race_interval_weeks) else None
    if weeks is None:
        interval_sc = 0.80
    elif weeks in _INTERVAL_SCORE:
        interval_sc = _INTERVAL_SCORE[weeks]
    else:
        interval_sc = _INTERVAL_DEFAULT

    return (margin_sc + interval_sc) / 2.0


def _burden_score(burden_weight) -> float:
    burden = float(burden_weight or 55.0)
    return max(0, (57.0 - burden) * 0.01)


def _weight_change_score(weight_diff) -> float:
    diff = float(weight_diff or 0.0)
    return max(0, 1.0 - abs(diff) * 0.005)


# ---------------------------------------------------------------------------
# メインスコア計算
# ---------------------------------------------------------------------------

def stat_score(
    entries: pd.DataFrame,
    horse_stats: dict[str, HorseStats],
    jockey_history: dict[str, float],
    course_stats: dict[str, dict[str, CourseStats]],
    weights: ScoreWeights,
) -> tuple[pd.Series, list[dict]]:
    """
    各馬のスコアを計算して返す。

    Args:
        entries: 出走馬DataFrame
            任意カラム: jockey, burden_weight, weight_diff,
                       ten_pattern, agari_pattern,
                       prev_margin, race_interval_weeks
        horse_stats: {horse_name: HorseStats}
        jockey_history: {jockey_name: win_rate} (フォールバック)
        course_stats: {horse_name: {category: CourseStats}}
            category = "父"|"母父"|"騎手"|"調教師"|"父小系統"|"母父小系統"|"父国×母父国"
        weights: ScoreWeights

    Returns:
        (スコア Series, 内訳リスト)
    """
    w = weights.normalized()
    scores = []
    breakdowns = []

    for _, row in entries.iterrows():
        name = str(row.get("horse_name", ""))
        cs = course_stats.get(name, {})

        # ── 各スコア ──────────────────────────────────────────
        sire_sc          = _course_score(cs.get("父"))
        bms_sc           = _course_score(cs.get("母父"))
        sire_line_sc     = _course_score(cs.get("父小系統"))
        bml_sc           = _course_score(cs.get("母父小系統"))
        nat_sc           = _course_score(cs.get("父国×母父国"))

        # 騎手: コースデータ優先、なければ過去勝率
        jockey_cs = cs.get("騎手")
        if jockey_cs:
            jockey_sc = _course_score(jockey_cs)
        else:
            jn = str(row.get("jockey", ""))
            fallback_rate = jockey_history.get(jn, 0.08)
            # 勝率を複勝率換算 (×3倍近似) して _course_score と揃える
            jockey_sc = fallback_rate*3 * 0.40 + 0.75*0.40 + fallback_rate*0.20

        trainer_sc = _course_score(cs.get("調教師"))

        horse_sc   = _horse_score(horse_stats.get(name))
        pace_sc    = _pace_score(
            row.get("ten_pattern"),
            row.get("agari_pattern"),
        )
        form_sc    = _form_score(
            row.get("prev_margin"),
            row.get("race_interval_weeks"),
        )
        burden_sc  = _burden_score(row.get("burden_weight"))
        wc_sc      = _weight_change_score(row.get("weight_diff"))

        total = (
            w.sire          * sire_sc
            + w.broodmare_sire * bms_sc
            + w.sire_line    * sire_line_sc
            + w.broodmare_line * bml_sc
            + w.nationality  * nat_sc
            + w.jockey       * jockey_sc
            + w.trainer      * trainer_sc
            + w.horse        * horse_sc
            + w.pace         * pace_sc
            + w.form         * form_sc
            + w.burden       * burden_sc
            + w.weight_change * wc_sc
        )

        scores.append(total)
        breakdowns.append({
            "horse_name":      name,
            "父馬":            round(sire_sc, 3),
            "母父":            round(bms_sc, 3),
            "父小系統":        round(sire_line_sc, 3),
            "母父小系統":      round(bml_sc, 3),
            "父国×母父国":     round(nat_sc, 3),
            "騎手":            round(jockey_sc, 3),
            "調教師":          round(trainer_sc, 3),
            "馬自身":          round(horse_sc, 3),
            "テンP/上がりP":   round(pace_sc, 3),
            "前走/間隔":       round(form_sc, 3),
            "斤量":            round(burden_sc, 3),
            "馬体重":          round(wc_sc, 3),
            "合計":            round(total, 4),
        })

    arr = np.array(scores, dtype=float)
    rng = arr.max() - arr.min()
    if rng > 0:
        arr = (arr - arr.min()) / rng
    return pd.Series(arr, index=entries.index), breakdowns


# ---------------------------------------------------------------------------
# 確率変換
# ---------------------------------------------------------------------------

def odds_to_market_prob(odds_series: pd.Series) -> pd.Series:
    raw = 1.0 / odds_series.clip(lower=1.0)
    total = raw.sum()
    return raw / total if total > 0 else pd.Series(np.nan, index=odds_series.index)


# ---------------------------------------------------------------------------
# 予想エンジン
# ---------------------------------------------------------------------------

class HorseRacingPredictor:
    """
    統計・確率モデルによる競馬予想エンジン

    使い方:
        predictor = HorseRacingPredictor()
        predictor.weights.sire = 0.20       # 重みをその場で調整
        predictor.fit(historical_df)
        results = predictor.predict(race_df, course_stats, surface="芝", distance=1400)

    course_stats の構造:
        {
          "チムグクル": {
              "父":     CourseStats("ディープインパクト", "父", 152, 19, ...),
              "母父":   CourseStats("クロフネ", "母父", 40, 3, ...),
              "騎手":   CourseStats("川田将雅", "騎手", 80, 12, ...),
              ...
          },
          ...
        }
    """

    def __init__(
        self,
        market_weight: float = 0.40,
        stat_weight: float = 0.60,
        weights: Optional[ScoreWeights] = None,
    ):
        """
        Args:
            market_weight: オッズ市場確率の重み (market + stat = 1.0)
            stat_weight:   統計スコアの重み
            weights:       各要素の内部重み。None の場合デフォルト値を使用
        """
        assert abs(market_weight + stat_weight - 1.0) < 1e-9
        self.market_weight = market_weight
        self.stat_weight = stat_weight
        self.weights = weights or ScoreWeights()
        self.horse_stats: dict[str, HorseStats] = {}
        self.jockey_history: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "HorseRacingPredictor":
        df = preprocess(df)
        self.horse_stats = build_horse_stats(df)
        self.jockey_history = build_jockey_history(df)
        logger.info("学習完了: 馬数=%d, 騎手数=%d", len(self.horse_stats), len(self.jockey_history))
        return self

    def predict(
        self,
        race_df: pd.DataFrame,
        course_stats: Optional[dict[str, dict[str, CourseStats]]] = None,
        surface: str = "芝",
        distance: int = 1600,
    ) -> list[PredictionResult]:
        """
        Args:
            race_df: 出走馬DataFrame
                必須: horse_name, horse_number, odds
                任意: jockey, burden_weight, weight_diff,
                      ten_pattern (15/30/50), agari_pattern (15/30/50),
                      prev_margin (秒差・マイナスは負け), race_interval_weeks
            course_stats: {horse_name: {category: CourseStats}}
            surface: "芝" / "ダート"
            distance: レース距離 (m)
        """
        race_df = race_df.copy().reset_index(drop=True)
        cs = course_stats or {}

        # 市場確率
        if "odds" in race_df.columns:
            market_prob = odds_to_market_prob(race_df["odds"].astype(float))
        else:
            n = len(race_df)
            market_prob = pd.Series([1.0 / n] * n)

        # 統計スコア
        s_scores, breakdowns = stat_score(
            race_df, self.horse_stats, self.jockey_history, cs, self.weights
        )
        s_norm = s_scores / s_scores.sum() if s_scores.sum() > 0 else s_scores

        # 統合確率
        combined = self.market_weight * market_prob.fillna(0) + self.stat_weight * s_norm
        combined = combined / combined.sum()

        results = []
        for idx, row in race_df.iterrows():
            prob = float(combined.iloc[idx])
            odds_val = float(row.get("odds", np.nan))
            ev = prob * odds_val - 1.0 if not np.isnan(odds_val) and odds_val > 0 else np.nan

            horse_cs = cs.get(str(row.get("horse_name", "")), {})
            sire = horse_cs.get("父")

            results.append(PredictionResult(
                horse_number=int(row.get("horse_number", idx + 1)),
                horse_name=str(row.get("horse_name", f"馬{idx+1}")),
                market_prob=float(market_prob.iloc[idx]),
                final_prob=prob,
                expected_value=ev,
                odds=odds_val if not np.isnan(odds_val) else None,
                recommendation="",
                score_breakdown=breakdowns[idx],
                sire_name=sire.name if sire else "-",
                sire_place_rate=sire.place_rate if sire else 0.0,
                sire_place_return=sire.place_return if sire else 0.0,
            ))

        # 印付け
        results.sort(key=lambda r: r.final_prob, reverse=True)
        for i, r in enumerate(results[:4]):
            r.recommendation = ["◎本命", "○対抗", "▲単穴", "△穴"][i]

        return results


# ---------------------------------------------------------------------------
# 表示ユーティリティ
# ---------------------------------------------------------------------------

def print_prediction(results: list[PredictionResult], show_breakdown: bool = False) -> None:
    """予想結果を表示する"""
    has_sire = any(r.sire_name != "-" for r in results)

    if has_sire:
        hdr = (f"{'馬番':>4}  {'馬名':<14}  {'印':<5}  {'予測勝率':>7}  "
               f"{'オッズ':>6}  {'期待値':>7}  {'父複勝率':>6}  {'父複回収':>6}  父馬名")
        W = 85
    else:
        hdr = (f"{'馬番':>4}  {'馬名':<14}  {'印':<5}  {'予測勝率':>7}  "
               f"{'オッズ':>6}  {'期待値':>7}")
        W = 60

    print(f"\n{'='*W}")
    print(hdr)
    print(f"{'-'*W}")

    for r in results:
        ev_s   = f"{r.expected_value:+.2f}" if r.expected_value is not None and not np.isnan(r.expected_value) else "  N/A"
        odds_s = f"{r.odds:.1f}" if r.odds else " N/A"
        line = (f"{r.horse_number:>4}  {r.horse_name:<14}  {r.recommendation:<5}  "
                f"{r.final_prob:>6.1%}  {odds_s:>6}  {ev_s:>7}")
        if has_sire:
            line += f"  {r.sire_place_rate:>5.0%}  {r.sire_place_return:>5.0f}%  {r.sire_name}"
        print(line)

    print(f"{'='*W}")
    print("※ 期待値 > 0 の馬が統計的にプラス期待値")
    if has_sire:
        print("※ 父複回収 = 父馬の複勝回収率。100%超えで期待値プラス")

    if show_breakdown:
        print(f"\n{'─'*W}")
        print("スコア内訳:")
        print(f"{'─'*W}")
        keys = ["父馬","母父","父小系統","母父小系統","父国×母父国","騎手","調教師","馬自身","テンP/上がりP","前走/間隔","斤量","馬体重"]
        header = f"{'馬名':<14}  " + "  ".join(f"{k:>7}" for k in keys)
        print(header)
        for r in sorted(results, key=lambda x: x.horse_number):
            bd = r.score_breakdown
            vals = "  ".join(f"{bd.get(k, 0):>7.3f}" for k in keys)
            print(f"{r.horse_name:<14}  {vals}")
    print()


def print_bet_suggestions(results: list[PredictionResult]) -> None:
    """馬券購入候補を表示する"""
    top = results[:4]
    names = [f"[{r.horse_number}]{r.horse_name}" for r in top]

    print("─── 馬券推奨 ───────────────────────────────────")
    print(f"  単勝:     {names[0]}")
    print(f"  複勝:     {' / '.join(names[:3])}")
    print(f"  馬連ボックス: {' - '.join(names[:3])}")
    print(f"  3連複フォーメーション:")
    print(f"    軸: {names[0]}")
    print(f"    相手: {' / '.join(names[1:4])}")

    ev_plus = [r for r in results if not np.isnan(r.expected_value) and r.expected_value > 0]
    if ev_plus:
        print(f"  単勝期待値プラス候補:")
        for r in ev_plus:
            print(f"    [{r.horse_number}]{r.horse_name}  EV={r.expected_value:+.2f}  オッズ={r.odds}")

    place_ev = [r for r in results if r.sire_place_return >= 100]
    if place_ev:
        print(f"  父複回収100%超え (複勝期待値候補):")
        for r in place_ev:
            print(f"    [{r.horse_number}]{r.horse_name}  父複回収={r.sire_place_return:.0f}%")
    print()
