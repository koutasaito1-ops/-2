"""
競馬予想バックテストエンジン

過去レースデータで予想モデルを検証し、
どの要素（父コース複勝率・上がりP・騎手スコア等）が
実際の回収率・期待値と相関するかを分析する。

使い方:
  python backtest.py              # 内蔵サンプルレースで分析
  python backtest.py --verbose    # 各レースの詳細も表示
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from model import (
    CourseStats,
    HorseRacingPredictor,
    ScoreWeights,
    stat_score,
    odds_to_market_prob,
    preprocess,
    build_horse_stats,
    build_jockey_history,
)

logging.basicConfig(level=logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════
# データ構造
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ActualResult:
    """1頭分の実際の結果"""
    horse_name: str
    finish_order: int
    win_payout: float    # 単勝払戻 (1着のみ。例: 630 → 630円。0=不的中)
    place_payout: float  # 複勝払戻 (3着以内のみ。0=不的中)


@dataclass
class PastRace:
    """過去レース1レース分のデータ (予想前情報 + 実際の結果)"""
    race_id: str
    race_name: str
    date: str
    venue: str
    surface: str
    distance: int
    entries: pd.DataFrame           # 出走馬データ (horse_name, horse_number, odds, etc.)
    course_stats: dict              # {horse_name: {category: CourseStats}}
    actual_results: list[ActualResult]  # 実際の着順・払戻
    note: str = ""


@dataclass
class HorseRecord:
    """バックテスト1頭分の記録 (相関分析に使う)"""
    race_id: str
    race_name: str
    horse_name: str
    horse_number: int
    odds: float
    predicted_prob: float
    market_prob: float
    # ── 実際の結果 ──────────────────────────────────────
    finish_order: int
    is_win: int           # 1着=1, 他=0
    is_place: int         # 3着以内=1, 他=0
    win_return: float     # 単勝回収率 (100円賭けた場合の払戻 ÷ 100)
    place_return: float   # 複勝回収率
    # ── 各要素のスコア (モデル内部値) ──────────────────
    score_father: float        # 父馬コーススコア
    score_broodmare: float     # 母父コーススコア
    score_sire_line: float     # 父小系統コーススコア
    score_broodmare_line: float# 母父小系統コーススコア
    score_nationality: float   # 父国×母父国コーススコア
    score_jockey: float        # 騎手コーススコア
    score_trainer: float       # 調教師コーススコア
    score_horse: float         # 馬自身スコア
    score_pace: float          # テンP/上がりPスコア
    score_form: float          # 前走着差・間隔スコア
    score_burden: float        # 斤量スコア
    score_weight_change: float # 馬体重増減スコア
    # ── スマート出走表 生データ ────────────────────────
    father_place_rate: float   # 父馬 複勝率
    father_place_return: float # 父馬 複回収率
    father_win_return: float   # 父馬 単回収率
    father_total_races: int    # 父馬 サンプル数
    jockey_place_rate: float   # 騎手 複勝率
    jockey_place_return: float # 騎手 複回収率
    ten_pattern: Optional[int] # テンP (15/30/50/None)
    agari_pattern: Optional[int]# 上がりP
    popularity: int            # 人気順位 (オッズ順)
    ev_predicted: float        # 単勝期待値 (predicted_prob × odds - 1)


# ═══════════════════════════════════════════════════════════════════════════
# バックテストエンジン
# ═══════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    過去レースデータでモデルを検証し、要素別相関を分析するエンジン。

    使い方:
        engine = BacktestEngine()
        engine.add_history(hist_df)          # 学習データを渡す
        engine.run(past_races)               # 過去レースを検証
        engine.print_summary()               # サマリ表示
        engine.print_factor_correlation()    # 要素別相関表示
    """

    def __init__(self, weights: Optional[ScoreWeights] = None):
        self.weights = weights or ScoreWeights()
        self.records: list[HorseRecord] = []
        self._predictor = HorseRacingPredictor(weights=self.weights)

    def add_history(self, df: pd.DataFrame) -> None:
        """学習用の過去成績データを渡す (馬自身スコア用)"""
        df = preprocess(df)
        self._predictor.horse_stats = build_horse_stats(df)
        self._predictor.jockey_history = build_jockey_history(df)

    def run(self, races: list[PastRace], verbose: bool = False) -> None:
        """過去レースリストを検証してレコードを蓄積する"""
        self.records.clear()
        for race in races:
            recs = self._run_one(race, verbose=verbose)
            self.records.extend(recs)
        print(f"\n✅ バックテスト完了: {len(races)}レース / {len(self.records)}頭分のデータ")

    def _run_one(self, race: PastRace, verbose: bool) -> list[HorseRecord]:
        entries = race.entries.copy().reset_index(drop=True)
        cs = race.course_stats or {}
        w = self.weights.normalized()

        # ── 市場確率 ──────────────────────────────────────────────
        market_prob = odds_to_market_prob(entries["odds"].astype(float))

        # ── 統計スコア (内訳付き) ──────────────────────────────────
        from model import (
            _course_score, _horse_score, _pace_score,
            _form_score, _burden_score, _weight_change_score,
        )

        s_scores_raw = []
        breakdowns = []
        for _, row in entries.iterrows():
            name = str(row.get("horse_name", ""))
            horse_cs = cs.get(name, {})

            sc_father       = _course_score(horse_cs.get("父"))
            sc_bms          = _course_score(horse_cs.get("母父"))
            sc_sire_line    = _course_score(horse_cs.get("父小系統"))
            sc_bml          = _course_score(horse_cs.get("母父小系統"))
            sc_nat          = _course_score(horse_cs.get("父国×母父国"))

            jockey_cs = horse_cs.get("騎手")
            if jockey_cs:
                sc_jockey = _course_score(jockey_cs)
            else:
                jn = str(row.get("jockey", ""))
                fb = self._predictor.jockey_history.get(jn, 0.08)
                sc_jockey = fb*3*0.40 + 0.75*0.40 + fb*0.20

            sc_trainer = _course_score(horse_cs.get("調教師"))
            sc_horse   = _horse_score(self._predictor.horse_stats.get(name))
            sc_pace    = _pace_score(row.get("ten_pattern"), row.get("agari_pattern"))
            sc_form    = _form_score(row.get("prev_margin"), row.get("race_interval_weeks"))
            sc_burden  = _burden_score(row.get("burden_weight"))
            sc_wc      = _weight_change_score(row.get("weight_diff"))

            total = (
                w.sire           * sc_father
                + w.broodmare_sire * sc_bms
                + w.sire_line     * sc_sire_line
                + w.broodmare_line * sc_bml
                + w.nationality   * sc_nat
                + w.jockey        * sc_jockey
                + w.trainer       * sc_trainer
                + w.horse         * sc_horse
                + w.pace          * sc_pace
                + w.form          * sc_form
                + w.burden        * sc_burden
                + w.weight_change * sc_wc
            )
            s_scores_raw.append(total)
            breakdowns.append({
                "父": sc_father, "母父": sc_bms, "父小系統": sc_sire_line,
                "母父小系統": sc_bml, "父国×母父国": sc_nat,
                "騎手": sc_jockey, "調教師": sc_trainer, "馬自身": sc_horse,
                "テンP/上がりP": sc_pace, "前走/間隔": sc_form,
                "斤量": sc_burden, "馬体重": sc_wc,
            })

        arr = np.array(s_scores_raw, dtype=float)
        rng = arr.max() - arr.min()
        if rng > 0:
            arr = (arr - arr.min()) / rng
        s_scores_norm = arr / arr.sum() if arr.sum() > 0 else arr

        # ── 統合確率 ──────────────────────────────────────────────
        mw = self._predictor.market_weight
        sw = self._predictor.stat_weight
        combined = mw * market_prob.fillna(0) + sw * s_scores_norm
        combined = combined / combined.sum()

        # ── 人気順位 (オッズ昇順) ─────────────────────────────────
        odds_arr = entries["odds"].astype(float).values
        popularity = np.argsort(np.argsort(odds_arr)) + 1  # 1-indexed

        # ── 実際の結果とマッチング ────────────────────────────────
        result_map = {r.horse_name: r for r in race.actual_results}
        records = []

        for idx, row in entries.iterrows():
            name = str(row.get("horse_name", ""))
            actual = result_map.get(name)
            if actual is None:
                continue

            odds_val = float(row.get("odds", np.nan))
            prob = float(combined.iloc[idx])
            mp = float(market_prob.iloc[idx])
            ev = prob * odds_val - 1.0 if odds_val > 0 else np.nan

            horse_cs = cs.get(name, {})
            father = horse_cs.get("父")
            jockey_cs = horse_cs.get("騎手")

            rec = HorseRecord(
                race_id=race.race_id,
                race_name=race.race_name,
                horse_name=name,
                horse_number=int(row.get("horse_number", idx + 1)),
                odds=odds_val,
                predicted_prob=prob,
                market_prob=mp,
                finish_order=actual.finish_order,
                is_win=int(actual.finish_order == 1),
                is_place=int(actual.finish_order <= 3),
                win_return=actual.win_payout / 100.0 if actual.finish_order == 1 else 0.0,
                place_return=actual.place_payout / 100.0 if actual.finish_order <= 3 else 0.0,
                score_father=breakdowns[idx]["父"],
                score_broodmare=breakdowns[idx]["母父"],
                score_sire_line=breakdowns[idx]["父小系統"],
                score_broodmare_line=breakdowns[idx]["母父小系統"],
                score_nationality=breakdowns[idx]["父国×母父国"],
                score_jockey=breakdowns[idx]["騎手"],
                score_trainer=breakdowns[idx]["調教師"],
                score_horse=breakdowns[idx]["馬自身"],
                score_pace=breakdowns[idx]["テンP/上がりP"],
                score_form=breakdowns[idx]["前走/間隔"],
                score_burden=breakdowns[idx]["斤量"],
                score_weight_change=breakdowns[idx]["馬体重"],
                father_place_rate=father.place_rate if father else 0.0,
                father_place_return=father.place_return if father else 0.0,
                father_win_return=father.win_return if father else 0.0,
                father_total_races=father.total_races if father else 0,
                jockey_place_rate=jockey_cs.place_rate if jockey_cs else 0.0,
                jockey_place_return=jockey_cs.place_return if jockey_cs else 0.0,
                ten_pattern=row.get("ten_pattern"),
                agari_pattern=row.get("agari_pattern"),
                popularity=int(popularity[idx]),
                ev_predicted=ev,
            )
            records.append(rec)

        if verbose:
            _print_race_detail(race, records, combined, entries)

        return records

    # ── 分析・表示 ────────────────────────────────────────────────

    def print_summary(self) -> None:
        """バックテスト全体サマリを表示する"""
        if not self.records:
            print("データがありません")
            return

        df = pd.DataFrame([r.__dict__ for r in self.records])
        races = df["race_id"].nunique()
        n = len(df)

        # 推奨馬 (本命◎ = predicted_prob が各レース1位の馬)
        top_pick = df.loc[df.groupby("race_id")["predicted_prob"].idxmax()]
        top2 = df[df.groupby("race_id")["predicted_prob"].transform("rank", ascending=False) <= 2]
        top3 = df[df.groupby("race_id")["predicted_prob"].transform("rank", ascending=False) <= 3]

        print(f"\n{'='*60}")
        print(f"  バックテスト結果サマリ ({races}レース / {n}頭)")
        print(f"{'='*60}")
        print(f"  【本命◎ 単勝】")
        print(f"    的中率: {top_pick['is_win'].mean():.1%}")
        print(f"    単勝回収率: {top_pick['win_return'].mean() * 100:.0f}%")
        print(f"  【◎○ 馬連ボックス参考】")
        win_pair = top2.groupby("race_id")["is_win"].max()
        print(f"    2頭内1着率: {win_pair.mean():.1%}")
        print(f"  【◎○▲ 複勝 / 3連複参考】")
        place3 = top3.groupby("race_id")["is_place"].max()
        print(f"    3頭内複勝率: {place3.mean():.1%}")
        print(f"{'='*60}")

        # 期待値プラス推奨 (ev > 0) の成績
        ev_plus = df[df["ev_predicted"] > 0]
        if len(ev_plus) > 0:
            print(f"  【期待値プラス推奨馬 ({len(ev_plus)}頭)】")
            print(f"    単勝的中率: {ev_plus['is_win'].mean():.1%}")
            print(f"    複勝的中率: {ev_plus['is_place'].mean():.1%}")
            avg_ret = ev_plus["win_return"].sum() / len(ev_plus) * 100
            print(f"    単勝平均回収率: {avg_ret:.0f}%")
            avg_pret = ev_plus["place_return"].sum() / len(ev_plus) * 100
            print(f"    複勝平均回収率: {avg_pret:.0f}%")
        print()

    def print_factor_correlation(self) -> None:
        """各要素と実際の回収率の相関係数を表示する"""
        if not self.records:
            return

        df = pd.DataFrame([r.__dict__ for r in self.records])
        n = len(df)

        # ── 分析する要素の定義 ────────────────────────────────────
        factors = {
            # スコア系
            "父馬コーススコア":      "score_father",
            "母父コーススコア":      "score_broodmare",
            "父小系統コーススコア":  "score_sire_line",
            "母父小系統コーススコア":"score_broodmare_line",
            "父国×母父国スコア":     "score_nationality",
            "騎手コーススコア":      "score_jockey",
            "調教師コーススコア":    "score_trainer",
            "馬自身スコア":          "score_horse",
            "テンP/上がりPスコア":   "score_pace",
            "前走/間隔スコア":       "score_form",
            # 生データ系
            "父馬複勝率(生)":        "father_place_rate",
            "父馬複回収率(生)":      "father_place_return",
            "父馬単回収率(生)":      "father_win_return",
            "父馬サンプル数":        "father_total_races",
            "騎手複勝率(生)":        "jockey_place_rate",
            "騎手複回収率(生)":      "jockey_place_return",
            # モデル出力
            "モデル予測確率":        "predicted_prob",
            "市場確率(オッズ逆数)":  "market_prob",
            "予測EV(期待値)":        "ev_predicted",
            "人気順位":              "popularity",
        }

        rows = []
        for label, col in factors.items():
            if col not in df.columns:
                continue
            x = df[col].astype(float)
            # NaN除去
            mask = x.notna() & df["win_return"].notna() & df["place_return"].notna()
            xv = x[mask].values
            wr = df.loc[mask, "win_return"].values
            pr = df.loc[mask, "place_return"].values
            iw = df.loc[mask, "is_win"].values
            ip = df.loc[mask, "is_place"].values

            if len(xv) < 5 or xv.std() < 1e-9:
                continue

            r_wr, p_wr = stats.spearmanr(xv, wr)
            r_pr, p_pr = stats.spearmanr(xv, pr)
            r_iw, _    = stats.spearmanr(xv, iw)
            r_ip, _    = stats.spearmanr(xv, ip)

            # 上位25% vs 下位25% の回収率比較
            q75, q25 = np.percentile(xv, [75, 25])
            high = df.loc[mask & (x >= q75)]
            low  = df.loc[mask & (x <= q25)]
            high_win_roi = high["win_return"].mean() * 100
            low_win_roi  = low["win_return"].mean() * 100
            high_pl_roi  = high["place_return"].mean() * 100
            low_pl_roi   = low["place_return"].mean() * 100

            rows.append({
                "要素":           label,
                "単勝相関(Sp)":   round(r_wr, 3),
                "複勝相関(Sp)":   round(r_pr, 3),
                "勝率相関":       round(r_iw, 3),
                "複勝率相関":     round(r_ip, 3),
                "上位25%単回%":   round(high_win_roi, 0),
                "下位25%単回%":   round(low_win_roi, 0),
                "上位25%複回%":   round(high_pl_roi, 0),
                "下位25%複回%":   round(low_pl_roi, 0),
                "p値(単勝)":      round(p_wr, 3),
            })

        result_df = pd.DataFrame(rows)
        result_df = result_df.sort_values("複勝相関(Sp)", ascending=False)

        print(f"\n{'='*100}")
        print(f"  要素別 回収率相関分析 (Spearman ρ, n={n}頭)")
        print(f"  ※ 相関係数: +1.0→完全正相関, -1.0→完全負相関, 0→無相関")
        print(f"  ※ 上位25%/下位25%: その要素が高い/低い馬の平均回収率")
        print(f"{'='*100}")
        hdr = (f"{'要素':<22}  {'単勝相関':>8}  {'複勝相関':>8}  {'勝率相関':>8}  "
               f"{'複勝率相関':>9}  {'上位単回':>8}  {'下位単回':>8}  {'上位複回':>8}  {'下位複回':>8}  {'p値':>6}")
        print(hdr)
        print("─" * 100)
        for _, row in result_df.iterrows():
            corr_mark = ""
            if abs(row["複勝相関(Sp)"]) >= 0.3:
                corr_mark = " ★"
            elif abs(row["複勝相関(Sp)"]) >= 0.15:
                corr_mark = " ☆"
            print(
                f"{row['要素']:<22}  {row['単勝相関(Sp)']:>+8.3f}  {row['複勝相関(Sp)']:>+8.3f}  "
                f"{row['勝率相関']:>+8.3f}  {row['複勝率相関']:>+9.3f}  "
                f"{row['上位25%単回%']:>7.0f}%  {row['下位25%単回%']:>7.0f}%  "
                f"{row['上位25%複回%']:>7.0f}%  {row['下位25%複回%']:>7.0f}%  "
                f"{row['p値(単勝)']:>6.3f}{corr_mark}"
            )
        print(f"{'='*100}")
        print("★=|ρ|≥0.30 (有意な相関), ☆=|ρ|≥0.15 (弱い相関)")

    def suggest_weights(self) -> ScoreWeights:
        """
        相関分析結果をもとに重み改善案を提案する。
        相関係数の絶対値が高い要素ほど重みを増やす。
        """
        if not self.records:
            return self.weights

        df = pd.DataFrame([r.__dict__ for r in self.records])
        factor_cols = {
            "sire":           "score_father",
            "broodmare_sire": "score_broodmare",
            "sire_line":      "score_sire_line",
            "broodmare_line": "score_broodmare_line",
            "nationality":    "score_nationality",
            "jockey":         "score_jockey",
            "trainer":        "score_trainer",
            "horse":          "score_horse",
            "pace":           "score_pace",
            "form":           "score_form",
        }

        correlations = {}
        for key, col in factor_cols.items():
            if col not in df.columns:
                continue
            x = df[col].astype(float)
            y = df["place_return"].astype(float)
            mask = x.notna() & y.notna() & (x.std() > 1e-9)
            if mask.sum() < 5:
                continue
            r, _ = stats.spearmanr(x[mask].values, y[mask].values)
            correlations[key] = abs(r)

        if not correlations:
            return self.weights

        # 現在の重みに相関係数を比例させて調整 (平均回帰を防ぐため50:50でブレンド)
        from dataclasses import asdict
        current = asdict(self.weights)
        total_corr = sum(correlations.values()) or 1.0
        suggested = ScoreWeights()
        fixed_sum = current.get("burden", 0.01) + current.get("weight_change", 0.01)
        adjustable_budget = 1.0 - fixed_sum

        for key, abs_r in correlations.items():
            new_w = adjustable_budget * (abs_r / total_corr)
            old_w = current.get(key, 0.0)
            blended = 0.5 * old_w + 0.5 * new_w
            setattr(suggested, key, round(blended, 4))

        suggested.burden = self.weights.burden
        suggested.weight_change = self.weights.weight_change

        print(f"\n{'─'*60}")
        print("  ⚡ 重み改善案 (相関ベース)")
        print(f"{'─'*60}")
        labels = {
            "sire": "父馬", "broodmare_sire": "母父",
            "sire_line": "父小系統", "broodmare_line": "母父小系統",
            "nationality": "父国×母父国", "jockey": "騎手",
            "trainer": "調教師", "horse": "馬自身",
            "pace": "テンP/上がりP", "form": "前走/間隔",
        }
        for key, label in labels.items():
            old = current.get(key, 0)
            new = getattr(suggested, key, 0)
            arrow = "↑" if new > old else "↓" if new < old else "→"
            print(f"  {label:<12} {old:.3f} → {new:.3f}  {arrow}")
        print(f"{'─'*60}")
        print("  ※ モデルへの適用: predictor.weights = suggested_weights")
        print(f"{'─'*60}\n")

        return suggested


# ═══════════════════════════════════════════════════════════════════════════
# 表示ヘルパー
# ═══════════════════════════════════════════════════════════════════════════

def _print_race_detail(
    race: PastRace,
    records: list[HorseRecord],
    combined: pd.Series,
    entries: pd.DataFrame,
) -> None:
    result_map = {r.horse_name: r for r in race.actual_results}
    sorted_recs = sorted(records, key=lambda r: r.predicted_prob, reverse=True)
    marks = ["◎", "○", "▲", "△"]

    print(f"\n{'─'*75}")
    print(f"  {race.race_name}  {race.date}  {race.venue} {race.surface}{race.distance}m")
    print(f"{'─'*75}")
    print(f"  {'馬番':>4}  {'馬名':<14} {'印':>3}  {'予測':>7}  {'オッズ':>6}  {'実際':>4}  {'単回':>7}  {'複回':>7}")
    for i, r in enumerate(sorted_recs):
        mark = marks[i] if i < 4 else " "
        order_str = f"{r.finish_order}着"
        win_r = f"{r.win_return*100:.0f}%" if r.is_win else "  -  "
        pl_r  = f"{r.place_return*100:.0f}%" if r.is_place else "  -  "
        hit = "✅" if (i == 0 and r.is_win) or (i < 3 and r.is_place) else ""
        print(f"  {r.horse_number:>4}  {r.horse_name:<14} {mark:>3}  "
              f"{r.predicted_prob:>6.1%}  {r.odds:>6.1f}  {order_str:>4}  "
              f"{win_r:>7}  {pl_r:>7} {hit}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# エントリポイント
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="競馬予想バックテスト & 要素相関分析")
    parser.add_argument("--verbose", "-v", action="store_true", help="各レースの詳細を表示")
    parser.add_argument("--suggest", "-s", action="store_true", help="重み改善案を提案")
    args = parser.parse_args()

    from sample_races import ALL_SAMPLE_RACES
    from predict import SAMPLE_HISTORY

    print("\n" + "="*70)
    print("  競馬予想バックテスト & 要素相関分析")
    print(f"  対象レース: {len(ALL_SAMPLE_RACES)}レース")
    print("="*70)

    engine = BacktestEngine()
    engine.add_history(SAMPLE_HISTORY)
    engine.run(ALL_SAMPLE_RACES, verbose=args.verbose)
    engine.print_summary()
    engine.print_factor_correlation()

    if args.suggest:
        suggested = engine.suggest_weights()
        print("\n  ↑ 上記の重みをコードに反映するには predict.py の")
        print("    ScoreWeights の各値を更新してください")


if __name__ == "__main__":
    main()
