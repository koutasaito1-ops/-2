"""
競馬データスクレイパー
netkeibaから過去レース結果・馬情報を取得する

注意: スクレイピングはnetkeiba利用規約を遵守してください。
      過度なアクセスは避け、適切な待機時間を設けてください。
"""

import time
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://db.netkeiba.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


@dataclass
class RaceEntry:
    """1レースの出走馬エントリ情報"""
    race_id: str
    horse_number: int
    horse_name: str
    jockey: str
    trainer: str
    weight: Optional[float]        # 馬体重 (kg)
    weight_diff: Optional[float]   # 馬体重増減
    odds: Optional[float]          # 単勝オッズ
    popularity: Optional[int]      # 人気順位
    age: Optional[int]
    sex: str
    burden_weight: Optional[float] # 斤量


@dataclass
class RaceResult:
    """過去レース結果"""
    race_id: str
    date: str
    place: str
    race_name: str
    distance: int
    surface: str   # 芝 / ダート
    weather: str
    condition: str # 馬場状態
    entries: list = field(default_factory=list)  # list[RaceEntry + finish_order]


class NetkeibaScaper:
    """netkeibaスクレイパー"""

    def __init__(self, interval: float = 2.0):
        """
        Args:
            interval: リクエスト間隔(秒)。サーバー負荷を下げるために必ず設定する
        """
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.interval = interval

    def _get(self, url: str) -> BeautifulSoup:
        time.sleep(self.interval)
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return BeautifulSoup(resp.text, "html.parser")

    # ------------------------------------------------------------------
    # レース一覧取得
    # ------------------------------------------------------------------
    def get_race_id_list(self, year: int, place_code: str = "05") -> list[str]:
        """
        指定年・開催場所のレースIDリストを取得する。

        Args:
            year: 年 (例: 2024)
            place_code: 競馬場コード
                01=札幌 02=函館 03=福島 04=新潟 05=東京
                06=中山 07=中京 08=京都 09=阪神 10=小倉

        Returns:
            レースIDのリスト
        """
        race_ids = []
        for kai in range(1, 7):      # 開催回
            for day in range(1, 13): # 開催日
                url = (
                    f"https://race.netkeiba.com/top/race_list_sub.html"
                    f"?kaisai_id={year}{place_code:02}{kai:02}{day:02}"
                )
                try:
                    soup = self._get(url)
                    links = soup.select("a[href*='/race/result']")
                    for a in links:
                        m = re.search(r"race_id=(\d+)", a["href"])
                        if m:
                            race_ids.append(m.group(1))
                except requests.HTTPError as e:
                    logger.debug("skip %s: %s", url, e)
        logger.info("取得レース数: %d", len(race_ids))
        return race_ids

    # ------------------------------------------------------------------
    # レース結果取得
    # ------------------------------------------------------------------
    def get_race_result(self, race_id: str) -> Optional[pd.DataFrame]:
        """
        レース結果ページをスクレイピングしてDataFrameで返す。

        Returns:
            columns: finish_order, horse_number, horse_name, jockey,
                     burden_weight, odds, popularity, time, margin,
                     corner_pass, last_3f, weight, weight_diff, trainer
        """
        url = f"{BASE_URL}/race/{race_id}/"
        try:
            soup = self._get(url)
        except requests.HTTPError as e:
            logger.warning("レース取得失敗 %s: %s", race_id, e)
            return None

        table = soup.select_one("table.race_table_01")
        if table is None:
            logger.warning("結果テーブルが見つかりません: %s", race_id)
            return None

        rows = []
        for tr in table.select("tr")[1:]:
            tds = [td.get_text(strip=True) for td in tr.select("td")]
            if len(tds) < 10:
                continue
            rows.append(tds)

        if not rows:
            return None

        columns = [
            "finish_order", "frame_number", "horse_number", "horse_name",
            "sex_age", "burden_weight", "jockey", "time", "margin",
            "popularity", "odds", "last_3f", "corner_pass",
            "trainer", "weight", "prize"
        ]
        df = pd.DataFrame(rows, columns=columns[:len(rows[0])])
        df["race_id"] = race_id

        # 型変換
        for col in ["finish_order", "horse_number", "popularity"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["burden_weight", "odds", "last_3f"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 馬体重と増減を分割 (例: "480(-2)")
        weight_parsed = df["weight"].str.extract(r"(\d+)\(([+-]?\d+)\)")
        df["weight_kg"] = pd.to_numeric(weight_parsed[0], errors="coerce")
        df["weight_diff"] = pd.to_numeric(weight_parsed[1], errors="coerce")

        return df

    # ------------------------------------------------------------------
    # 馬の過去成績取得
    # ------------------------------------------------------------------
    def get_horse_history(self, horse_id: str) -> Optional[pd.DataFrame]:
        """
        馬の過去全成績を取得する。

        Args:
            horse_id: netkeiba の馬ID (URLから取得)
        """
        url = f"{BASE_URL}/horse/{horse_id}/"
        try:
            soup = self._get(url)
        except requests.HTTPError as e:
            logger.warning("馬情報取得失敗 %s: %s", horse_id, e)
            return None

        table = soup.select_one("table.db_h_race_results")
        if table is None:
            return None

        rows = []
        for tr in table.select("tr")[1:]:
            tds = [td.get_text(strip=True) for td in tr.select("td")]
            if tds:
                rows.append(tds)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["horse_id"] = horse_id
        return df

    # ------------------------------------------------------------------
    # 出走表取得（予想用）
    # ------------------------------------------------------------------
    def get_shutuba(self, race_id: str) -> Optional[pd.DataFrame]:
        """
        出走表（まだ走っていないレース）を取得する。

        Returns:
            出走予定馬のDataFrame
        """
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        try:
            soup = self._get(url)
        except requests.HTTPError as e:
            logger.warning("出走表取得失敗 %s: %s", race_id, e)
            return None

        table = soup.select_one("table.Shutuba_Table")
        if table is None:
            logger.warning("出走表テーブルが見つかりません: %s", race_id)
            return None

        rows = []
        for tr in table.select("tr.HorseList"):
            tds = [td.get_text(strip=True) for td in tr.select("td")]
            if tds:
                rows.append(tds)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["race_id"] = race_id
        return df


def collect_race_data(
    years: list[int],
    place_codes: list[str],
    interval: float = 3.0,
    save_path: str = "race_data.csv",
) -> pd.DataFrame:
    """
    指定年・開催場所の過去レースデータを一括収集してCSVに保存する。

    Args:
        years: 対象年リスト (例: [2022, 2023, 2024])
        place_codes: 競馬場コードリスト (例: ["05", "06"])
        interval: リクエスト間隔(秒)
        save_path: 保存先CSVパス

    Returns:
        収集したレースデータのDataFrame
    """
    scraper = NetkeibaScaper(interval=interval)
    all_dfs = []

    for year in years:
        for place in place_codes:
            logger.info("収集中: %d年 場所コード=%s", year, place)
            race_ids = scraper.get_race_id_list(year, place)
            for race_id in race_ids:
                df = scraper.get_race_result(race_id)
                if df is not None:
                    all_dfs.append(df)

    if not all_dfs:
        logger.warning("データが取得できませんでした")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_csv(save_path, index=False, encoding="utf-8-sig")
    logger.info("保存完了: %s (%d行)", save_path, len(result))
    return result
