from datetime import datetime
from enum import Enum

from dateutil.relativedelta import relativedelta
from pytz import timezone


class DateFormat(Enum):
    yyyy_mm = "%Y-%m"
    yyyy_mm_dd = "%Y-%m-%d"
    yyyy_mm_dd_hh_mm_ss = "%Y-%m-%d %H:%M:%S"


class DateValues:
    @staticmethod
    def get_current_date() -> str:
        """오늘 날짜를 반환합니다.

        Returns:
            str: 오늘 날짜 (%Y-%m-%d 포맷)
        """
        return datetime.now(timezone("Asia/Seoul")).strftime(
            DateFormat.yyyy_mm_dd.value
        )

    def get_before_7_days() -> str:
        """일주일 전 날짜를 반환합니다.

        Returns:
            str: 일주일 전 날짜 (%Y-%m-%d 포맷)
        """
        return (
            datetime.now(timezone("Asia/Seoul")) - relativedelta(days=7)
        ).strftime(DateFormat.yyyy_mm_dd.value)
