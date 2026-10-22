"""独立进程拉 baostock 日线，便于超时 terminate。不要 import zhiku。"""

from __future__ import annotations

import json
import os
import sys


def _silence_fd1():
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    return saved


def _restore_fd1(saved: int) -> None:
    os.dup2(saved, 1)
    os.close(saved)


def query(bs_code: str, start: str, end: str, adj: str) -> dict:
    import baostock as bs

    saved = _silence_fd1()
    sink = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = sink
    try:
        lg = bs.login()
        if lg.error_code != "0":
            return {"error": f"登录失败: {lg.error_msg}"}
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag=adj,
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code not in {"0", ""}:
                return {"error": rs.error_msg or "query 失败", "rows": rows}
            return {"rows": rows}
        finally:
            try:
                bs.logout()
            except Exception:
                pass
    finally:
        try:
            sink.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_out, old_err
        sink.close()
        _restore_fd1(saved)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 4:
        json.dump({"error": "用法: code start end adj"}, sys.stdout)
        return 2
    payload = query(args[0], args[1], args[2], args[3])
    json.dump(payload, sys.stdout, ensure_ascii=False)
    return 1 if payload.get("error") and not payload.get("rows") else 0


if __name__ == "__main__":
    raise SystemExit(main())
