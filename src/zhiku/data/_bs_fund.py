"""独立进程拉 baostock 季报 / 预告 / 分红。登录会串台，一个子进程只 login 一次。不要 import zhiku。"""

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


def _rows(rs) -> list[dict]:
    if rs is None:
        return []
    fields = list(getattr(rs, "fields", []) or [])
    out = []
    while getattr(rs, "error_code", "1") == "0" and rs.next():
        raw = rs.get_row_data()
        if fields and len(fields) == len(raw):
            out.append(dict(zip(fields, raw)))
        else:
            out.append({"_raw": raw})
    return out


def _query(bs, fn_name: str, **kwargs) -> list[dict]:
    fn = getattr(bs, fn_name, None)
    if fn is None:
        return []
    try:
        rs = fn(**kwargs)
    except Exception:
        return []
    return _rows(rs)


def fetch(bs_code: str, start_year: int, end_year: int) -> dict:
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
            statements = []
            forecasts = []
            dividends = []
            for year in range(int(start_year), int(end_year) + 1):
                for quarter in (1, 2, 3, 4):
                    for kind, fn in (
                        ("income", "query_profit_data"),
                        ("balance", "query_balance_data"),
                        ("cashflow", "query_cash_flow_data"),
                    ):
                        rows = _query(bs, fn, code=bs_code, year=year, quarter=quarter)
                        for row in rows:
                            row["_kind"] = kind
                            statements.append(row)
                dividends.extend(_query(bs, "query_dividend_data", code=bs_code, year=str(year), yearType="report"))
            for row in _query(
                bs,
                "query_forecast_report",
                code=bs_code,
                start_date=f"{start_year}-01-01",
                end_date=f"{end_year}-12-31",
            ):
                forecasts.append(row)
            return {"statements": statements, "forecasts": forecasts, "dividends": dividends}
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
    if len(args) < 3:
        json.dump({"error": "用法: code start_year end_year"}, sys.stdout)
        return 2
    payload = fetch(args[0], int(args[1]), int(args[2]))
    json.dump(payload, sys.stdout, ensure_ascii=False)
    return 1 if payload.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
