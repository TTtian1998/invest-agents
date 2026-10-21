"""独立进程拉 baostock 历史指数成分。登录会串台，每个查询单独 login。不要 import zhiku。"""

from __future__ import annotations

import json
import os
import sys

KIND_FN = {
    "hs300": "query_hs300_stocks",
    "zz500": "query_zz500_stocks",
    "sz50": "query_sz50_stocks",
}

INDEX_KINDS = {
    "000300": ["hs300"],
    "000905": ["zz500"],
    "000906": ["hs300", "zz500"],
    "000016": ["sz50"],
}


def _silence_fd1():
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    return saved


def _restore_fd1(saved: int) -> None:
    os.dup2(saved, 1)
    os.close(saved)


def _query_one(kind: str, day: str) -> dict:
    import baostock as bs

    fn_name = KIND_FN.get(kind)
    if not fn_name:
        return {"error": f"未知指数 {kind}", "rows": []}
    saved = _silence_fd1()
    sink = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = sink
    try:
        lg = bs.login()
        if lg.error_code != "0":
            return {"error": f"登录失败: {lg.error_msg}", "rows": []}
        try:
            fn = getattr(bs, fn_name)
            rs = fn(date=day)
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code not in {"0", ""} and not rows:
                return {"error": rs.error_msg or "query 失败", "rows": []}
            return {"rows": rows, "fields": list(getattr(rs, "fields", []) or [])}
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


def query_jobs(jobs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for job in jobs:
        index_code = str(job.get("index") or "")
        day = str(job.get("date") or "")
        kinds = INDEX_KINDS.get(index_code, [])
        if not kinds or not day:
            out.append({"index": index_code, "date": day, "error": "缺指数或日期", "codes": []})
            continue
        codes: list[str] = []
        errors: list[str] = []
        for kind in kinds:
            payload = _query_one(kind, day)
            if payload.get("error") and not payload.get("rows"):
                errors.append(f"{kind}:{payload.get('error')}")
                continue
            for row in payload.get("rows") or []:
                if len(row) >= 2:
                    codes.append(str(row[1]))
        if not codes:
            out.append(
                {
                    "index": index_code,
                    "date": day,
                    "error": "; ".join(errors) or "空成分",
                    "codes": [],
                }
            )
        else:
            out.append({"index": index_code, "date": day, "codes": codes, "error": None})
    return out


def main(argv: list[str] | None = None) -> int:
    del argv
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        json.dump({"error": "stdin 不是 JSON"}, sys.stdout, ensure_ascii=False)
        return 2
    jobs = payload.get("jobs") or []
    results = query_jobs(jobs)
    json.dump({"results": results}, sys.stdout, ensure_ascii=False)
    failed = sum(1 for r in results if r.get("error") and not r.get("codes"))
    return 1 if failed and failed == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
