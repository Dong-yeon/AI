# data_loader.py
from config import get_connection
import pandas as pd


def load_data():
    conn = get_connection()
    if conn is None:
        return None, None, None, None, None, None

    try:
        query0 = "SELECT PLAN_DT, QTY, CUST_NM FROM PRD_PLAN_MGT"

        query1 = (
            "SELECT "
            "ITEM_CD, "
            "PART_NO, "
            "WORK_START_DT, "
            "WORK_END_DT, "
            "REV, "
            "WORK_ORDER_MGT_SEQ "
            "FROM PRD_WORK_ORDER_MGT "
            "WHERE WORK_START_DT IS NOT NULL AND WORK_END_DT IS NOT NULL "
            "ORDER BY WORK_ORDER_MGT_SEQ"
        )

        query1_1 = (
            "SELECT "
            "WORK_ORDER_MGT_SEQ, "
            "MAX(ITEM_CD) AS ITEM_CD, "
            "BOP_CD, "
            "BOP_NO, "
            "MIN(SORT) AS MIN_SORT, "
            "MAX(SORT) AS MAX_SORT, "
            "MAX(WORK_START_DT) AS WORK_START_DT, "
            "MAX(WORK_END_DT) AS WORK_END_DT, "
            "MAX(TIME) AS TIME "
            "FROM PRD_WORK_PROC_MGT "
            "WHERE WORK_START_DT IS NOT NULL AND WORK_END_DT IS NOT NULL "
            "GROUP BY WORK_ORDER_MGT_SEQ, BOP_CD, BOP_NO "
            "ORDER BY WORK_ORDER_MGT_SEQ, CAST(MIN(SORT) AS INT)"
        )

        query2 = (
            "SELECT "
            "WORK_ORDER_MGT_SEQ, "
            "MAX(ITEM_CD) AS ITEM_CD, "
            "BOP_CD, "
            "BOP_NO, "
            "MIN(SORT) AS MIN_SORT, "
            "MAX(SORT) AS MAX_SORT, "
            "MAX(WORK_START_DT) AS WORK_START_DT, "
            "MAX(WORK_END_DT) AS WORK_END_DT "
            "FROM PRD_WORK_PROC_MGT "
            "WHERE WORK_START_DT IS NOT NULL AND WORK_END_DT IS NOT NULL "
            "GROUP BY WORK_ORDER_MGT_SEQ, BOP_CD, BOP_NO "
            "ORDER BY WORK_ORDER_MGT_SEQ, CAST(MIN(SORT) AS INT)"
        )

        query2_1 = (
            "SELECT "
            "WORK_ORDER_MGT_SEQ, "
            "MAX(ITEM_CD) AS ITEM_CD, "
            "BOP_CD, "
            "BOP_NO, "  # ← 여기 쉼표 빠졌었음
            "MIN(SORT) AS MIN_SORT, "
            "MAX(SORT) AS MAX_SORT "
            "FROM PRD_WORK_PROC_MGT AS A "
            "WHERE WORK_START_DT IS NULL AND WORK_END_DT IS NULL "
            "GROUP BY WORK_ORDER_MGT_SEQ, BOP_CD, BOP_NO "
            "ORDER BY WORK_ORDER_MGT_SEQ, CAST(MIN(SORT) AS INT)"
        )

        query3 = (
            "SELECT "
            "BOP_CD, "
            "MAX(WORK_NM) AS WORK_NM, "
            "MAX(USE_EQUIP) AS USE_EQUIP, "
            "COUNT(*) AS TOTAL_WORK_QTY, "
            "SUM(CASE WHEN STTS = 'M' THEN 1 ELSE 0 END) AS DEFECT_QTY, "
            "ROUND(CAST(SUM(CASE WHEN STTS = 'M' THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0), 3) AS DEFECT_RATE "
            "FROM PRD_WORK_PROC_MGT AS A "
            "WHERE WORK_NM IS NOT NULL "
            "GROUP BY BOP_CD, WORK_NM, USE_EQUIP "
            "ORDER BY WORK_NM "
        )

        df0 = pd.read_sql(query0, conn)
        df1 = pd.read_sql(query1, conn)
        df1_1 = pd.read_sql(query1_1, conn)
        df2 = pd.read_sql(query2, conn)
        df2_1 = pd.read_sql(query2_1, conn)
        df3 = pd.read_sql(query3, conn)

        print("데이터 로딩 완료")
        return df0, df1, df1_1, df2, df2_1, df3

    except Exception as e:
        print("데이터 조회 오류:", e)
        return None, None, None, None, None, None

    finally:
        conn.close()
