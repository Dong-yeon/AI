# main.py
from predictor import plot_monthly_qty_by_customer
from predictor import forecast_monthly_qty
from predictor import analyze_process_times
from predictor import estimate_process_times
from predictor import 불량률예측
from data_loder import load_data


def main():
    df0, df1, df1_1, df2, df2_1, df3 = load_data()

    # 수주 계획 관련 처리
    if df0 is not None and not df0.empty:
        print("수주 예측 및 시각화 실행")
        plot_monthly_qty_by_customer(df0) # 3월까지 시각화
        # forecast_monthly_qty(df0) # 4월 예측
    else:
        print("수주 계획 데이터(df0)가 없습니다. 종료합니다.")

    # 생산 흐름 예측 처리
    # if df1 is not None and df2 is not None and not df1.empty and not df2.empty:
    #     print("생산 흐름 예측")
    #     analyze_process_times(df1, df1_1)
    # else:
    #     print("생산 계획 또는 공정 데이터가 없습니다.")

    # 공정 종료시간 예측 처리
    # if df2 is not None and df2_1 is not None and not df2.empty and not df2_1.empty:
    #     print("공정 종료시간 예측")
    #     result = estimate_process_times(df2, df2_1)

    # if df3 is not None and not df3.empty and not df3.empty:
    #     불량률예측(df3)

if __name__ == "__main__":
    main()
