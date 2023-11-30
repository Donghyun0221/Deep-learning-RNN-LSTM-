from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px

app = Flask(__name__)

# 엑셀 파일에서 데이터를 읽는 함수
def read_excel_data():
    # 엑셀 파일에서 데이터 읽기
    df = pd.read_excel('qkrdata60.xlsx')
    return df
def read_excel_data1():
    # 엑셀 파일에서 데이터 읽기
    df1 = pd.read_excel('predicted_prices.xlsx')
    return df1
def read_excel_data2():
    # 엑셀 파일에서 데이터 읽기
    df2 = pd.read_excel('qkrdataday.xlsx')
    return df2

def read_excel_data3():
    # 엑셀 파일에서 데이터 읽기
    df3 = pd.read_excel('predictedday_prices.xlsx')
    return df3

def read_excel_data4():
    # 엑셀 파일에서 데이터 읽기
    df4 = pd.read_excel('qkrdata10.xlsx')
    return df4

def read_excel_data5():
    # 엑셀 파일에서 데이터 읽기
    df5 = pd.read_excel('predicted10_prices.xlsx')
    return df5

# 차트를 그리는 함수
def create_chart():
    # 엑셀 파일에서 데이터 읽기
    data = read_excel_data()

    # Plotly를 사용하여 차트 생성
    fig = px.line(data, x='Date', y='Close', title='쿼크체인 시간별 차트')
    return fig.to_json()

def create_chart1():
    # 엑셀 파일에서 데이터 읽기
    data1 = read_excel_data1()

    # Plotly를 사용하여 차트 생성
    fig = px.line(data1, x='Date', y='Close', title='쿼크체인 시간별 예측 데이터')
    return fig.to_json()

def create_chart2():
    # 엑셀 파일에서 데이터 읽기
    data1 = read_excel_data2()

    # Plotly를 사용하여 차트 생성
    fig = px.line(data1, x='Date', y='Close', title='쿼크체인 일별 데이터')
    return fig.to_json()

def create_chart3():
    # 엑셀 파일에서 데이터 읽기
    data1 = read_excel_data3()

    # Plotly를 사용하여 차트 생성
    fig = px.line(data1, x='Date', y='Close', title='쿼크체인 일별 예측 데이터')
    return fig.to_json()

def create_chart4():
    # 엑셀 파일에서 데이터 읽기
    data1 = read_excel_data4()

    # Plotly를 사용하여 차트 생성
    fig = px.line(data1, x='Date', y='Close', title='쿼크체인 10분단위 데이터')
    return fig.to_json()

def create_chart5():
    # 엑셀 파일에서 데이터 읽기
    data1 = read_excel_data5()

    # Plotly를 사용하여 차트 생성
    fig = px.line(data1, x='Date', y='Close', title='쿼크체인 10분단위 예측 데이터')
    return fig.to_json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/show_chart', methods=['GET'])
def show_chart():
    chart_data = create_chart()
    return jsonify(chart_data)  # JSON으로 차트 데이터 반환

@app.route('/show1_chart', methods=['GET'])
def show1_chart():
    chart_data1 = create_chart1()  # 수정: create_chart1 함수 호출
    return jsonify(chart_data1)  # JSON으로 차트 데이터 반환

@app.route('/show2_chart', methods=['GET'])
def show2_chart():
    chart_data2 = create_chart2()  # 수정: create_chart1 함수 호출
    return jsonify(chart_data2)  # JSON으로 차트 데이터 반환

@app.route('/show3_chart', methods=['GET'])
def show3_chart():
    chart_data3 = create_chart3()  # 수정: create_chart1 함수 호출
    return jsonify(chart_data3)  # JSON으로 차트 데이터 반환

@app.route('/show4_chart', methods=['GET'])
def show4_chart():
    chart_data4 = create_chart4()  # 수정: create_chart1 함수 호출
    return jsonify(chart_data4)  # JSON으로 차트 데이터 반환

@app.route('/show5_chart', methods=['GET'])
def show5_chart():
    chart_data5 = create_chart5()  # 수정: create_chart1 함수 호출
    return jsonify(chart_data5)  # JSON으로 차트 데이터 반환


if __name__ == '__main__':
    app.run(debug=True)