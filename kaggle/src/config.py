import pyodbc

DB_CONFIG = {
    "server": "125.189.149.10,14332",
    "database": "KCPARTS",
    "username": "sa",
    "password": "!@34link2us%^78"
}

def get_connection():
    try:
        conn_str = f'''
            DRIVER={{ODBC Driver 17 for SQL Server}};
            SERVER={DB_CONFIG["server"]};
            DATABASE={DB_CONFIG["database"]};
            UID={DB_CONFIG["username"]};
            PWD={DB_CONFIG["password"]};
            TrustServerCertificate=yes;
        '''
        conn = pyodbc.connect(conn_str)
        print("DB 연결 성공")
        return conn
    except Exception as e:
        print("DB 연결 실패:", e)
        return None
