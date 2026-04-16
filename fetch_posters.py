"""
fetch_posters.py
从 TMDB API 获取电影海报和简介，存入 MySQL

运行方式：
    python fetch_posters.py
"""
import pymysql
import requests
import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
import time

TMDB_API_KEY = "21cbacc305a272311c7660f152257123"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"

MYSQL_CONFIG = {
    "host":     "monorail.proxy.rlwy.net",
    "port":     45320,
    "user":     "root",
    "password": "DYXyCLhhbxGXaTJCYZhNJOSzzQXrtREA",
    "database": "railway",
    "charset":  "utf8mb4",
}


def search_tmdb(title, year=None):
    """搜索 TMDB 获取电影信息"""
    params = {
        "api_key": TMDB_API_KEY,
        "query":   title,
        "language": "en-US",
    }
    if year:
        params["year"] = year

    try:
        r = requests.get(TMDB_SEARCH_URL, params=params, timeout=10)
        data = r.json()
        results = data.get("results", [])
        if results:
            return results[0]
    except Exception as e:
        print(f"  TMDB 请求失败: {e}")
    return None


def fetch_all_posters():
    conn = pymysql.connect(**MYSQL_CONFIG)
    print("MySQL 连接成功！")

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT movie_id, title, year FROM movies WHERE poster_path IS NULL")
            movies = cursor.fetchall()

        print(f"需要获取海报的电影: {len(movies)} 部")
        success = 0

        for i, (movie_id, title, year) in enumerate(movies):
            result = search_tmdb(title, year)
            if result:
                poster_path = result.get("poster_path", "")
                overview    = result.get("overview", "")
                tmdb_id     = result.get("id")

                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE movies
                        SET poster_path=%s, overview=%s, tmdb_id=%s
                        WHERE movie_id=%s
                    """, (poster_path, overview, tmdb_id, movie_id))
                conn.commit()
                success += 1

                if (i + 1) % 50 == 0:
                    print(f"进度: {i+1}/{len(movies)}, 成功: {success}")
            else:
                print(f"  未找到: {title}")

            # 避免触发 TMDB 限速（每秒最多 40 次请求）
            time.sleep(0.1)

        print(f"\n✅ 完成！成功获取 {success}/{len(movies)} 部电影的海报")

    finally:
        conn.close()


if __name__ == "__main__":
    fetch_all_posters()
