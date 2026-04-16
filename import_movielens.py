"""
import_movielens.py
把 MovieLens 数据导入到 Railway MySQL

运行方式：
    python import_movielens.py
"""
import pandas as pd
import pymysql
import os

# ── Railway MySQL 连接配置 ─────────────────────
# 注意：Railway 外网连接需要用 PUBLIC URL
# 去 Railway MySQL 服务 → Variables → 找 MYSQL_PUBLIC_URL
# 格式是 mysql://user:password@host:port/database
# 这里手动填写各项

MYSQL_CONFIG = {
    "host":     "monorail.proxy.rlwy.net",
    "port":     45320,
    "user":     "root",
    "password": "DYXyCLhhbxGXaTJCYZhNJOSzzQXrtREA",
    "database": "railway",
    "charset":  "utf8mb4",
}

DATA_DIR = r"C:\Users\Lenovo\Desktop\rec_system\ml-latest-small"
MAX_MOVIES  = 500    # 导入电影数量
MAX_RATINGS = 20000  # 导入评分数量


def import_data():
    print("读取 MovieLens 数据...")
    movies_df  = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

    print(f"原始电影数: {len(movies_df)}, 原始评分数: {len(ratings_df)}")

    # 取评分最多的前 MAX_MOVIES 部电影
    top_movies = ratings_df["movieId"].value_counts().head(MAX_MOVIES).index
    movies_df  = movies_df[movies_df["movieId"].isin(top_movies)].copy()
    ratings_df = ratings_df[ratings_df["movieId"].isin(top_movies)].head(MAX_RATINGS).copy()

    # 处理类型格式（MovieLens 用 | 分隔，和我们格式一致）
    movies_df["genres"] = movies_df["genres"].str.replace("(no genres listed)", "Unknown")

    # 提取年份
    movies_df["year"] = movies_df["title"].str.extract(r'\((\d{4})\)').astype("Int64")
    movies_df["title"] = movies_df["title"].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

    print(f"筛选后电影数: {len(movies_df)}, 评分数: {len(ratings_df)}")

    conn = pymysql.connect(**MYSQL_CONFIG)
    print("MySQL 连接成功！")

    try:
        with conn.cursor() as cursor:
            # 清空旧数据
            cursor.execute("SET FOREIGN_KEY_CHECKS=0")
            cursor.execute("TRUNCATE TABLE ratings")
            cursor.execute("TRUNCATE TABLE movies")
            cursor.execute("TRUNCATE TABLE users")
            cursor.execute("SET FOREIGN_KEY_CHECKS=1")
            print("旧数据已清空")

            # 插入电影
            movie_id_map = {}
            for _, row in movies_df.iterrows():
                cursor.execute(
                    "INSERT INTO movies (title, genres, year) VALUES (%s, %s, %s)",
                    (row["title"], row["genres"], row["year"] if pd.notna(row["year"]) else None)
                )
                new_id = cursor.lastrowid
                movie_id_map[row["movieId"]] = new_id
            print(f"电影插入完成: {len(movie_id_map)} 部")

            # 插入用户
            user_ids = ratings_df["userId"].unique()
            user_id_map = {}
            for uid in user_ids:
                cursor.execute(
                    "INSERT INTO users (username) VALUES (%s)",
                    (f"user_{uid}",)
                )
                user_id_map[uid] = cursor.lastrowid
            print(f"用户插入完成: {len(user_id_map)} 个")

            # 插入评分
            # 批量插入评分
            rating_data = []
            for _, row in ratings_df.iterrows():
                new_movie_id = movie_id_map.get(row["movieId"])
                new_user_id = user_id_map.get(row["userId"])
                if new_movie_id and new_user_id:
                    rating_data.append((new_user_id, new_movie_id, row["rating"]))

            # 每次插入 1000 条
            batch_size = 1000
            for i in range(0, len(rating_data), batch_size):
                batch = rating_data[i:i + batch_size]
                cursor.executemany(
                    "INSERT IGNORE INTO ratings (user_id, movie_id, rating) VALUES (%s, %s, %s)",
                    batch
                )
                print(f"评分插入进度: {min(i + batch_size, len(rating_data))}/{len(rating_data)}")

            print(f"评分插入完成: {len(rating_data)} 条")

        conn.commit()
        print("\n✅ 导入完成！")

    finally:
        conn.close()


if __name__ == "__main__":
    import_data()
