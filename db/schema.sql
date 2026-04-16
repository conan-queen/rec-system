-- ─────────────────────────────────────────────
-- 推荐系统数据库建表语句
-- 运行方式：
--   mysql -u root -p rec_system < db/schema.sql
-- ─────────────────────────────────────────────

CREATE DATABASE IF NOT EXISTS rec_system
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE rec_system;

-- ── 电影表 ──────────────────────────────────
CREATE TABLE IF NOT EXISTS movies (
    movie_id    INT PRIMARY KEY AUTO_INCREMENT,
    title       VARCHAR(255) NOT NULL,
    genres      VARCHAR(255) NOT NULL,        -- 用 | 分隔，如 Action|Comedy
    year        INT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_title (title)
);

-- ── 用户表 ──────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id     INT PRIMARY KEY AUTO_INCREMENT,
    username    VARCHAR(100) UNIQUE NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── 用户评分表（核心交互数据）────────────────
CREATE TABLE IF NOT EXISTS ratings (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id     INT NOT NULL,
    movie_id    INT NOT NULL,
    rating      FLOAT NOT NULL CHECK (rating >= 1.0 AND rating <= 5.0),
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_user_movie (user_id, movie_id),   -- 每个用户对每部电影只能评分一次
    INDEX idx_user_id  (user_id),
    INDEX idx_movie_id (movie_id),
    INDEX idx_created  (created_at),
    FOREIGN KEY (user_id)  REFERENCES users(user_id)  ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE
);

-- ── 推荐日志表（记录推荐历史，用于 A/B 分析）──
CREATE TABLE IF NOT EXISTS recommendation_logs (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id     INT NOT NULL,
    movie_id    INT NOT NULL,
    score       FLOAT,
    source      VARCHAR(20),                  -- cf / cb / hybrid
    position    INT,                          -- 推荐位置（1=最靠前）
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id  (user_id),
    INDEX idx_created  (created_at)
);

-- ── 插入示例电影数据 ─────────────────────────
INSERT IGNORE INTO movies (title, genres, year) VALUES
('Inception',               'Sci-Fi|Action|Adventure',          2010),
('The Godfather',           'Drama|Crime',                      1972),
('Interstellar',            'Sci-Fi|Adventure|Drama',           2014),
('Toy Story',               'Animation|Comedy|Adventure',       1995),
('The Dark Knight',         'Action|Crime|Drama',               2008),
('Forrest Gump',            'Drama|Romance|Comedy',             1994),
('Pulp Fiction',            'Crime|Drama|Thriller',             1994),
('The Matrix',              'Sci-Fi|Action',                    1999),
('Schindler''s List',       'Drama|History',                    1993),
('Goodfellas',              'Crime|Drama',                      1990),
('Fight Club',              'Drama|Thriller',                   1999),
('The Silence of the Lambs','Thriller|Crime|Horror',            1991),
('Saving Private Ryan',     'War|Drama|Action',                 1998),
('Gladiator',               'Action|Adventure|Drama',           2000),
('The Shawshank Redemption','Drama',                            1994),
('Avengers: Endgame',       'Action|Adventure|Sci-Fi',          2019),
('Parasite',                'Drama|Thriller|Comedy',            2019),
('Your Name',               'Animation|Romance|Drama',          2016),
('Spirited Away',           'Animation|Adventure|Fantasy',      2001),
('The Lion King',           'Animation|Adventure|Drama',        1994);

-- ── 插入示例用户数据 ─────────────────────────
INSERT IGNORE INTO users (username) VALUES
('alice'), ('bob'), ('charlie'), ('diana'), ('eve');

-- ── 插入示例评分数据 ─────────────────────────
INSERT IGNORE INTO ratings (user_id, movie_id, rating) VALUES
(1, 1, 5.0), (1, 3, 4.5), (1, 5, 4.0), (1, 8, 3.5), (1, 15, 5.0),
(2, 2, 5.0), (2, 9, 4.5), (2, 10, 4.0), (2, 12, 3.5), (2, 7, 4.5),
(3, 4, 5.0), (3, 18, 4.5), (3, 19, 5.0), (3, 20, 4.0), (3, 6, 3.5),
(4, 16, 4.0), (4, 17, 5.0), (4, 11, 4.5), (4, 13, 4.0), (4, 14, 5.0),
(5, 1, 4.0), (5, 5, 4.5), (5, 8, 3.0), (5, 15, 4.5), (5, 16, 3.5);
