CREATE TABLE IF NOT EXISTS config (
  id INT PRIMARY KEY DEFAULT 1,
  session_image_cap INT NOT NULL DEFAULT 20,
  min_required_mixed INT NOT NULL DEFAULT 4,
  min_required_same INT NOT NULL DEFAULT 3,
  file_size_limit_mb INT NOT NULL DEFAULT 25
);
INSERT INTO config(id) VALUES(1) ON CONFLICT (id) DO NOTHING;

