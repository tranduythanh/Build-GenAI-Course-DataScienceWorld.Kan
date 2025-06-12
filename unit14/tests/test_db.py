import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import unit14.db as db


class TestDB(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        db.DB_PATH = Path(path)

    def tearDown(self):
        os.remove(db.DB_PATH)

    def test_log_session(self):
        db.init_db()
        db.log_session("img.png", "text", None, "sample", "score", "fb")
        conn = sqlite3.connect(db.DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions")
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
