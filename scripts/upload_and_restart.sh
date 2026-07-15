#!/bin/bash
set -e

cd /home/trinity/ganjuur

# Read base64 from stdin and decode to longcat.py
base64 -d > longcat.py

echo "[SAVED] $(wc -c < longcat.py) bytes"
file longcat.py

# Verify Python syntax
/home/trinity/ganjuur/ganjuur_env/bin/python3 -c "import ast; ast.parse(open('longcat.py', encoding='utf8').read()); print('Syntax OK')"

# Ensure the backend column exists (migration)
/home/trinity/ganjuur/ganjuur_env/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('transliterations.db')
cols = {r[1] for r in conn.execute('PRAGMA table_info(analytics_jobs)')}
if 'backend' not in cols:
    conn.execute('ALTER TABLE analytics_jobs ADD COLUMN backend TEXT NOT NULL DEFAULT \"docker\"')
    conn.commit()
    print('backend column added')
else:
    print('backend column already exists')
conn.close()
"
