#!/bin/bash
cd /home/trinity/ganjuur
base64 -d > longcat.py << 'B64EOF'
B64EOF
echo "[WRITTEN] $(wc -c < longcat.py) bytes"
file longcat.py
/home/trinity/ganjuur/ganjuur_env/bin/python3 -c "import ast; ast.parse(open('longcat.py', encoding='utf8').read()); print('Syntax OK')"
