$py = "import subprocess, time`n"
$py += "for i in range(6):`n"
$py += "    sz = subprocess.check_output('stat -c %s /home/trinity/ganjuur/bdrc_ingest.log', shell=True).decode().strip()`n"
$py += "    print('t+%ds  log_bytes=%s' % (i*10, sz))`n"
$py += "    r = subprocess.check_output('pgrep -af ingest_from_bdrc | grep -v pgrep', shell=True).decode().strip()`n"
$py += "    print('   proc:', r.splitlines()[-1] if r else 'NONE')`n"
$py += "    if int(sz) > 0:`n"
$py += "        print(subprocess.check_output('tail -4 /home/trinity/ganjuur/bdrc_ingest.log', shell=True, stderr=subprocess.DEVNULL).decode())`n"
$py += "        break`n"
$py += "    time.sleep(10)`n"
$b = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($py))
[System.IO.File]::WriteAllText("C:\Users\morfii\Desktop\ganjuur\check3.b64", $b, [Text.Encoding]::ASCII)
Write-Host "len:" $b.Length
