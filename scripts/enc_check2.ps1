$py = "import subprocess`n"
$py += 'print("=== SERVICE ===")' + "`n"
$py += 'print(subprocess.check_output("systemctl is-active ganjuur.service",shell=True).decode().strip())' + "`n"
$py += 'print("=== INGEST PROC ===")' + "`n"
$py += 'r=subprocess.check_output("pgrep -af ingest_from_bdrc",shell=True).decode().strip()' + "`n"
$py += 'print(r if r else "NOT RUNNING")' + "`n"
$py += 'print("=== INGEST LOG (last 8) ===")' + "`n"
$py += 'print(subprocess.check_output("tail -8 /home/trinity/ganjuur/bdrc_ingest.log",shell=True,stderr=subprocess.DEVNULL).decode() or "(empty log)")' + "`n"
$b = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($py))
[System.IO.File]::WriteAllText("C:\Users\morfii\Desktop\ganjuur\check2.b64", $b, [Text.Encoding]::ASCII)
Write-Host "len:" $b.Length
