$p = [System.IO.File]::ReadAllText("C:\Users\morfii\Desktop\ganjuur\check_raw.py", [Text.Encoding]::UTF8)
$b = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($p))
[System.IO.File]::WriteAllText("C:\Users\morfii\Desktop\ganjuur\lq.b64", $b, [Text.Encoding]::ASCII)
Write-Host ("len:" + $b.Length)
