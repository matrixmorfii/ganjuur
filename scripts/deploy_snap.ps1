$p = [System.IO.File]::ReadAllText("C:\Users\morfii\Desktop\ganjuur\snapshot.sh", [Text.Encoding]::UTF8)
$b = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($p))
[System.IO.File]::WriteAllText("C:\Users\morfii\Desktop\ganjuur\snap.b64", $b, [Text.Encoding]::ASCII)
Write-Host ("len:" + $b.Length)
