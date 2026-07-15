$py = (Get-Content "C:\Users\morfii\Desktop\ganjuur\migrate.sh" -Raw -Encoding UTF8)
$b = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($py))
[System.IO.File]::WriteAllText("C:\Users\morfii\Desktop\ganjuur\migrate.sh.b64", $b, [Text.Encoding]::ASCII)
Write-Host "b64 len: " + $b.Length
