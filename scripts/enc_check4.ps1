$py = "import urllib.request, json`n"
$py += "d=json.load(urllib.request.urlopen('http://127.0.0.1:6333/collections'))`n"
$py += 'for c in d["result"]["collections"]:' + "`n"
$py += '    print(f\'  {c["name"]:24s} {c.get("points_count","?"):>8} pts\')' + "`n"
$b = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($py))
[System.IO.File]::WriteAllText("C:\Users\morfii\Desktop\ganjuur\check4.b64", $b, [Text.Encoding]::ASCII)
Write-Host "len:" $b.Length
