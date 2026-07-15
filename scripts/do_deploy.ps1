$plink = "C:\Program Files\PuTTY\plink.exe"
$user = "trinity"
$pass = "pass#1234"
$host = "192.168.0.55"
$shellScript = "C:\Users\morfii\Desktop\ganjuur\upload_and_restart.sh"
$b64File = "C:\Users\morfii\Desktop\ganjuur\longcat.b64"

$b64 = Get-Content $b64File -Raw -Encoding ASCII
$cmd = "echo '$b64' | $(shellEscape $shellScript)"

$plinkArgs = @(
    "-ssh",
    "-pw", $pass,
    "-no-antispoof",
    "-batch",
    "$user@$host"
)

& $plinkArgs[0] $plinkArgs[1] $plinkArgs[2] $plinkArgs[3] $plinkArgs[4] $plinkArgs[5] $plinkArgs[6] $plinkArgs[7] $plinkArgs[8] "echo $b64 | bash $shellScript"
