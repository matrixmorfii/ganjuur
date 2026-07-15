Write-Host "=== DISK ==="
Get-PSDrive -PSProvider FileSystem | Format-Table Name,@{N='Used(GB)';E={[math]::Round($_.Used/1GB,1)}},@{N='Free(GB)';E={[math]::Round($_.Free/1GB,1)}} -AutoSize
