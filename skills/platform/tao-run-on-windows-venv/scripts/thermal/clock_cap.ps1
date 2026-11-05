# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# clock_cap.ps1 - apply, restore or probe the GPU SM clock cap and power limit with nvidia-smi (validated on an RTX Spark N1X).
# Usage: powershell -ExecutionPolicy Bypass -File clock_cap_probe.ps1 [-Apply <mhz>] [-Restore] [-Power <watts>] [-Plan <guid-or-name>]
#   (no args)   probe: try -lgc 1800/1600/1400 and -pl, read back each, restore with -rgc / -pl default, print a table
#   -Apply 1600 set an SM clock cap and leave it (used around a measured run); -Restore resets clocks and power limit
# Never leaves a cap applied in probe mode. Records everything to -LogPath (default .\evidence\clock_cap_probe.log).
param([int]$Apply = 0, [switch]$Restore, [int]$Power = 0, [string]$Plan = '',
      [string]$LogPath = (Join-Path (Get-Location) 'evidence\clock_cap_probe.log'))
$log = $LogPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
function Log($s) { $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $s; Write-Output $line; Add-Content -Path $log -Value $line }
function Smi([string[]]$smiArgs) { $out = & nvidia-smi @smiArgs 2>&1 | Out-String; return $out.Trim() }  # ($args is reserved in PowerShell)
function ReadBack() { Smi @('-i','0','--query-gpu=clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit,power.default_limit,power.max_limit,power.min_limit,temperature.gpu','--format=csv,noheader') }
Log "driver: $(Smi @('--query-gpu=driver_version','--format=csv,noheader'))  readback(sm,max.sm,app.gfx,power.limit,default,max,min,temp): $(ReadBack)"
if ($Restore) { Log "rgc: $(Smi @('-i','0','-rgc'))"; Log "pl default: $(Smi @('-i','0','-pl',(Smi @('-i','0','--query-gpu=power.default_limit','--format=csv,noheader,nounits')).Trim()))"; Log "after restore: $(ReadBack)"; exit 0 }
if ($Apply -gt 0) { Log "lgc $Apply : $(Smi @('-i','0','-lgc',"$Apply"))"; Log "after apply: $(ReadBack)"; exit 0 }
if ($Power -gt 0) { Log "pl $Power : $(Smi @('-i','0','-pl',"$Power"))"; Log "after pl: $(ReadBack)"; exit 0 }
if ($Plan) { Log "powercfg setactive $Plan : $(& powercfg /SETACTIVE $Plan 2>&1 | Out-String)"; Log "active scheme: $(& powercfg /GETACTIVESCHEME | Out-String)"; exit 0 }
# probe mode
Log "supported clocks (head): $((Smi @('-i','0','-q','-d','SUPPORTED_CLOCKS')) -split "`n" | Select-Object -First 12 | Out-String)"
foreach ($mhz in 1800, 1600, 1400) {
  Log "lgc $mhz -> $(Smi @('-i','0','-lgc',"$mhz"))"
  Log "   readback: $(ReadBack)"
  Log "   rgc -> $(Smi @('-i','0','-rgc'))"
}
Log "lmc 1000 (memory clock, informational) -> $(Smi @('-i','0','-lmc','1000'))"; Log "   rmc -> $(Smi @('-i','0','-rmc'))"
$pl = (Smi @('-i','0','--query-gpu=power.default_limit','--format=csv,noheader,nounits')).Trim()
Log "power.default_limit = $pl ; pl 40 -> $(Smi @('-i','0','-pl','40'))"
Log "   readback: $(ReadBack)"
if ($pl -match '^[\d.]+$') { Log "   pl default ($pl) -> $(Smi @('-i','0','-pl',$pl))" } else { Log "   (no numeric default power limit; nothing to restore)" }
Log "power plans: $(& powercfg /LIST 2>&1 | Out-String)"
Log "active scheme: $(& powercfg /GETACTIVESCHEME 2>&1 | Out-String)"
Log "final readback: $(ReadBack)"

