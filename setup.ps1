# Setup der Python-Umgebung (Windows / PowerShell)
#
# Aufruf im Hauptordner des Repos:
#     powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# Was passiert:
#   1. Python >= 3.10 suchen (bevorzugt 3.11)
#   2. virtuelle Umgebung .venv anlegen
#   3. Pakete aus requirements.txt installieren
#   4. spaCy-Modell de_core_news_lg herunterladen
#   5. 6_visualisations dauerhaft in den Suchpfad der Umgebung eintragen
#      (noetig, weil 7_statistical_analysis plot_style.py und
#       visualize_categories_v2.py von dort importiert)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Fail($msg) { Write-Host "FEHLER: $msg" -ForegroundColor Red; exit 1 }

# --- 1. Python finden -------------------------------------------------------
$pyCmd = $null
foreach ($candidate in @(@("py", "-3.11"), @("py", "-3.12"), @("py", "-3.10"), @("python"))) {
    try {
        $out = & $candidate[0] $candidate[1..($candidate.Count)] -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $ver = [version]$out.Trim()
            if ($ver -ge [version]"3.10") { $pyCmd = $candidate; $pyVer = $ver; break }
        }
    } catch { }
}
if (-not $pyCmd) { Fail "Kein Python >= 3.10 gefunden. Bitte Python 3.11 installieren (python.org)." }
Write-Host "Verwende Python $pyVer ($($pyCmd -join ' '))"
if ($pyVer -ne [version]"3.11") {
    Write-Host "Hinweis: getestet wurde mit Python 3.11." -ForegroundColor Yellow
}

# --- 2. venv anlegen --------------------------------------------------------
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Erstelle virtuelle Umgebung .venv ..."
    & $pyCmd[0] $pyCmd[1..($pyCmd.Count)] -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "venv konnte nicht erstellt werden." }
} else {
    Write-Host ".venv existiert bereits, wird weiterverwendet."
}

# --- 3. Pakete installieren -------------------------------------------------
Write-Host "Installiere Pakete aus requirements.txt ..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Installation der Pakete fehlgeschlagen." }

# --- 4. spaCy-Modell --------------------------------------------------------
Write-Host "Lade spaCy-Modell de_core_news_lg (ca. 500 MB) ..."
& $venvPython -m spacy download de_core_news_lg
if ($LASTEXITCODE -ne 0) { Fail "spaCy-Modell konnte nicht geladen werden." }

# --- 5. Suchpfad ------------------------------------------------------------
# .pth-Datei im site-packages der venv; relativer Pfad, damit der Ordner
# verschoben werden kann: .venv\Lib\site-packages -> ..\..\..\6_visualisations
$pth = Join-Path $PSScriptRoot ".venv\Lib\site-packages\thesis_paths.pth"
Set-Content -Path $pth -Value "..\..\..\6_visualisations" -Encoding ASCII

# --- Kontrolle --------------------------------------------------------------
& $venvPython -c "import plot_style, spacy; spacy.load('de_core_news_lg'); print('Import-Test OK')"
if ($LASTEXITCODE -ne 0) { Fail "Import-Test fehlgeschlagen." }

Write-Host ""
Write-Host "Fertig. Umgebung aktivieren mit:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "Danach die Befehle aus python_befehle.txt bzw. README.md ausfuehren."
