<#
.SYNOPSIS
    UAV Asynchronous Multi-Sensor EKF Fusion & Tactical Radar Launcher
.DESCRIPTION
    Taktik GUI konsolunu tek tikla dogrudan baslatan PowerShell betigi.
#>

param (
    [string]$Mode = "gui"
)

$Host.UI.RawUI.WindowTitle = "IHA Taktik Radar ve Cok Sensorlu EKF Fuzyon Konsolu"

function Show-Banner {
    Clear-Host
    Write-Host "=================================================================================" -ForegroundColor DarkGreen
    Write-Host "  _   _    _ __     __     ____      _    ____    _    ____                      " -ForegroundColor Green
    Write-Host " | | | |  / \\ \   / /    |  _ \    / \  |  _ \  / \  |  _ \                     " -ForegroundColor Green
    Write-Host " | | | | / _ \\ \ / /_____| |_) |  / _ \ | | | |/ _ \ | |_) |                    " -ForegroundColor Green
    Write-Host " | |_| |/ ___ \\ V /|_____|  _ <  / ___ \| |_| / ___ \|  _ <                     " -ForegroundColor Green
    Write-Host "  \___//_/   \_\\_/       |_| \_\/_/   \_\____/_/   \_\_| \_\                    " -ForegroundColor Green
    Write-Host ""
    Write-Host "  IHA HAVA SAHASI FARKINDALIK VE ASENKRON COK SENSORLU EKF FUZYON SISTEMI          " -ForegroundColor Cyan
    Write-Host "  Radar (10 Hz) + Kamera/EO (30 Hz) | Gecikmeler & OOSM Rollback/Replay Motoru     " -ForegroundColor DarkCyan
    Write-Host "=================================================================================" -ForegroundColor DarkGreen
    Write-Host ""
}

function Check-Dependencies {
    Write-Host "[1/2] Bagimliliklar denetleniyor..." -ForegroundColor Yellow
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Host "HATA: Sisteminizde Python bulunamadi! Lutfen Python 3.9+ kurunuz." -ForegroundColor Red
        Pause
        exit 1
    }

    $missing = python -c "import sys; pkgs=['numpy','scipy','matplotlib','PyQt6','pyqtgraph','OpenGL']; m=[p for p in pkgs if __import__('importlib.util').util.find_spec(p) is None]; print(','.join(m))"
    if ($missing) {
        Write-Host "Eksik paketler tespit edildi ($missing). Otomatik kuruluyor..." -ForegroundColor Yellow
        python -m pip install -r requirements.txt
    } else {
        Write-Host "[OK] Tum Python bagimliliklari eksiksiz mevcut." -ForegroundColor Green
    }
    Write-Host ""
}

function Run-GUI {
    Write-Host "[2/2] Canli Taktik Radar Konsolu Baslatiliyor (PyQt6 + PyQtGraph 60 FPS)..." -ForegroundColor Green
    python -m src.gui.app
}

function Run-Benchmark {
    Write-Host ">> Asama 2 Kiyaslama Benchmarki (Mod A vs Mod B vs Mod C) Kosturuluyor..." -ForegroundColor Cyan
    python src/main.py
    Write-Host ""
    Write-Host "Sonuc grafikleri results/ekf_oosm_comparison.png konumuna kaydedildi." -ForegroundColor Green
}

function Run-Tests {
    Write-Host ">> Tum Birim, Dinamik ve GUI Testleri Kosturuluyor..." -ForegroundColor Yellow
    python -m unittest discover tests -v
    Write-Host ""
}

function Install-Deps {
    Write-Host ">> Bagimliliklar guncelleniyor (pip install -r requirements.txt)..." -ForegroundColor Yellow
    python -m pip install --upgrade -r requirements.txt
    Write-Host "[OK] Kurulum tamamlandi." -ForegroundColor Green
}

# Banner ve Bagimlilik Kontrolu
Show-Banner
Check-Dependencies

# Dogrudan Hedef Modu Calistir (Varsayilan: GUI)
$targetMode = if ([string]::IsNullOrWhiteSpace($Mode)) { "gui" } else { $Mode.ToLower() }

if ($targetMode -eq "gui" -or $targetMode -eq "1") {
    Run-GUI
    exit 0
} elseif ($targetMode -eq "bench" -or $targetMode -eq "2") {
    Run-Benchmark
    exit 0
} elseif ($targetMode -eq "test" -or $targetMode -eq "3") {
    Run-Tests
    exit 0
} elseif ($targetMode -eq "install" -or $targetMode -eq "4") {
    Install-Deps
    exit 0
} else {
    Write-Host "Bilinmeyen mod: '$Mode'. Varsayilan olarak GUI baslatiliyor..." -ForegroundColor Yellow
    Run-GUI
    exit 0
}
