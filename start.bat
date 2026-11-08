@echo off
title IHA Taktik Radar ve Cok Sensorlu EKF Fuzyon Konsolu
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Mode gui %*
