# =============================================================
# setup.ps1 — Installation & Lancement
# Système de Surveillance Intelligente — ENSA Béni Mellal
# RTX 5060 / Windows / CUDA 13.x — via DirectML
#
# USAGE :
#   git clone https://github.com/othmane1244/Phase-4.git
#   cd Phase-4
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# =============================================================

Set-StrictMode -Off
$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Surveillance IA — Setup"

# ─── Couleurs helpers ─────────────────────────────────────────
function Write-Header  { param($t) Write-Host "`n══ $t ══" -ForegroundColor Cyan }
function Write-Ok      { param($t) Write-Host "  ✓  $t" -ForegroundColor Green }
function Write-Warn    { param($t) Write-Host "  ⚠  $t" -ForegroundColor Yellow }
function Write-Fail    { param($t) Write-Host "  ✗  $t" -ForegroundColor Red }
function Write-Info    { param($t) Write-Host "     $t" -ForegroundColor Gray }
function Pause-Key     { Write-Host "`nAppuyez sur une touche pour continuer..." -ForegroundColor DarkGray; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") }

# ─── Bannière ─────────────────────────────────────────────────
Clear-Host
Write-Host @"

  ╔══════════════════════════════════════════════════════════╗
  ║     Système de Surveillance Intelligente par Caméra      ║
  ║     ENSA Béni Mellal — IA & Cybersécurité 2025-2026      ║
  ║                                                          ║
  ║     ALIOUALI Othmane  •  ELJARIDA Saad Eddine            ║
  ╚══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

$ROOT = $PSScriptRoot   # dossier Phase-4/

# =============================================================
# ÉTAPE 1 — Vérification des prérequis
# =============================================================
Write-Header "ÉTAPE 1 — Vérification des prérequis"

# Python 3.11+
try {
    $pyver = python --version 2>&1
    if ($pyver -match "(\d+)\.(\d+)") {
        $maj = [int]$Matches[1]; $min = [int]$Matches[2]
        if ($maj -lt 3 -or ($maj -eq 3 -and $min -lt 11)) {
            Write-Fail "Python $pyver détecté — Python 3.11+ requis"
            Write-Info "Télécharger : https://www.python.org/downloads/"
            exit 1
        }
        Write-Ok "Python $pyver"
    }
} catch {
    Write-Fail "Python introuvable — installer Python 3.11+ et l'ajouter au PATH"
    Write-Info "https://www.python.org/downloads/"
    exit 1
}

# Node.js 20+
try {
    $nodever = node --version 2>&1
    if ($nodever -match "v(\d+)") {
        $nodeMaj = [int]$Matches[1]
        if ($nodeMaj -lt 20) {
            Write-Fail "Node.js $nodever détecté — Node.js 20+ requis"
            Write-Info "Télécharger : https://nodejs.org/"
            exit 1
        }
        Write-Ok "Node.js $nodever"
    }
} catch {
    Write-Fail "Node.js introuvable — installer Node.js 20+ et l'ajouter au PATH"
    Write-Info "https://nodejs.org/"
    exit 1
}

# npm
try {
    $npmver = npm --version 2>&1
    Write-Ok "npm $npmver"
} catch {
    Write-Fail "npm introuvable (normalement inclus avec Node.js)"
    exit 1
}

# GPU / nvidia-smi (optionnel)
try {
    $smi = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    Write-Ok "GPU détecté : $($smi.Trim())"
} catch {
    Write-Warn "nvidia-smi introuvable — DirectML fonctionnera quand même (via DirectX 12)"
}

# =============================================================
# ÉTAPE 2 — Environnement virtuel Python
# =============================================================
Write-Header "ÉTAPE 2 — Environnement virtuel Python"

$VENV = Join-Path $ROOT "hailo_env"
$PIP  = Join-Path $VENV "Scripts\pip.exe"
$PY   = Join-Path $VENV "Scripts\python.exe"

if (Test-Path $VENV) {
    Write-Warn "hailo_env déjà présent — réutilisation"
} else {
    Write-Info "Création de l'environnement virtuel..."
    python -m venv $VENV
    Write-Ok "hailo_env créé"
}

# Mise à jour pip
Write-Info "Mise à jour pip..."
& $PY -m pip install --upgrade pip --quiet
Write-Ok "pip à jour"

# =============================================================
# ÉTAPE 3 — Installation des dépendances Python
# =============================================================
Write-Header "ÉTAPE 3 — Dépendances Python (DirectML pour RTX 5060 / CUDA 13.x)"

# Supprimer les anciens packages onnxruntime conflictuels
Write-Info "Nettoyage anciens packages onnxruntime..."
& $PIP uninstall onnxruntime onnxruntime-gpu -y 2>$null | Out-Null

Write-Info "Installation depuis requirements.txt..."
Write-Info "(peut prendre 2-5 minutes selon la connexion)"
& $PIP install -r (Join-Path $ROOT "requirements.txt") --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Fail "Erreur lors de pip install — voir messages ci-dessus"
    exit 1
}
Write-Ok "Toutes les dépendances installées"

# Vérification DirectML
Write-Info "Vérification du provider GPU..."
$providers = & $PY -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))" 2>&1
if ($providers -match "DmlExecutionProvider") {
    Write-Ok "DirectML disponible — GPU sera utilisé pour l'inférence"
} elseif ($providers -match "CUDAExecutionProvider") {
    Write-Ok "CUDA disponible — GPU sera utilisé pour l'inférence"
} else {
    Write-Warn "Seul CPUExecutionProvider détecté — vérifier l'installation"
    Write-Info "Providers disponibles : $providers"
}

# =============================================================
# ÉTAPE 4 — Fichier .env (Supabase)
# =============================================================
Write-Header "ÉTAPE 4 — Configuration .env (Supabase)"

$ENV_FILE = Join-Path $ROOT ".env"

if (Test-Path $ENV_FILE) {
    Write-Warn ".env déjà présent — ignoré (supprimer manuellement pour reconfigurer)"
} else {
    Write-Host ""
    Write-Host "  Supabase est optionnel — sans lui, les alertes sont stockées" -ForegroundColor Yellow
    Write-Host "  en mémoire locale (mode simulation)." -ForegroundColor Yellow
    Write-Host ""
    $useSupabase = Read-Host "  Configurer Supabase maintenant ? (o/N)"

    if ($useSupabase -match "^[oOyY]") {
        $SUPABASE_URL = Read-Host "  SUPABASE_URL (ex: https://xxx.supabase.co)"
        $SUPABASE_KEY = Read-Host "  SUPABASE_SERVICE_KEY (clé service_role)"

        @"
SUPABASE_URL=$SUPABASE_URL
SUPABASE_SERVICE_KEY=$SUPABASE_KEY
API_HOST=127.0.0.1
API_PORT=8000
CAMERA_ID=cam_01_simulation
"@ | Set-Content $ENV_FILE -Encoding UTF8
        Write-Ok ".env créé avec les clés Supabase"
    } else {
        @"
# Supabase désactivé — mode simulation (alertes en mémoire locale)
# Décommenter et remplir pour activer la persistance cloud :
# SUPABASE_URL=https://votre-projet.supabase.co
# SUPABASE_SERVICE_KEY=votre-clé-service-role
API_HOST=127.0.0.1
API_PORT=8000
CAMERA_ID=cam_01_simulation
"@ | Set-Content $ENV_FILE -Encoding UTF8
        Write-Ok ".env créé en mode simulation (sans Supabase)"
    }
}

# =============================================================
# ÉTAPE 5 — Modèle YOLO11n (téléchargement + conversion ONNX)
# =============================================================
Write-Header "ÉTAPE 5 — Modèle YOLO11n → ONNX"

$ONNX_FILE = Join-Path $ROOT "yolo11n.onnx"

if (Test-Path $ONNX_FILE) {
    Write-Ok "yolo11n.onnx déjà présent — étape ignorée"
} else {
    Write-Info "Téléchargement de yolo11n.pt et conversion en ONNX..."
    Write-Info "(première exécution : ~200 MB à télécharger)"
    Set-Location $ROOT
    & $PY convert_to_onnx.py
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Erreur lors de la conversion ONNX"
        exit 1
    }
    Write-Ok "yolo11n.onnx généré"
}

# =============================================================
# ÉTAPE 6 — Dashboard Next.js
# =============================================================
Write-Header "ÉTAPE 6 — Dashboard Next.js 14"

$DASH = Join-Path $ROOT "dashboard"
$NM   = Join-Path $DASH "node_modules"

if (Test-Path $NM) {
    Write-Warn "node_modules déjà présent — npm install ignoré"
} else {
    Write-Info "Installation des dépendances Node.js..."
    Push-Location $DASH
    npm install --silent
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Erreur lors de npm install"
        Pop-Location; exit 1
    }
    Pop-Location
    Write-Ok "Dépendances dashboard installées"
}

# Fichier .env.local pour le dashboard
$DASH_ENV = Join-Path $DASH ".env.local"
if (Test-Path $DASH_ENV) {
    Write-Warn ".env.local dashboard déjà présent — ignoré"
} else {
    Write-Host ""
    Write-Host "  Pour que le dashboard affiche les alertes en temps réel," -ForegroundColor Yellow
    Write-Host "  il faut les clés Supabase publiques (anon key)." -ForegroundColor Yellow
    Write-Host "  Sans elles, le dashboard démarre mais sans données Supabase." -ForegroundColor Yellow
    Write-Host ""
    $useDashSupabase = Read-Host "  Configurer Supabase pour le dashboard ? (o/N)"

    if ($useDashSupabase -match "^[oOyY]") {
        $DASH_URL  = Read-Host "  NEXT_PUBLIC_SUPABASE_URL"
        $DASH_ANON = Read-Host "  NEXT_PUBLIC_SUPABASE_ANON_KEY (clé anon, pas service_role)"
        @"
NEXT_PUBLIC_SUPABASE_URL=$DASH_URL
NEXT_PUBLIC_SUPABASE_ANON_KEY=$DASH_ANON
"@ | Set-Content $DASH_ENV -Encoding UTF8
        Write-Ok ".env.local dashboard créé"
    } else {
        @"
# Décommenter pour activer Supabase dans le dashboard :
# NEXT_PUBLIC_SUPABASE_URL=https://votre-projet.supabase.co
# NEXT_PUBLIC_SUPABASE_ANON_KEY=votre-cle-anon
"@ | Set-Content $DASH_ENV -Encoding UTF8
        Write-Ok ".env.local dashboard créé (Supabase désactivé)"
    }
}

# =============================================================
# RÉCAPITULATIF
# =============================================================
Write-Host @"

  ╔══════════════════════════════════════════════════════════╗
  ║                  INSTALLATION TERMINÉE ✓                 ║
  ╚══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Host "  Que voulez-vous faire ?" -ForegroundColor White
Write-Host "  [1] Lancer tout le système (3 fenêtres séparées)" -ForegroundColor White
Write-Host "  [2] Lancer uniquement l'API + Simulateur (sans dashboard)" -ForegroundColor White
Write-Host "  [3] Quitter (lancement manuel)" -ForegroundColor White
Write-Host ""
$choice = Read-Host "  Votre choix (1/2/3)"

# =============================================================
# LANCEMENT
# =============================================================
$ACTIVATE = Join-Path $VENV "Scripts\Activate.ps1"

switch ($choice) {

    "1" {
        Write-Host ""
        Write-Info "Ouverture de 3 fenêtres..."

        # Fenêtre 1 — API FastAPI
        Start-Process powershell -ArgumentList @(
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-Command",
            "& { `$Host.UI.RawUI.WindowTitle = 'API FastAPI — :8000'; Set-Location '$ROOT'; & '$ACTIVATE'; Write-Host '>>> API FastAPI démarrée sur http://127.0.0.1:8000' -ForegroundColor Cyan; Write-Host '    Docs : http://127.0.0.1:8000/docs' -ForegroundColor Gray; uvicorn main:app --reload --host 127.0.0.1 --port 8000 }"
        )

        Start-Sleep -Seconds 3   # laisser FastAPI démarrer avant le simulateur

        # Fenêtre 2 — Simulateur webcam
        Start-Process powershell -ArgumentList @(
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-Command",
            "& { `$Host.UI.RawUI.WindowTitle = 'Simulateur — YOLO + DirectML'; Set-Location '$ROOT'; & '$ACTIVATE'; Write-Host '>>> Simulateur pipeline (webcam + ONNX DirectML)' -ForegroundColor Cyan; Write-Host '    Appuyer sur Q dans la fenêtre vidéo pour quitter' -ForegroundColor Gray; python simulator.py }"
        )

        # Fenêtre 3 — Dashboard Next.js
        Start-Process powershell -ArgumentList @(
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-Command",
            "& { `$Host.UI.RawUI.WindowTitle = 'Dashboard Next.js — :3000'; Set-Location '$DASH'; Write-Host '>>> Dashboard Next.js 14 sur http://localhost:3000' -ForegroundColor Cyan; npm run dev }"
        )

        Write-Host ""
        Write-Ok "3 fenêtres lancées"
        Write-Host ""
        Write-Host "  URLs utiles :" -ForegroundColor White
        Write-Host "    Dashboard    →  http://localhost:3000" -ForegroundColor Cyan
        Write-Host "    API Swagger  →  http://127.0.0.1:8000/docs" -ForegroundColor Cyan
        Write-Host "    Alertes live →  http://127.0.0.1:8000/alerts/buffer/" -ForegroundColor Cyan
        Write-Host "    Stats        →  http://127.0.0.1:8000/stats/" -ForegroundColor Cyan
    }

    "2" {
        Write-Host ""
        Write-Info "Ouverture de 2 fenêtres (API + Simulateur)..."

        Start-Process powershell -ArgumentList @(
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-Command",
            "& { `$Host.UI.RawUI.WindowTitle = 'API FastAPI — :8000'; Set-Location '$ROOT'; & '$ACTIVATE'; Write-Host '>>> API FastAPI démarrée sur http://127.0.0.1:8000' -ForegroundColor Cyan; uvicorn main:app --reload --host 127.0.0.1 --port 8000 }"
        )

        Start-Sleep -Seconds 3

        Start-Process powershell -ArgumentList @(
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-Command",
            "& { `$Host.UI.RawUI.WindowTitle = 'Simulateur — YOLO + DirectML'; Set-Location '$ROOT'; & '$ACTIVATE'; Write-Host '>>> Simulateur pipeline (webcam + ONNX DirectML)' -ForegroundColor Cyan; python simulator.py }"
        )

        Write-Host ""
        Write-Ok "2 fenêtres lancées"
        Write-Host ""
        Write-Host "  URLs utiles :" -ForegroundColor White
        Write-Host "    API Swagger  →  http://127.0.0.1:8000/docs" -ForegroundColor Cyan
        Write-Host "    Alertes live →  http://127.0.0.1:8000/alerts/buffer/" -ForegroundColor Cyan
    }

    default {
        Write-Host ""
        Write-Host "  Lancement manuel — commandes :" -ForegroundColor White
        Write-Host ""
        Write-Host "    # Terminal 1 — API :" -ForegroundColor Gray
        Write-Host "    cd '$ROOT'" -ForegroundColor DarkCyan
        Write-Host "    .\hailo_env\Scripts\Activate.ps1" -ForegroundColor DarkCyan
        Write-Host "    uvicorn main:app --reload --host 127.0.0.1 --port 8000" -ForegroundColor DarkCyan
        Write-Host ""
        Write-Host "    # Terminal 2 — Simulateur :" -ForegroundColor Gray
        Write-Host "    cd '$ROOT' ; .\hailo_env\Scripts\Activate.ps1" -ForegroundColor DarkCyan
        Write-Host "    python simulator.py" -ForegroundColor DarkCyan
        Write-Host ""
        Write-Host "    # Terminal 3 — Dashboard :" -ForegroundColor Gray
        Write-Host "    cd '$ROOT\dashboard'" -ForegroundColor DarkCyan
        Write-Host "    npm run dev" -ForegroundColor DarkCyan
    }
}

Write-Host ""
