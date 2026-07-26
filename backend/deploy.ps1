# backend/deploy.ps1 - SentiMetric SAM Deployment Automation
#
# Usage (from backend/ directory):
#   .\deploy.ps1 local    - stage + sam build + deploy to floci (LocalStack on :4566)
#   .\deploy.ps1 api      - stage + sam local start-api against floci
#   .\deploy.ps1 prod     - stage + sam build + deploy to real AWS
#
# Layout of deploy/:
#   deploy/src/            <- backend/src/* (FastAPI Lambda, CodeUri: src/)
#   deploy/lambda/         <- NOT copied; CodeUri: ../../lambda/ reaches root/lambda/ directly
#   deploy/template.yaml   <- source of truth, NOT overwritten by this script
#   deploy/samconfig.toml  <- copied from backend/samconfig.toml

$ErrorActionPreference = "Stop"

# Full paths - SAM/AWS are installed but not always on PATH in every shell session
$SAM = "C:\Program Files\Amazon\AWSSAMCLI\bin\sam.cmd"
$AWS = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"

$DEPLOY_DIR  = "deploy"
$BACKEND_SRC = "src"
$FLOCI_URL   = "http://localhost:4566"

# ── Stage: export requirements + copy src into deploy/ ────────────────────
function Sync-Folders {
    Write-Host ""
    Write-Host "==> Exporting requirements.txt from uv.lock (no dev deps)..." -ForegroundColor Cyan
    uv export --no-dev --no-hashes -o "$BACKEND_SRC\requirements.txt"
    if ($LASTEXITCODE -ne 0) { throw "uv export failed" }

    Write-Host "==> Staging deploy/src ..." -ForegroundColor Cyan

    # Clean only src/ inside deploy/ - leave template.yaml and lambda/ alone
    if (Test-Path "$DEPLOY_DIR\src") { Remove-Item -Recurse -Force "$DEPLOY_DIR\src" }
    New-Item -ItemType Directory -Force -Path "$DEPLOY_DIR\src" | Out-Null

    # Copy backend/src -> deploy/src (includes the requirements.txt just exported)
    Copy-Item -Path "$BACKEND_SRC\*" -Destination "$DEPLOY_DIR\src\" -Recurse -Force

    # Copy samconfig (template.yaml stays as-is in deploy/ - it is the source of truth)
    Copy-Item -Path "samconfig.toml" -Destination "$DEPLOY_DIR\samconfig.toml" -Force

    Write-Host "    Staged OK: deploy/src + samconfig.toml" -ForegroundColor Green
}

# ── Local: build + deploy to floci (LocalStack) ───────────────────────────
function Deploy-Local {
    Sync-Folders
    Write-Host ""
    Write-Host "==> Building with SAM (no container)..." -ForegroundColor Cyan
    Push-Location $DEPLOY_DIR
    try {
        & $SAM build
        if ($LASTEXITCODE -ne 0) { throw "sam build failed" }

        Write-Host ""
        Write-Host "==> Deploying to floci/LocalStack ($FLOCI_URL)..." -ForegroundColor Cyan
        # sam deploy has no --endpoint-url flag; LocalStack intercepts via env var
        $env:AWS_ENDPOINT_URL        = $FLOCI_URL
        $env:AWS_ACCESS_KEY_ID       = "test"
        $env:AWS_SECRET_ACCESS_KEY   = "test"
        $env:AWS_DEFAULT_REGION      = "us-east-1"
        try {
            & $SAM deploy `
                --stack-name sentimetric-local `
                --resolve-s3 `
                --capabilities CAPABILITY_IAM `
                --no-confirm-changeset
            if ($LASTEXITCODE -ne 0) { throw "sam deploy (local) failed" }

            Write-Host ""
            Write-Host "==> Verifying deployed functions in LocalStack..." -ForegroundColor Cyan
            & $AWS lambda list-functions --endpoint-url $FLOCI_URL --query "Functions[].{Name:FunctionName,State:State,Runtime:Runtime}" --output table
        } finally {
            Remove-Item Env:\AWS_ENDPOINT_URL      -ErrorAction SilentlyContinue
            Remove-Item Env:\AWS_ACCESS_KEY_ID     -ErrorAction SilentlyContinue
            Remove-Item Env:\AWS_SECRET_ACCESS_KEY -ErrorAction SilentlyContinue
            Remove-Item Env:\AWS_DEFAULT_REGION    -ErrorAction SilentlyContinue
        }
    }
    finally { Pop-Location }

    Write-Host ""
    Write-Host "Local deploy complete." -ForegroundColor Green
}

# ── API: start SAM local HTTP API server ──────────────────────────────────
function Start-Api-Local {
    Sync-Folders
    Write-Host ""
    Write-Host "==> Starting local API on http://127.0.0.1:3000 ..." -ForegroundColor Cyan
    Push-Location $DEPLOY_DIR
    try {
        $envFile = if (Test-Path "env.json") { "env.json" } elseif (Test-Path "..\env.json") { "..\env.json" } else { $null }
        $envArg  = if ($envFile) { @("--env-vars", $envFile) } else { @() }
        & $SAM local start-api --warm-containers EAGER @envArg
    }
    finally { Pop-Location }
}

# ── Prod: build + deploy to real AWS ──────────────────────────────────────
function Deploy-Prod {
    Sync-Folders
    Write-Host ""
    Write-Host "==> Building for AWS..." -ForegroundColor Cyan
    Push-Location $DEPLOY_DIR
    try {
        & $SAM build
        if ($LASTEXITCODE -ne 0) { throw "sam build failed" }

        Write-Host ""
        Write-Host "==> Deploying to AWS (config-env default)..." -ForegroundColor Cyan
        & $SAM deploy --config-env default --no-confirm-changeset
        if ($LASTEXITCODE -ne 0) { throw "sam deploy (prod) failed" }
    }
    finally { Pop-Location }

    Write-Host ""
    Write-Host "Production deploy complete." -ForegroundColor Green
}

# ── Dispatch ───────────────────────────────────────────────────────────────
switch ($args[0]) {
    "local" { Deploy-Local }
    "api"   { Start-Api-Local }
    "prod"  { Deploy-Prod }
    default {
        Write-Host ""
        Write-Host "Usage: .\deploy.ps1 [local|api|prod]" -ForegroundColor Yellow
        Write-Host "  local  - stage + sam build + sam deploy to floci (LocalStack :4566)"
        Write-Host "  api    - stage + sam local start-api"
        Write-Host "  prod   - stage + sam build + sam deploy to AWS cloud"
    }
}