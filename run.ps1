$ErrorActionPreference = 'Stop'
$localConfig = Join-Path $PSScriptRoot 'deployment.psd1'
if (Test-Path -LiteralPath $localConfig) {
    $settings = Import-PowerShellDataFile -LiteralPath $localConfig
    foreach ($name in @('HF_HOME','HF_HUB_OFFLINE','IMMICH_ML_URL','QWEN_ST_DEVICE','QWEN_ST_IDLE_SECONDS','QWEN_ST_MODEL_NAME')) {
        if ($settings.ContainsKey($name)) {
            [Environment]::SetEnvironmentVariable($name, [string]$settings[$name], 'Process')
        }
    }
}
$logDirectory = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$options = @{
    FilePath = (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
    ArgumentList = @('-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', '17933', '--no-access-log')
    WorkingDirectory = $PSScriptRoot
    WindowStyle = 'Hidden'
    RedirectStandardOutput = (Join-Path $logDirectory 'stdout.log')
    RedirectStandardError = (Join-Path $logDirectory 'stderr.log')
    PassThru = $true
    Wait = $true
}
$server = Start-Process @options
exit $server.ExitCode
