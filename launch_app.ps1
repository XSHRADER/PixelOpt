# Set up and start PixelOpt. All the work is in launch.py; this only finds a
# Python to run it with. Extra arguments are passed on, e.g.  .\launch_app.ps1 --check
Set-Location -Path $PSScriptRoot

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 launch.py @args
} else {
    & python launch.py @args
}
exit $LASTEXITCODE
