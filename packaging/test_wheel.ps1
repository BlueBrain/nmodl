param (
   [Parameter(Mandatory=$true)][string]$wheel,
   [bool]$venv=$true
)

$TEST_DIR = "$($Env:temp)/tmp$([convert]::tostring((get-random 65535),16).padleft(4,'0')).tmp"
New-Item -ItemType Directory -Path $TEST_DIR
New-Item -ItemType Directory -Path $TEST_DIR/input
New-Item -ItemType Directory -Path $TEST_DIR/output

$NMODL_ROOT=(Split-Path -Parent $PSScriptRoot)

Write-Output $NMODL_ROOT
Get-ChildItem -Path (Join-Path $NMODL_ROOT "python/nmodl/ext/example") -Filter "*.mod" | ForEach-Object {
   Copy-Item $_  $TEST_DIR/input
}
Copy-Item "$NMODL_ROOT/test/integration/mod/cabpump.mod" $TEST_DIR/input
Copy-Item "$NMODL_ROOT/test/integration/mod/var_init.inc" $TEST_DIR/input
Copy-Item "$NMODL_ROOT/test/integration/mod/glia_sparse.mod" $TEST_DIR/input

if ($venv) {
   python -m venv wheel_test_venv
   ./wheel_test_venv/Scripts/activate.ps1

   pip uninstall -y nmodl nmodl-nightly
   pip install "${wheel}[test]"
   pip show nmodl-nightly
}

Get-ChildItem -Path $TEST_DIR/input -Filter "*.mod" | ForEach-Object {
   $path = $_ -replace "\\","/"
   Write-Output "nmodl -o $TEST_DIR/output $path sympy --analytic"
   nmodl -o $TEST_DIR/output $path sympy --analytic
   if (! $?) {
      Write-Output "Failed NMODL run"
      Exit 1
   }
   python -c "import nmodl; driver = nmodl.NmodlDriver(); driver.parse_file('$path')"
   if (! $?) {
      Write-Output "Failed NMODL Python module parsing"
      Exit 1
   }
}

# rm -r $TEST_DIR