
# Re-run all 5 datasets with the depthwise-conv patch embedding (~150K ViT-module
# params, much closer to the paper's stated ~154,628 than the default GAP+Linear's
# ~99K), to see whether closer architectural fidelity to the paper's stated param
# budget changes accuracy. Outputs land in runs/<dataset>_depthwise/, existing
# gap_linear results in runs/<dataset>/ are untouched.
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\src"

python run_all.py --patch_embed depthwise

Write-Host "=== DONE: all 5 datasets finished with depthwise patch embedding ==="
