param(
    [Parameter(Mandatory = $true)]
    [string]$CKRoot,
    [ValidateSet('all', 'qkv', 'mlp_up', 'mlp_down', 'attn_out')]
    [string]$Shape = 'all',
    [int]$Rows = 3802,
    [int]$Warmup = 3,
    [int]$Iterations = 11,
    [switch]$Check
)

$python = 'C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe'
$env:PYTHONPATH = $CKRoot
$arguments = @(
    "$PSScriptRoot\convrot_int8_bench.py",
    '--ck-root', $CKRoot,
    '--shape', $Shape,
    '--rows', $Rows,
    '--warmup', $Warmup,
    '--iterations', $Iterations
)
if ($Check) {
    $arguments += '--check'
}

& $python @arguments
exit $LASTEXITCODE
