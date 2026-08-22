param(
    [Parameter(Mandatory = $true)]
    [string]$CKRoot,
    [string]$Python = 'python',
    [ValidateSet('all', 'qkv', 'mlp_up', 'mlp_down', 'attn_out')]
    [string]$Shape = 'all',
    [int]$Rows = 3802,
    [int]$Warmup = 3,
    [int]$Iterations = 11,
    [switch]$Check
)

$env:PYTHONPATH = if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
    $CKRoot
} else {
    "$CKRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
}
$arguments = @(
    (Join-Path $PSScriptRoot 'convrot_int8_bench.py'),
    '--ck-root', $CKRoot,
    '--shape', $Shape,
    '--rows', $Rows,
    '--warmup', $Warmup,
    '--iterations', $Iterations
)
if ($Check) {
    $arguments += '--check'
}

& $Python @arguments
exit $LASTEXITCODE
