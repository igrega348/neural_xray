$NALL = ( Get-ChildItem .\outputs\cube\nerf_xray\  | Measure-Object ).Count
$n=0
Get-ChildItem .\outputs\cube\nerf_xray\ | ForEach-Object -Process {
    Write-Progress -Activity "Modulo" -Status "Running $n / $NALL" -PercentComplete ([math]::ceiling($n*100/$NALL))
    # check if outputs exists. If not, run eval. If yes, skip.
    if (Test-Path $(Join-Path $_.FullName 'output.json')) {
        "Skipping $($_.FullName)"
        return
    }
    python ./nerfstudio/nerfstudio/scripts/eval.py --load_config $(Join-Path $_.FullName 'config.yml') --output_path $(Join-Path $_.FullName 'output.json') #--render_output_path $(Join-Path $_.FullName 'render')
    $n=$n+1
}