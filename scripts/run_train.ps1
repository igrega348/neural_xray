for ($m=2; $m -le 256; $m=$m*2) {
    $OuterLoopProgressParameters = @{
        ID = 0
        Activity = "Modulo"
        Status = "Running $m"
        PercentComplete = $m/256*100
    }
    Write-Progress @OuterLoopProgressParameters
    $i0=0
    while ($i0 -lt $m) {
        $InnerLoopProgressParameters = @{
            ID = 1
            ParentID = 0
            Activity = "Modulo $m"
            Status = "Running $i0"
            PercentComplete = $i0/$m*100
        }
        Write-Progress @InnerLoopProgressParameters
        python ./nerfstudio/scripts/train.py method-template --data "C:/temp/nerf_data/cube" --vis "tensorboard" --max_num_iterations 2000 xray-dataparser --modulo $m --i0 $i0
        $i0=$i0+[math]::ceiling($m/8)
    }
}
for ($m=87; $m -le 87; $m=$m*2) { # 3 projections
    $OuterLoopProgressParameters = @{
        ID = 0
        Activity = "Modulo"
        Status = "Running $m"
        PercentComplete = $m/256*100
    }
    Write-Progress @OuterLoopProgressParameters
    $i0=0
    while ($i0 -lt $m) {
        $InnerLoopProgressParameters = @{
            ID = 1
            ParentID = 0
            Activity = "Modulo $m"
            Status = "Running $i0"
            PercentComplete = $i0/$m*100
        }
        Write-Progress @InnerLoopProgressParameters
        python ./nerfstudio/scripts/train.py method-template --data "C:/temp/nerf_data/cube" --vis "tensorboard" --max_num_iterations 2000 xray-dataparser --modulo $m --i0 $i0
        $i0=$i0+[math]::ceiling($m/8)
    }
}
# 2 orthogonal projections
$m=64
$i0=0
while ($i0 -lt $m) {
    $imax=$m+$i0+1
    Write-Progress -Activity "Modulo $m" -Status "Running i0=$i0, imax=$imax" -PercentComplete $i0/$m*100
    python ./nerfstudio/scripts/train.py method-template --data "C:/temp/nerf_data/cube" --vis "tensorboard" xray-dataparser --modulo $m --i0 $i0 --imax $imax
    $i0=$i0+[math]::ceiling($m/8)
}