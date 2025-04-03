$pf="kel-I-20240826"
$r="kel-I"
$od="kel_I"

New-Item -ItemType Directory -Force -Path "..\..\Cambridge University Dropbox\Ivan Grega\neural_xray\data\experimental\processed\$od"


for ($i=0; $i -lt 1; $i++){
    $s="{0:d2}" -f $i
    "Processing $s..."
    tar -xzf "..\..\Cambridge University Dropbox\Ivan Grega\neural_xray\data\experimental\raw\$pf\$r-$i.tar.gz" `
        -C C:\temp\ 
    # tar -xzf C:\temp\$r-$i.tar.gz -C C:\temp\
    if ($i -lt 10) {
        mv "C:\temp\images_$i" "C:\temp\images_$s"
    }
    
    $lambda=(Get-Content "..\..\Cambridge University Dropbox\Ivan Grega\neural_xray\data\experimental\raw\$pf\normalization\lambdas2.csv" -TotalCount $($i+1) -ReadCount 100)[-1]
    python nerf_data/scripts/tiff_to_png.py --input-folder "C:\temp\images_$s" --out-fn-pattern "train_{0:04d}.png" --greyscale-fn "lambda x: $lambda" --thresh-min 36000 --thresh-max 58000
    rm "C:\temp\images_$s\*.tif"
    if ($i % 5 -eq 0 -and $i -ne 5) {
        $eval='0129'
        $xt=''
    } else {
        $eval='0013'
        $xt=''
    }
    cp "..\..\Cambridge University Dropbox\Ivan Grega\neural_xray\data\experimental\raw\$pf\$r-$i.ang" C:\temp
    cp "..\..\Cambridge University Dropbox\Ivan Grega\neural_xray\data\experimental\raw\$pf\$r-$i$xt.xtekct" C:\temp
    mv "C:\temp\images_$s\train_$eval.png" "C:\temp\images_$s\eval_$eval.png"
    python .\nerf_data\scripts\compute_transforms.py --folder "C:\temp" --images-folder "images_$s" --xtekct-file "$r-$i$xt.xtekct" --angles-file "$r-$i.ang" --output-fname "C:\temp\transforms_$s.json"
    Move-Item "C:\temp\images_$s"  -Destination "..\..\Cambridge University Dropbox\Ivan Grega\neural_xray\data\experimental\processed\$od"
    Move-Item "C:\temp\transforms_$s.json" -Destination "..\..\Cambridge University Dropbox\Ivan Grega\neural_xray\data\experimental\processed\$od"
    rm "C:\temp\*"
    # # $pth="..\..\Cambridge University Dropbox\Ivan Grega\neural_xray\data\experimental\raw\$pf"
    # # Get-ChildItem -Path $pth -Filter *.raw -Recurse -Name | ForEach {
    # #     $fn = $_
    # #     "$fn"
    # #     python .\nerf_data\scripts\raw_to_npy.py `
    # #     --input $(Join-Path $pth $fn) --dtype UINT16 `
    # #     --resolution 500 500 500 --out-dtype UINT8 #--out-resolution 500 500 500
    # # }
    # # Get-ChildItem -Path $pth -Filter *.npz -Recurse | ForEach {
    # #     $fn = $_
    # #     Move-Item $fn "..\..\Cambridge University Dropbox\Ivan Grega\neural_xray\data\experimental\processed\$od\$fn.npz"
    # # }
}