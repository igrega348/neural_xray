$pf="kel-Q-20250604"
$r="kel-Q"
$od="kel_Q"
$raw_dir="E:\Deshpande Dropbox\Ivan Grega\neural_xray\data\experimental\raw\$pf"
$processed_dir="E:\Deshpande Dropbox\Ivan Grega\neural_xray\data\experimental\processed\$od"
$temp_dir="C:\temp\neural_xray_data"
$tiff_to_png="C:\temp\neural_xray\nerf_data\scripts\tiff_to_png.py"
$comp_transf="C:\temp\neural_xray\nerf_data\scripts\compute_transforms.py"
$compute_transforms=$true

New-Item -ItemType Directory -Force -Path "$processed_dir" -Confirm
New-Item -ItemType Directory -Force -Path "$temp_dir" -Confirm


for ($i=0; $i -lt 1; $i++){
    # $s="{0:d2}" -f $i
    $s="def"
    "Processing $s..."
    $images_folder="images_$s"
    # $rd=$(Get-ChildItem -Path $raw_dir -Filter "kel-cont-$exp-$nproj-$i*" -Name)
    
    # mkdir "$temp_dir\$images_folder"
    # tar -xzf "$raw_dir\$rd\kel-cont-$exp-$nproj-$i.tar.gz" `
    #     -C "$temp_dir\$images_folder" 
    
    # $tmin=27000
    # $tmax=59000

    # # $lambda=(Get-Content "$raw_dir\normalization\lambdas.csv" -TotalCount $($i+1) -ReadCount 100)[-1]
    # python $tiff_to_png --input-folder "$temp_dir\$images_folder" --out-fn-pattern "train_{0:04d}.png" --thresh-min $tmin --thresh-max $tmax #--dtype UINT16
    # # # python $tiff_to_png --input-folder "$temp_dir\images_$s" --out-fn-pattern "train_{0:04d}.png" #--greyscale-fn "lambda x: $lambda" --thresh-min 19000 --thresh-max 58500 #--dtype UINT16
    
    # rm "$temp_dir\$images_folder\*.tif"

    # $eval=((ls "$temp_dir\$images_folder" | Sort-Object)[-1]).Basename.Split("_")[-1]
    # mv "$temp_dir\$images_folder\train_$eval.png" "$temp_dir\$images_folder\eval_$eval.png"

    # for ($i=1; $i -lt 2867; $i++){
    #     if ($i%28 -ne 1){
    #         $s="{0:d4}" -f $i
    #         rm "$temp_dir\$images_folder\train_$s.png"
    #     }
    # }

    # cp -LiteralPath "$raw_dir\$rd\_ctdata.txt" -Destination $temp_dir
    # cp -LiteralPath "$raw_dir\$rd\kel-cont-$exp-$nproj-$i.xtekct" -Destination $temp_dir
    # cp -LiteralPath "$raw_dir\$rd\kel-cont-$exp-$nproj-$i.ctinfo.xml" -Destination $temp_dir
    
    $ff=0.89
    if ($compute_transforms){
        python "$comp_transf" --folder "$temp_dir" --images-folder "$images_folder" --output-fname "$temp_dir\transforms_$s.json" --deblurring Gauss --deblurring-points 1 --flat-field $ff --angles-file "$temp_dir\_ctdata-$s.txt" #--xtekct-file "$temp_dir\$r-cont-2.xtekct" 
        # Move-Item "$temp_dir\transforms_0_to_1.json" -Destination "$processed_dir"
    }
    # rm "$temp_dir\kel-cont-$exp-$nproj-$i.xtekct"
    # rm "$temp_dir\kel-cont-$exp-$nproj-$i.ctinfo.xml"
    # rm "$temp_dir\_ctdata.txt"

    # if ($compute_transforms){
    #     cp "$raw_dir\$r-$i.ang" $temp_dir
    #     cp "$raw_dir\$r-$i$xt.xtekct" $temp_dir
    #     python "$comp_transf" --folder "$temp_dir" --images-folder "images_$s" --xtekct-file "$r-$i$xt.xtekct" --angles-file "$r-$i.ang" --output-fname "$temp_dir\transforms_$s.json"
    #     Move-Item "$temp_dir\transforms_$s.json" -Destination "$processed_dir"
    #     rm "$temp_dir\$r-$i$xt.xtekct"
    #     rm "$temp_dir\$r-$i.ang"
    # }
    # Move-Item "$temp_dir\images_$s" -Destination "$processed_dir"

    # rm "C:\temp\*"
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