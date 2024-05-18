# python nerfstudio/scripts/train.py nerf_bspline --data "../nerf_data/synthetic/bspline/transforms.json" `
#     --max_num_iterations 200 xray-temp-dataparser

# python .\nerfstudio\scripts\exporter.py image-stack --plane xz --plot-engine opencv `
#     --load-config .\outputs\bspline\nerf_bspline\2024-05-11_121643\config.yml `
#     --output-dir .\outputs\bspline\nerf_bspline\2024-05-11_121643\slices 

# python .\nerfstudio\scripts\train.py nerf_bspline --data "..\nerf_data\synthetic\bspline\transforms_d.json" `
#     --max_num_iterations 200 --pipeline.model.train_density_field False --pipeline.model.train_deformation_field True `
#     --pipeline.load_density_ckpt ".\outputs\bspline\nerf_bspline\2024-05-11_122934\nerfstudio_models\step-000000199.ckpt" `
#     xray-temp-dataparser

python .\nerfstudio\scripts\exporter.py image-stack --plane xz --plot-engine opencv `
    --load-config outputs\bspline\nerf_bspline\2024-05-11_141048\config.yml `
    --output-dir .\outputs\bspline\nerf_bspline\2024-05-11_141048\slices 