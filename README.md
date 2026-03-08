# ReFrame
ReFrame: A Resource-Friendly Cloud-Assisted On-Device Deep Learning Framework for Vision Services [IEEE TSC 2025]

## Environment

The prototype system of `reframe`, this demo requires the PyTorch and ONNX.

```bash
pip install onnx==1.14.1
pip install onnxruntime==1.15.1
```

## Usage


Training, Transformation, and Surgery (cloud-side)
```bash
# ResNet-18
python nlresnet_train_transf_surg_cloud.py --model r18 --seed 0 --initw True --epochs 200 > ./r18_baseline.log
# Non-learnable ResNet-18 C_{NLconv} = dw
python nlresnet_train_transf_surg_cloud.py --model nlr18 --seed 0 --nl-seed 1 --mono-map eqdiff --init-option kaiming_normal --nlcvoutch -1 --shuffle True --initw True --epochs 200 > ./nlr18_groupconv_max_mixavg_dw_g4_chsh_bs128_2e_s0_nls1_eqdiff.log
```

Deployment (device-side)
```bash
# Non-learnable ResNet-18 C_{NLconv} = dw
python nlresnet_deployment_device_raspi.py --model nlr18 --seed 0 --nl-seed 1 --mono-map eqdiff --init-option kaiming_normal --nlcvoutch -1
# Non-learnable ResNet-18 C_{NLconv} = 512
python nlresnet_deployment_device_raspi.py --model nlr18 --seed 0 --nl-seed 1 --mono-map eqdiff --init-option kaiming_normal --nlcvoutch 512
# Non-learnable ResNet-18 C_{NLconv} = 768
python nlresnet_deployment_device_raspi.py --model nlr18 --seed 0 --nl-seed 1 --mono-map eqdiff --init-option kaiming_normal --nlcvoutch 768
```

Static Inference (device-side)
```bash
# ResNet-18
python nlresnet_onnxruntime_static_inference_mem.py --model r18 > ./logs/r18_static_infer_mem.log
# Non-learnable ResNet-18 C_{NLconv} = dw
python nlresnet_onnxruntime_static_inference_mem.py --model nlr18 --nlcvoutch -1 > ./logs/nlr18_groupconv_max_mixavg_dw_g4_chsh_static_infer_mem.log
# Non-learnable ResNet-18 C_{NLconv} = 512
python nlresnet_onnxruntime_static_inference_mem.py --model nlr18 --nlcvoutch 512 > ./logs/nlr18_groupconv_max_mixavg_512_g4_chsh_static_infer_mem.log
# Non-learnable ResNet-18 C_{NLconv} = 768
python nlresnet_onnxruntime_static_inference_mem.py --model nlr18 --nlcvoutch 768 > ./logs/nlr18_groupconv_max_mixavg_768_g4_chsh_static_infer_mem.log
```

Dynamic Inference (device-side)
```bash
# Non-learnable ResNet-18 C_{NLconv} = dw
python nlresnet_onnxruntime_dynamic_inference_mem.py --model nlr18 --seed 0 --nl-seed 1 --mono-map eqdiff --init-option kaiming_normal --nlcvoutch -1 > ./logs/nlr18_groupconv_max_mixavg_dw_g4_chsh_dyn_infer_mem.log
# Non-learnable ResNet-18 C_{NLconv} = 512
python nlresnet_onnxruntime_dynamic_inference_mem.py --model nlr18 --seed 0 --nl-seed 1 --mono-map eqdiff --init-option kaiming_normal --nlcvoutch 512 > ./logs/nlr18_groupconv_max_mixavg_512_g4_chsh_dyn_infer_mem.log
# Non-learnable ResNet-18 C_{NLconv} = 768
python nlresnet_onnxruntime_dynamic_inference_mem.py --model nlr18 --seed 0 --nl-seed 1 --mono-map eqdiff --init-option kaiming_normal --nlcvoutch 768 > ./logs/nlr18_groupconv_max_mixavg_768_g4_chsh_dyn_infer_mem.log
```

## Publication

Jianhang Xie, Chuntao Ding, Qingji Guan, Ao Zhou, Yidong Li, “ReFrame: A Resource-Friendly Cloud-Assisted On-Device Deep Learning Framework for Vision Services,” ***IEEE Transactions on Services Computing***, vol.18, no.3, pp.1711-1723, 2025.

URL: [DOI](https://doi.org/10.1109/TSC.2025.3552328), [Accepted Manuscript](https://drive.google.com/file/d/1OMlm8BM0OmuZ9vAYw8Gq1zW_nL07bPnq/view?pli=1)

```bibtex
@article{xie.tsc2025reframe,
  author = {Xie, Jianhang and Ding, Chuntao and Guan, Qingji and Zhou, Ao and Li, Yidong},
  journal = {IEEE Transactions on Services Computing},
  title = {ReFrame: A Resource-Friendly Cloud-Assisted On-Device Deep Learning Framework for Vision Services},
  year = {2025},
  volume = {18},
  number = {3},
  pages = {1711-1723},
  doi = {10.1109/TSC.2025.3552328},
}
```

## Acknowledgement

The ONNX surgery modified from https://github.com/bindog/onnx-surgery


