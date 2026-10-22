"""Does bf16 / torch.compile make training faster? Run from the repo root, e.g.

    python bench_precision_compile.py --cfenet --batch 64 --steps 30 \
        --config configs/whu/sanity/unet_swint_whu_cfenet_s42_sanity.yaml --train_h5 /path/whu_train.h5

Times full training steps (forward, loss, backward, optimizer) on synthetic input for each variant (fp16/bf16,
eager or with the encoder / CFENet / decoder compiled, with and without the check_hook hooks), and,
if --config and --train_h5 are given, how many samples/s your real training DataLoader delivers.
Whichever is lower is your ceiling: if the loader is slower than the GPU, neither bf16 nor compile
will shorten a real epoch."""
import argparse, time, warnings
import torch
warnings.filterwarnings("ignore")

from models.segmentation_model import build_model
from training.loss_function import BCEDiceLoss

# name: (autocast dtype, parts to compile in place, remove the check_hook forward hooks first?)
# check_hook calls .item() (a GPU sync) and cannot be traced by torch.compile, so every variant that
# compiles cfenet or decoder removes it; "no hooks" variants separate the hooks' own cost from compile.
VARIANTS = {
    "fp16 eager":                    (torch.float16,  (), False),
    "bf16 eager":                    (torch.bfloat16, (), False),
    "bf16 eager, no hooks":          (torch.bfloat16, (), True),
    "bf16 compile encoder":          (torch.bfloat16, ("encoder",), False),
    "bf16 compile cfenet":           (torch.bfloat16, ("cfenet",), True),
    "bf16 compile decoder":          (torch.bfloat16, ("decoder",), True),
    "bf16 compile enc+cfe+dec":      (torch.bfloat16, ("encoder", "cfenet", "decoder"), True),
}

def sync(dev):
    if dev == "cuda": torch.cuda.synchronize()

def bench(name, dtype, parts, no_hooks, args):
    dev = args.device
    torch.manual_seed(0)
    cfg = {"model": {"encoder": "swin_t", "timm_name": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
                     "cfenet": args.cfenet, "pretrained": False}}
    model = build_model(cfg).to(dev).train()
    groups = [{"params": model.encoder.parameters(), "lr": 6e-5}, {"params": model.decoder.parameters(), "lr": 6e-4}]
    if args.cfenet: groups.append({"params": model.cfenet.parameters(), "lr": 6e-4})
    if no_hooks:
        for m in (model.cfenet, model.decoder):
            if m is not None: m._forward_hooks.clear()
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    loss_fn = BCEDiceLoss()
    scaler = torch.amp.GradScaler(dev) if dtype == torch.float16 else None

    x = torch.randn(args.batch, 3, 224, 224, device=dev)
    y = (torch.rand(args.batch, 1, 224, 224, device=dev) > 0.8).float()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev, dtype=dtype):
            loss = loss_fn(model(x), y)["total_loss"]
        if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()

    t0 = time.time()
    step(); sync(dev)                    # first step builds the lazy CFENet layers (eager) ...
    for part in parts:                   # in place: state_dict keys are unchanged
        getattr(model, part).compile(backend=args.backend)
    step(); sync(dev)                    # ... and this one pays the compile cost
    first = time.time() - t0
    for _ in range(args.warmup): step()
    sync(dev); t = time.time()
    for _ in range(args.steps): step()
    sync(dev); per = (time.time() - t) / args.steps
    mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
    print(f"{name:24s} {per*1000:8.1f} ms/step {args.batch/per:8.0f} samples/s   first 2 steps (incl. compile): {first:6.1f}s   peak mem {mem:5.1f} GiB", flush=True)
    del model, opt; 
    if dev == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch._dynamo.reset()

def loader_rate(args):
    from trainer import load_config
    from data.datamodule import BuildingDataModule
    cfg = load_config(args.config)
    dm = BuildingDataModule(cfg, args.train_h5, args.train_h5); dm.setup()
    loader = dm.train_dataloader()
    def batches():                                    # restart when an epoch ends (short epochs)
        while True:
            yield from loader
    it = batches()
    for _ in range(3): next(it)                       # worker start-up
    t = time.time(); n = 0
    for _ in range(args.loader_batches): n += next(it)[0].shape[0]
    rate = n / (time.time() - t)
    print(f"{'DataLoader (real data)':24s} {'':8s}    {'':3s}{rate:8.0f} samples/s   (num_workers={dm.num_workers}, batch={cfg['data']['train_batch_size']})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfenet", action="store_true"); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=30); p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda"); p.add_argument("--backend", default="inductor"); p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    p.add_argument("--config"); p.add_argument("--train_h5"); p.add_argument("--loader_batches", type=int, default=30)
    args = p.parse_args()
    print(f"torch {torch.__version__} | device {args.device} | batch {args.batch} | cfenet {args.cfenet}")
    for name in args.variants:
        if "cfenet" in name and not args.cfenet and "enc+cfe" not in name:
            continue                      # no CFENet in this model
        bench(name, *VARIANTS[name], args)
    if args.config and args.train_h5: loader_rate(args)
