import re
import yaml
import json
from pathlib import Path

from utils import convert_numerics


class ConfigNameError(ValueError):
    """Raised when a config's name, location or fields break the naming rules."""


# {decoder}_{encoder}_{dataset}_{neck}_s{seed}[_{variant}]
NAME_RE = re.compile(
    r"^(?P<decoder>unet)"
    r"_(?P<encoder>swint|swins|swinb)"
    r"_(?P<dataset>whu)"
    r"_(?P<neck>base|cfenet)"
    r"_s(?P<seed>0|[1-9][0-9]*)"
    r"(?:_(?P<variant>[A-Za-z0-9][A-Za-z0-9._-]*))?$"
)
TEMPLATE = "{decoder}_{encoder}_{dataset}_{neck}_s{seed}[_{variant}]"

# name token -> (model.encoder value, required prefix of model.timm_name)
ENCODERS = {
    "swint": ("swin_t", "swin_tiny"),
    "swins": ("swin_s", "swin_small"),
    "swinb": ("swin_b", "swin_base"),
}

# Keys that do not change what an experiment does; ignored when looking for duplicate settings.
IGNORED_FOR_DUPLICATES = {"experiment_name", "tags", "num_workers", "platform", "logging"}


def _yaml_files(root):
    root = Path(root)
    return sorted(p for pattern in ("*.yaml", "*.yml") for p in root.rglob(pattern))


def validate_config_name(config, config_path, configs_root="configs"):
    """Check the experiment name against the template, the file name, the config content,
    and that no other file under configs_root has the same file name.
    Raises ConfigNameError listing all problems. No other config is opened."""
    path = Path(config_path).resolve()
    root = Path(configs_root).resolve()

    name = config.get("experiment_name")
    if not isinstance(name, str):
        raise ConfigNameError("experiment_name is missing or is not a string.")

    match = NAME_RE.match(name)
    if match is None:
        raise ConfigNameError(
            f"experiment_name '{name}' does not match the template {TEMPLATE}\n"
            f"  decoder: unet | encoder: swint/swins/swinb | dataset: whu | "
            f"neck: base/cfenet | seed: s<int, no leading zeros> | variant: optional free text"
        )
    parts = match.groupdict()

    problems = []

    # --- file name and location ---
    if path.stem != name:
        problems.append(f"file name '{path.name}' must be '{name}.yaml'.")
    try:
        path.relative_to(root)
    except ValueError:
        problems.append(f"config must live under {root}.")

    # --- name fields against the real config content ---
    model = config.get("model") or {}
    data = config.get("data") or {}

    if model.get("decoder") != parts["decoder"]:
        problems.append(f"name says decoder '{parts['decoder']}' but model.decoder is {model.get('decoder')!r}.")

    enc_key, timm_prefix = ENCODERS[parts["encoder"]]
    if model.get("encoder") != enc_key:
        problems.append(f"name says encoder '{parts['encoder']}' ({enc_key}) but model.encoder is {model.get('encoder')!r}.")
    if not str(model.get("timm_name", "")).startswith(timm_prefix):
        problems.append(f"name says encoder '{parts['encoder']}' but model.timm_name is {model.get('timm_name')!r} "
                        f"(expected to start with '{timm_prefix}').")

    if data.get("dataset") != parts["dataset"]:
        problems.append(f"name says dataset '{parts['dataset']}' but data.dataset is {data.get('dataset')!r}.")

    cfenet = model.get("cfenet")
    if not isinstance(cfenet, bool):
        problems.append(f"model.cfenet must be true or false, found {cfenet!r}.")
    elif cfenet != (parts["neck"] == "cfenet"):
        problems.append(f"name says neck '{parts['neck']}' but model.cfenet is {cfenet}.")

    seed = config.get("seed")
    seed_ok = (
        isinstance(seed, (int, float))
        and not isinstance(seed, bool)
        and float(seed).is_integer()
        and int(seed) == int(parts["seed"])
    )
    if not seed_ok:
        problems.append(f"name says seed {parts['seed']} but seed is {config.get('seed')!r}.")

    # --- file names must be unique across all folders (compares names only, opens nothing) ---
    # Compared case-insensitively because Windows and macOS treat 'A.yaml' and 'a.yaml' as one file.
    for other in _yaml_files(root):
        if other.resolve() != path and other.stem.lower() == name.lower():
            problems.append(f"another config has the same file name: {other}.")

    if problems:
        raise ConfigNameError(f"Config '{path.name}' breaks the naming rules:\n- " + "\n- ".join(problems))
    return parts


def _settings_key(cfg):
    kept = {k: v for k, v in convert_numerics(cfg).items() if k not in IGNORED_FOR_DUPLICATES}
    return json.dumps(kept, sort_keys=True, default=str)


def find_duplicate_settings(config, config_path, configs_root="configs"):
    """Other configs whose settings are identical to this one apart from name, tags,
    num_workers, platform and logging. Meant as a warning, not an error."""
    me = Path(config_path).resolve()
    key = _settings_key(config)
    duplicates = []
    for other in _yaml_files(configs_root):
        if other.resolve() == me:
            continue
        try:
            with open(other, "r") as f:
                other_cfg = yaml.safe_load(f)
        except yaml.YAMLError:
            continue
        if isinstance(other_cfg, dict) and _settings_key(other_cfg) == key:
            duplicates.append(other)
    return duplicates
