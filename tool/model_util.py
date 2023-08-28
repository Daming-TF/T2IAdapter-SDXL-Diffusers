import diffusers
from pathlib import Path


def from_single_file(pretrained_model_link_or_path, **kwargs):
    # import here to avoid circular dependency
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

    config_file = kwargs.pop('config_file', None)
    extract_ema = kwargs.pop("extract_ema", False)
    image_size = kwargs.pop("image_size", None)
    scheduler_type = kwargs.pop("scheduler_type", "pndm")
    num_in_channels = kwargs.pop("num_in_channels", None)
    upcast_attention = kwargs.pop("upcast_attention", None)
    load_safety_checker = kwargs.pop("load_safety_checker", True)
    prediction_type = kwargs.pop("prediction_type", None)
    text_encoder = kwargs.pop("text_encoder", None)
    vae = kwargs.pop("vae", None)
    controlnet = kwargs.pop("controlnet", None)
    tokenizer = kwargs.pop("tokenizer", None)

    torch_dtype = kwargs.pop("torch_dtype", None)

    file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
    from_safetensors = file_extension == "safetensors"

    # TODO: For now we only support stable diffusion
    stable_unclip = None
    model_type = None

    # remove huggingface url
    for prefix in ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]:
        if pretrained_model_link_or_path.startswith(prefix):
            pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix):]

    pipe = download_from_original_stable_diffusion_ckpt(
        pretrained_model_link_or_path,
        model_type=model_type,
        stable_unclip=stable_unclip,
        controlnet=controlnet,
        from_safetensors=from_safetensors,
        extract_ema=extract_ema,
        image_size=image_size,
        scheduler_type=scheduler_type,
        num_in_channels=num_in_channels,
        upcast_attention=upcast_attention,
        load_safety_checker=load_safety_checker,
        prediction_type=prediction_type,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        original_config_file=config_file if config_file is not None else None,
    )

    if torch_dtype is not None:
        pipe.to(torch_dtype=torch_dtype)

    return pipe
