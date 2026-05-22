import spaces
import os
import gradio as gr
import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderTiny, AutoencoderKL
import random
import uuid
from typing import Tuple, Union, List, Optional, Any, Dict
import numpy as np
import time
import zipfile
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from typing import Iterable

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

base_model_dev = "black-forest-labs/FLUX.1-dev"

pipe_dev = DiffusionPipeline.from_pretrained(
    base_model_dev,
    torch_dtype=torch.bfloat16
)

lora_repo = "strangerzonehf/Flux-Super-Realism-LoRA"
trigger_word = "Super Realism"

pipe_dev.load_lora_weights(lora_repo)
pipe_dev.to("cuda")

dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

taef1 = AutoencoderTiny.from_pretrained(
    "madebyollin/taef1",
    torch_dtype=dtype
).to(device)

good_vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-Krea-dev",
    subfolder="vae",
    torch_dtype=dtype
).to(device)

pipe_krea = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Krea-dev",
    torch_dtype=dtype,
    vae=taef1
).to(device)


@torch.inference_mode()
def flux_pipe_call_that_returns_an_iterable_of_images(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    good_vae: Optional[Any] = None,
):

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device

    lora_scale = (
        joint_attention_kwargs.get("scale", None)
        if joint_attention_kwargs is not None
        else None
    )

    prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    num_channels_latents = self.transformer.config.in_channels // 4

    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

    image_seq_len = latents.shape[1]

    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )

    self._num_timesteps = len(timesteps)

    guidance = (
        torch.full(
            [1],
            guidance_scale,
            device=device,
            dtype=torch.float32
        ).expand(latents.shape[0])
        if self.transformer.config.guidance_embeds
        else None
    )

    for i, t in enumerate(timesteps):

        if self.interrupt:
            continue

        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        latents_for_image = self._unpack_latents(
            latents,
            height,
            width,
            self.vae_scale_factor
        )

        latents_for_image = (
            latents_for_image / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor

        image = self.vae.decode(
            latents_for_image,
            return_dict=False
        )[0]

        yield self.image_processor.postprocess(
            image,
            output_type=output_type
        )[0]

        latents = self.scheduler.step(
            noise_pred,
            t,
            latents,
            return_dict=False
        )[0]

        torch.cuda.empty_cache()

    latents = self._unpack_latents(
        latents,
        height,
        width,
        self.vae_scale_factor
    )

    latents = (
        latents / good_vae.config.scaling_factor
    ) + good_vae.config.shift_factor

    image = good_vae.decode(
        latents,
        return_dict=False
    )[0]

    self.maybe_free_model_hooks()

    torch.cuda.empty_cache()

    yield self.image_processor.postprocess(
        image,
        output_type=output_type
    )[0]


pipe_krea.flux_pipe_call_that_returns_an_iterable_of_images = (
    flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe_krea)
)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):

    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed."
        )

    if timesteps is not None:

        scheduler.set_timesteps(
            timesteps=timesteps,
            device=device,
            **kwargs
        )

        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    elif sigmas is not None:

        scheduler.set_timesteps(
            sigmas=sigmas,
            device=device,
            **kwargs
        )

        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    else:

        scheduler.set_timesteps(
            num_inference_steps,
            device=device,
            **kwargs
        )

        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps

style_list = [
    {
        "name": "3840 x 2160",
        "prompt": "hyper-realistic 8K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "",
    },
    {
        "name": "2560 x 1440",
        "prompt": "hyper-realistic 4K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "",
    },
    {
        "name": "HD+",
        "prompt": "hyper-realistic 2K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "",
    },
    {
        "name": "Style Zero",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
]

styles = {
    k["name"]: (k["prompt"], k["negative_prompt"])
    for k in style_list
}

DEFAULT_STYLE_NAME = "3840 x 2160"
STYLE_NAMES = list(styles.keys())


def apply_style(style_name: str, positive: str) -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n


@spaces.GPU
def generate_dev(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    style_name: str = DEFAULT_STYLE_NAME,
    num_inference_steps: int = 30,
    num_images: int = 1,
    zip_images: bool = False,
    progress=gr.Progress(track_tqdm=True),
):

    positive_prompt, style_negative_prompt = apply_style(
        style_name,
        prompt
    )

    if use_negative_prompt:
        final_negative_prompt = (
            style_negative_prompt + " " + negative_prompt
        )
    else:
        final_negative_prompt = style_negative_prompt

    final_negative_prompt = final_negative_prompt.strip()

    if trigger_word:
        positive_prompt = f"{trigger_word} {positive_prompt}"

    seed = int(randomize_seed_fn(seed, randomize_seed))

    generator = torch.Generator(device="cuda").manual_seed(seed)

    start_time = time.time()

    images = pipe_dev(
        prompt=positive_prompt,
        negative_prompt=(
            final_negative_prompt
            if final_negative_prompt else None
        ),
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        generator=generator,
        output_type="pil",
    ).images

    end_time = time.time()
    duration = end_time - start_time

    image_paths = [save_image(img) for img in images]

    zip_path = None

    if zip_images:
        zip_name = str(uuid.uuid4()) + ".zip"

        with zipfile.ZipFile(zip_name, "w") as zipf:
            for i, img_path in enumerate(image_paths):
                zipf.write(img_path, arcname=f"Img_{i}.png")

        zip_path = zip_name

    return image_paths, seed, f"{duration:.2f}", zip_path


@spaces.GPU
def generate_krea(
    prompt: str,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 4.5,
    randomize_seed: bool = False,
    num_inference_steps: int = 28,
    num_images: int = 1,
    zip_images: bool = False,
    progress=gr.Progress(track_tqdm=True),
):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)

    start_time = time.time()

    images = []

    for _ in range(num_images):

        final_img = list(
            pipe_krea.flux_pipe_call_that_returns_an_iterable_of_images(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=generator,
                output_type="pil",
                good_vae=good_vae,
            )
        )[-1]

        images.append(final_img)

    end_time = time.time()
    duration = end_time - start_time

    image_paths = [save_image(img) for img in images]

    zip_path = None

    if zip_images:
        zip_name = str(uuid.uuid4()) + ".zip"

        with zipfile.ZipFile(zip_name, "w") as zipf:
            for i, img_path in enumerate(image_paths):
                zipf.write(img_path, arcname=f"Img_{i}.png")

        zip_path = zip_name

    return image_paths, seed, f"{duration:.2f}", zip_path


@spaces.GPU
def generate(
    model_choice: str,
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    style_name: str = DEFAULT_STYLE_NAME,
    num_inference_steps: int = 30,
    num_images: int = 1,
    zip_images: bool = False,
    progress=gr.Progress(track_tqdm=True),
):

    if model_choice == "flux.1-dev-realism":

        return generate_dev(
            prompt=prompt,
            negative_prompt=negative_prompt,
            use_negative_prompt=use_negative_prompt,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            randomize_seed=randomize_seed,
            style_name=style_name,
            num_inference_steps=num_inference_steps,
            num_images=num_images,
            zip_images=zip_images,
            progress=progress,
        )

    elif model_choice == "flux.1-krea-dev":

        return generate_krea(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            randomize_seed=randomize_seed,
            num_inference_steps=num_inference_steps,
            num_images=num_images,
            zip_images=zip_images,
            progress=progress,
        )

    else:
        raise ValueError("Invalid model choice")

examples = [
    "Ultra realistic cinematic portrait of a woman standing in neon rain, cyberpunk atmosphere",
    "Professional fashion photography of a handsome man wearing black suit, studio lighting",
    "Dreamy purple aesthetic portrait with glowing lights and glasses",
    "Photorealistic mountain landscape during golden hour with volumetric lighting",
]


css = """
.gradio-container {
    max-width: 1600px !important;
    margin: auto !important;
    padding-top: 10px !important;
}

#main-title {
    text-align: left;
    margin-bottom: 10px;
}

#main-title h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
}

.left-column {
    border-right: 1px solid rgba(255,255,255,0.08);
    padding-right: 18px;
}

.right-column {
    padding-left: 18px;
}

.run-btn {
    height: 52px;
    font-size: 18px !important;
    font-weight: 700 !important;
}

footer {
    visibility: hidden;
}
"""

with gr.Blocks() as demo:

    with gr.Row():

        with gr.Column(scale=7, elem_classes="left-column"):

            gr.Markdown(
                "# Flux Realism Dev",
                elem_id="main-title"
            )

            result = gr.Gallery(
                label="Generated Images",
                columns=2,
                height=450,
                preview=True,
                object_fit="contain"
            )

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt...",
                lines=4
            )

            run_button = gr.Button(
                "Generate Images",
                variant="primary",
                elem_classes="run-btn"
            )

        with gr.Column(scale=3, elem_classes="right-column"):

            model_choice = gr.Dropdown(
                choices=[
                    "flux.1-krea-dev",
                    "flux.1-dev-realism"
                ],
                label="Select Model",
                value="flux.1-krea-dev"
            )

            with gr.Accordion(
                "Additional Options",
                open=False
            ):

                style_selection = gr.Dropdown(
                    label="Quality Style",
                    choices=STYLE_NAMES,
                    value=DEFAULT_STYLE_NAME,
                    interactive=True,
                )

                use_negative_prompt = gr.Checkbox(
                    label="Use Negative Prompt",
                    value=False
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    placeholder="Enter negative prompt",
                    visible=False,
                )

                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )

                randomize_seed = gr.Checkbox(
                    label="Randomize Seed",
                    value=True
                )

                with gr.Row():

                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=1024,
                    )

                    height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=1024,
                    )

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=0.1,
                    maximum=20.0,
                    step=0.1,
                    value=4.5,
                )

                num_inference_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=1,
                    maximum=40,
                    step=1,
                    value=28,
                )

                num_images = gr.Slider(
                    label="Number of Images",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=1,
                )

                zip_images = gr.Checkbox(
                    label="Zip Generated Images",
                    value=False
                )

                gr.Markdown("### Output Information")

                seed_display = gr.Textbox(
                    label="Seed Used",
                    interactive=False
                )

                generation_time = gr.Textbox(
                    label="Generation Time (s)",
                    interactive=False
                )

                zip_file = gr.File(
                    label="Download ZIP"
                )

            gr.Markdown("## Examples")

            gr.Examples(
                examples=examples,
                inputs=prompt,
            )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            model_choice,
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
            style_selection,
            num_inference_steps,
            num_images,
            zip_images,
        ],
        outputs=[
            result,
            seed_display,
            generation_time,
            zip_file,
        ],
        api_name="run",
    )

if __name__ == "__main__":

    demo.queue(max_size=30).launch(
        css=css,
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
    )
