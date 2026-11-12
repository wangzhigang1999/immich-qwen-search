"""Encode Crossmodal-3600 fixtures and compare with the matching GGUF vectors."""

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np


LANGUAGES = ("zh", "en")


def load_fixtures(path):
    with ZipFile(path) as source:
        return json.loads(source.read("manifest.json"))


def verify_checkpoint(encoder):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from encoder import MODEL, REVISION

    checkpoint = hf_hub_download(MODEL, "model.safetensors", revision=REVISION)
    model = encoder.model[0].auto_model
    parameters = dict(model.named_parameters())
    keys = (
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.27.self_attn.q_proj.weight",
        "model.visual.blocks.0.attn.qkv.weight",
    )
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        for name in keys:
            loaded = parameters.get(name)
            if loaded is None:
                loaded = parameters[name.removeprefix("model.")]
            expected = source.get_slice(name)[:8, :8]
            if not torch.equal(loaded[:8, :8].detach().cpu(), expected.to(loaded.dtype)):
                raise RuntimeError(f"Loaded weights differ from the official checkpoint: {name}")
    print("Verified checkpoint tensors in the text and vision towers", flush=True)


def encode(fixtures, destination, limit):
    import torch
    from encoder import Encoder, preview

    manifest = load_fixtures(fixtures)
    records = manifest["records"][:limit]
    start = perf_counter()
    encoder = Encoder()
    cuda = encoder.device.startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
    loaded = perf_counter()
    verify_checkpoint(encoder)
    images = []
    image_times = []
    texts = {lang: [] for lang in LANGUAGES}
    labels = {lang: [] for lang in LANGUAGES}
    with ZipFile(fixtures) as source:
        for index, record in enumerate(records):
            image = preview(source.read(f"{record['id']}.jpg"))
            before = perf_counter()
            images.append(encoder.encode([image], image=True)[0])
            image_times.append(perf_counter() - before)
            for lang in LANGUAGES:
                captions = record["captions"][lang]
                texts[lang].extend(captions)
                labels[lang].extend([index] * len(captions))
            if (index + 1) % 32 == 0 or index + 1 == len(records):
                print(f"Sentence Transformers images {index + 1}/{len(records)}", flush=True)

    arrays = {"images": np.stack(images)}
    timings = {"image_p50_ms": float(1000 * np.median(image_times[1:] or image_times))}
    for lang in LANGUAGES:
        before = perf_counter()
        captions = texts[lang]
        arrays[lang] = np.concatenate(
            [encoder.encode(captions[i : i + 16]) for i in range(0, len(captions), 16)]
        )
        arrays[f"labels_{lang}"] = np.asarray(labels[lang])
        timings[f"{lang}_batched_total_s"] = round(perf_counter() - before, 3)
        print(f"Sentence Transformers {lang} queries {len(captions)}", flush=True)

    memory = {}
    if cuda:
        memory = {
            "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        }
    info = {
        "images": [record["id"] for record in records],
        "seed": manifest["seed"],
        "max_edge": manifest["max_edge"],
        "device": encoder.device,
        "torch": torch.__version__,
        "dtype": str(next(encoder.model.parameters()).dtype),
        "load_s": round(loaded - start, 3),
        "timings": timings,
        "memory": memory,
    }
    np.savez_compressed(destination, **arrays, info=json.dumps(info))
    print(json.dumps({k: v for k, v in info.items() if k != "images"}), flush=True)


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)[:, :512]
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def ranks(queries, gallery, labels):
    scores = queries @ gallery.T
    positive = scores[np.arange(len(labels)), labels][:, None]
    earlier = np.arange(len(gallery))[None, :] < labels[:, None]
    return 1 + np.count_nonzero((scores > positive) | ((scores == positive) & earlier), axis=1)


def hits(rank):
    return {
        "queries": len(rank),
        "top1": int(np.count_nonzero(rank <= 1)),
        "top5": int(np.count_nonzero(rank <= 5)),
        "top10": int(np.count_nonzero(rank <= 10)),
        "median_rank": float(np.median(rank)),
    }


def paired_ci(new, old, labels, image_count):
    difference = (new <= 1).astype(float) - (old <= 1)
    by_image = np.bincount(labels, weights=difference, minlength=image_count)
    counts = np.bincount(labels, minlength=image_count)
    choices = np.random.default_rng(20261111).integers(0, image_count, (2000, image_count))
    deltas = by_image[choices].sum(axis=1) / counts[choices].sum(axis=1)
    return [round(value * 100, 2) for value in np.quantile(deltas, [0.025, 0.975])]


def compare(gguf_path, st_path, destination):
    with gzip.open(gguf_path, "rt", encoding="utf-8") as source:
        gguf = json.load(source)
    with np.load(st_path, allow_pickle=False) as source:
        st = {name: source[name] for name in source.files}
    info = json.loads(str(st.pop("info")))
    if info["images"] != gguf["info"]["images"]:
        raise ValueError("Image order or set differs between embeddings")
    old_images, new_images = unit(gguf["images"]), unit(st["images"])
    image_count = len(old_images)
    mixed_images = old_images.copy()
    new_indices = np.random.default_rng(20261111).permutation(image_count)[: image_count // 2]
    mixed_images[new_indices] = new_images[new_indices]
    result = {
        "dataset": "Crossmodal-3600",
        "images": image_count,
        "dimension": 512,
        "st": {k: v for k, v in info.items() if k != "images"},
        "gguf_timings": gguf["info"].get("timings", {}),
        "same_image_cosine_median": float(np.median((old_images * new_images).sum(axis=1))),
        "retrieval": {},
    }
    for lang in LANGUAGES:
        labels = st[f"labels_{lang}"]
        if not np.array_equal(labels, gguf[f"labels_{lang}"]):
            raise ValueError(f"{lang} caption order differs")
        old_queries, new_queries = unit(gguf[lang]), unit(st[lang])
        comparisons = {
            "gguf_text_to_gguf_images": ranks(old_queries, old_images, labels),
            "st_text_to_st_images": ranks(new_queries, new_images, labels),
            "st_text_to_gguf_images": ranks(new_queries, old_images, labels),
            "gguf_text_to_st_images": ranks(old_queries, new_images, labels),
            "st_text_to_half_old_half_new_images": ranks(new_queries, mixed_images, labels),
        }
        baseline = comparisons["gguf_text_to_gguf_images"]
        result["retrieval"][lang] = {
            **{
                key: {
                    **hits(rank),
                    "top1_delta_pp_95ci": paired_ci(rank, baseline, labels, image_count),
                }
                for key, rank in comparisons.items()
            },
            "same_caption_cosine_median": float(np.median((old_queries * new_queries).sum(axis=1))),
        }
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("encode")
    run.add_argument("fixtures", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--limit", type=int)
    diff = sub.add_parser("compare")
    diff.add_argument("gguf", type=Path)
    diff.add_argument("st", type=Path)
    diff.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "encode":
        encode(args.fixtures, args.output, args.limit)
    else:
        compare(args.gguf, args.st, args.output)
