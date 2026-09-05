#!/usr/bin/env python
"""Convert an existing stacked MOUSE NeXus file to MoDaCor-style rank padding."""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import h5py
import numpy as np

DEFAULT_PRIMARY_DATA_PATH = "entry1/instrument/detector00/data"

MOUSE_STACKED_DATASETS_TO_PAD = [
    "entry1/duration",
    "entry1/experiment/chamber_pressure",
    "entry1/experiment/environment_temperature",
    "entry1/experiment/stage_temperature",
    "entry1/instrument/beamstop/transformations/arm_length",
    "entry1/instrument/beamstop/transformations/bs_r",
    "entry1/instrument/beamstop/transformations/bs_z",
    "entry1/instrument/collimator1/blade_bottom",
    "entry1/instrument/collimator1/blade_negy",
    "entry1/instrument/collimator1/blade_posy",
    "entry1/instrument/collimator1/blade_top",
    "entry1/instrument/collimator2/blade_bottom",
    "entry1/instrument/collimator2/blade_negy",
    "entry1/instrument/collimator2/blade_posy",
    "entry1/instrument/collimator2/blade_top",
    "entry1/instrument/detector00/averaged_number_of_frames",
    "entry1/instrument/detector00/count_time",
    "entry1/instrument/detector00/detectorSpecific/detector_readout_period",
    "entry1/instrument/detector00/detectorSpecific/frame_count_time",
    "entry1/instrument/detector00/detectorSpecific/frame_period",
    "entry1/instrument/detector00/detectorSpecific/nframes_sum",
    "entry1/instrument/detector00/detectorSpecific/nimages",
    "entry1/instrument/detector00/detectorSpecific/nsequences",
    "entry1/instrument/detector00/detectorSpecific/ntrigger",
    "entry1/instrument/detector00/detectorSpecific/photon_energy",
    "entry1/instrument/detector00/detectorSpecific/summation_nimages",
    "entry1/instrument/detector00/detectorSpecific/test_mode",
    "entry1/instrument/detector00/detector_module/fast_pixel_direction",
    "entry1/instrument/detector00/detector_module/module_offset",
    "entry1/instrument/detector00/detector_module/slow_pixel_direction",
    "entry1/instrument/detector00/detector_readout_time",
    "entry1/instrument/detector00/frame_exposure_time",
    "entry1/instrument/detector00/frame_total_time",
    "entry1/instrument/detector00/threshold_energy",
    "entry1/instrument/detector00/transformations/det_x",
    "entry1/instrument/detector00/transformations/det_y",
    "entry1/instrument/detector00/transformations/det_z",
    "entry1/instrument/detector00/transformations/euler_a",
    "entry1/instrument/detector00/transformations/euler_b",
    "entry1/instrument/detector00/transformations/euler_c",
    "entry1/instrument/source/current",
    "entry1/instrument/source/voltage",
    "entry1/processing/direct_beam_profile/beam_analysis/FluxOverImage",
    "entry1/processing/direct_beam_profile/beam_analysis/achieved_coverage",
    "entry1/processing/direct_beam_profile/beam_analysis/sigma_major",
    "entry1/processing/direct_beam_profile/beam_analysis/sigma_minor",
    "entry1/processing/sample_beam_profile/beam_analysis/FluxOverImage",
    "entry1/processing/sample_beam_profile/beam_analysis/flux",
    "entry1/sample/absorptionDerivedThickness",
    "entry1/sample/absorption_by_bg",
    "entry1/sample/absorption_by_sample",
    "entry1/sample/absorption_total",
    "entry1/sample/beam/flux",
    "entry1/sample/beam/incident_wavelength",
    "entry1/sample/beam/wavelength_error",
    "entry1/sample/density",
    "entry1/sample/matrixfraction",
    "entry1/sample/overall_mu",
    "entry1/sample/samplethickness",
    "entry1/sample/scattering_probability_estimate",
    "entry1/sample/temperature",
    "entry1/sample/thickness",
    "entry1/sample/transformations/sample_x",
    "entry1/sample/transformations/sample_y",
    "entry1/sample/transformations/sample_z",
    "entry1/sample/transmission",
    "entry1/sample/transmission_beam",
    "entry1/sample/transmission_correction_factor",
    "entry1/sample/transmission_image",
]


def _normalize_hdf5_path(path: str) -> str:
    return path.strip("/")


def _dataset_create_kwargs(source: h5py.Dataset, new_shape: tuple[int, ...]) -> dict:
    kwargs = {
        "shape": new_shape,
        "maxshape": new_shape,
        "dtype": source.dtype,
    }
    if source.chunks is not None:
        kwargs["chunks"] = (*source.chunks, *((1,) * (len(new_shape) - len(source.shape))))
    if source.compression is not None:
        kwargs["compression"] = source.compression
        kwargs["compression_opts"] = source.compression_opts
    if source.shuffle:
        kwargs["shuffle"] = True
    if source.fletcher32:
        kwargs["fletcher32"] = True
    if source.scaleoffset is not None:
        kwargs["scaleoffset"] = source.scaleoffset
    if source.fillvalue is not None:
        kwargs["fillvalue"] = source.fillvalue
    return kwargs


def _can_pad_dataset(dataset: h5py.Dataset, stack_length: int, target_rank: int) -> bool:
    if dataset.ndim == 0 or dataset.ndim >= target_rank:
        return False
    return dataset.shape[0] == stack_length


def _target_shape(dataset: h5py.Dataset, target_rank: int) -> tuple[int, ...]:
    return (*dataset.shape, *((1,) * (target_rank - dataset.ndim)))


def _replace_dataset_with_reshaped_copy(h5f: h5py.File, path: str, new_shape: tuple[int, ...]) -> None:
    dataset = h5f[path]
    data = np.asarray(dataset[()]).reshape(new_shape)
    attrs = dict(dataset.attrs.items())
    parent_path, name = path.rsplit("/", 1)
    parent = h5f[parent_path] if parent_path else h5f
    tmp_name = f"{name}__modacor_tmp"
    while tmp_name in parent:
        tmp_name += "_"

    replacement = parent.create_dataset(tmp_name, data=data, **_dataset_create_kwargs(dataset, new_shape))
    replacement.attrs.update(attrs)
    del parent[name]
    parent.move(tmp_name, name)


def find_datasets_to_pad(
    filename: Path,
    primary_data_path: str = DEFAULT_PRIMARY_DATA_PATH,
) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    primary_data_path = _normalize_hdf5_path(primary_data_path)
    paths = [_normalize_hdf5_path(path) for path in MOUSE_STACKED_DATASETS_TO_PAD]
    candidates: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    with h5py.File(filename, "r") as h5f:
        if primary_data_path not in h5f:
            raise ValueError(f"primary detector data path {primary_data_path!r} not found in {filename}")
        primary = h5f[primary_data_path]
        if not isinstance(primary, h5py.Dataset) or primary.ndim < 2:
            raise ValueError(f"primary detector data path {primary_data_path!r} is not a stacked dataset")
        stack_length = primary.shape[0]
        target_rank = primary.ndim

        for path in sorted(set(paths)):
            if path not in h5f or not isinstance(h5f[path], h5py.Dataset):
                continue
            dataset = h5f[path]
            if _can_pad_dataset(dataset, stack_length=stack_length, target_rank=target_rank):
                candidates.append((path, dataset.shape, _target_shape(dataset, target_rank)))

    return candidates


def convert_stacked_file_to_modacor(
    input_file: Path,
    output_file: Path,
    primary_data_path: str = DEFAULT_PRIMARY_DATA_PATH,
    overwrite: bool = False,
) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    if not input_file.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_file}")
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {output_file}")

    datasets_to_pad = find_datasets_to_pad(
        input_file,
        primary_data_path=primary_data_path,
    )

    if output_file.exists():
        output_file.unlink()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_file, output_file)

    with h5py.File(output_file, "r+") as h5f:
        for path, _old_shape, new_shape in datasets_to_pad:
            _replace_dataset_with_reshaped_copy(h5f, path, new_shape)

    return datasets_to_pad


def modacor_output_path(input_file: Path, suffix: str = "_modacor") -> Path:
    if input_file.stem.endswith(suffix):
        raise ValueError(f"{input_file} already appears to use the {suffix!r} suffix")
    return input_file.with_name(f"{input_file.stem}{suffix}{input_file.suffix}")


def expand_input_paths(path_args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for path_arg in path_args:
        if glob.has_magic(path_arg):
            matches = sorted(Path(match) for match in glob.glob(path_arg))
            if not matches:
                raise ValueError(f"input pattern did not match any files: {path_arg}")
            paths.extend(matches)
        else:
            paths.append(Path(path_arg))
    return paths


def conversion_jobs(paths: list[Path], output: Path | None = None, suffix: str = "_modacor") -> list[tuple[Path, Path]]:
    if output is not None:
        if len(paths) != 1:
            raise ValueError("--output can only be used with one input file")
        return [(paths[0], output)]
    if len(paths) == 2 and paths[0].is_file() and not paths[1].exists():
        return [(paths[0], paths[1])]
    return [(path, modacor_output_path(path, suffix=suffix)) for path in paths]


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a stacked MOUSE NeXus/HDF5 file and pad known MOUSE per-repetition datasets "
            "with trailing singleton dimensions to match the detector data rank."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "Input stacked .nxs/.h5 file(s). With one input, or with multiple existing inputs, "
            "outputs are written as <stem>_modacor<suffix>. The old 'input output' form is still "
            "accepted when the second path does not exist."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Explicit output path for a single input file.",
    )
    parser.add_argument(
        "--primary-data-path",
        default=DEFAULT_PRIMARY_DATA_PATH,
        help=f"Detector data path used to determine target rank. Default: {DEFAULT_PRIMARY_DATA_PATH}",
    )
    parser.add_argument(
        "--suffix",
        default="_modacor",
        help="Suffix to append to each input stem in auto-output mode. Default: _modacor",
    )
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite the output file if it exists")
    parser.add_argument("--dry-run", action="store_true", help="List datasets that would be padded without writing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = setup_argparser().parse_args(argv)
    jobs = conversion_jobs(expand_input_paths(args.paths), output=args.output, suffix=args.suffix)
    for input_file, output_file in jobs:
        print(f"{input_file} -> {output_file}")
        if args.dry_run:
            datasets = find_datasets_to_pad(input_file, primary_data_path=args.primary_data_path)
        else:
            datasets = convert_stacked_file_to_modacor(
                input_file,
                output_file,
                primary_data_path=args.primary_data_path,
                overwrite=args.force,
            )

        for path, old_shape, new_shape in datasets:
            print(f"  {path}: {old_shape} -> {new_shape}")
        action = "Would pad" if args.dry_run else "Padded"
        print(f"{action} {len(datasets)} dataset(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
