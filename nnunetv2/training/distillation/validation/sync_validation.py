from __future__ import annotations

import warnings
from time import sleep
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot


def perform_actual_validation_sync(
    trainer,
    save_probabilities: bool = False,
    use_mirroring: bool = False,
    tile_step_size: float = 0.5,
    validation_folder: str = "validation",
) -> Optional[dict]:
    """
    Synchronous validation/export for distillation runs.

    This mirrors nnUNetTrainer.perform_actual_validation but avoids sending
    very large prediction tensors across process boundaries. That can stall on
    large 3D volumes with many classes.
    """
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=True,
        device=trainer.device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        trainer.network,
        trainer.plans_manager,
        trainer.configuration_manager,
        None,
        trainer.dataset_json,
        trainer.__class__.__name__,
        trainer.inference_allowed_mirroring_axes,
    )

    validation_output_folder = join(trainer.output_folder, validation_folder)
    maybe_mkdir_p(validation_output_folder)

    # Determine validation keys for this rank.
    _, val_keys = trainer.do_split()
    last_barrier_at_idx = None
    if trainer.is_ddp:
        world_size = dist.get_world_size()
        last_barrier_at_idx = len(val_keys) // world_size - 1
        val_keys = val_keys[trainer.local_rank::world_size]

    dataset_val = trainer.dataset_class(
        trainer.preprocessed_dataset_folder,
        val_keys,
        folder_with_segs_from_previous_stage=trainer.folder_with_segs_from_previous_stage,
    )

    next_stages = trainer.configuration_manager.next_stage_names
    if next_stages is not None:
        _ = [maybe_mkdir_p(join(trainer.output_folder_base, "predicted_next_stage", n)) for n in next_stages]

    for i, k in enumerate(dataset_val.identifiers):
        trainer.print_to_log_file(f"predicting {k}")
        data, _, seg_prev, properties = dataset_val.load_case(k)

        # Convert blosc2 to numpy.
        data = data[:]

        if trainer.is_cascaded:
            seg_prev = seg_prev[:]
            data = np.vstack(
                (
                    data,
                    convert_labelmap_to_one_hot(
                        seg_prev, trainer.label_manager.foreground_labels, output_dtype=data.dtype
                    ),
                )
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = torch.from_numpy(data)

        trainer.print_to_log_file(f"{k}, shape {data.shape}, rank {trainer.local_rank}")
        output_filename_truncated = join(validation_output_folder, k)

        prediction = predictor.predict_sliding_window_return_logits(data).cpu()

        # Synchronous export avoids large IPC/pickling costs.
        export_prediction_from_logits(
            prediction,
            properties,
            trainer.configuration_manager,
            trainer.plans_manager,
            trainer.dataset_json,
            output_filename_truncated,
            save_probabilities,
        )

        if next_stages is not None:
            for n in next_stages:
                next_stage_config_manager = trainer.plans_manager.get_configuration(n)
                expected_preprocessed_folder = join(
                    nnUNet_preprocessed, trainer.plans_manager.dataset_name, next_stage_config_manager.data_identifier
                )
                dataset_class = infer_dataset_class(expected_preprocessed_folder)

                try:
                    tmp = dataset_class(expected_preprocessed_folder, [k])
                    d, _, _, _ = tmp.load_case(k)
                except FileNotFoundError:
                    trainer.print_to_log_file(
                        f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                        f"Run the preprocessing for this configuration first!"
                    )
                    continue

                target_shape = d.shape[1:]
                output_folder = join(trainer.output_folder_base, "predicted_next_stage", n)
                output_file_truncated = join(output_folder, k)

                resample_and_save(
                    prediction,
                    target_shape,
                    output_file_truncated,
                    trainer.plans_manager,
                    trainer.configuration_manager,
                    properties,
                    trainer.dataset_json,
                    default_num_processes,
                    dataset_class,
                )

        if trainer.is_ddp and last_barrier_at_idx is not None and i < last_barrier_at_idx and (i + 1) % 20 == 0:
            dist.barrier()
        else:
            # Yield briefly to keep logs responsive on very long cases.
            sleep(0.01)

    if trainer.is_ddp:
        dist.barrier()

    metrics = None
    if trainer.local_rank == 0:
        metrics = compute_metrics_on_folder(
            join(trainer.preprocessed_dataset_folder_base, "gt_segmentations"),
            validation_output_folder,
            join(validation_output_folder, "summary.json"),
            trainer.plans_manager.image_reader_writer_class(),
            trainer.dataset_json["file_ending"],
            trainer.label_manager.foreground_regions if trainer.label_manager.has_regions else trainer.label_manager.foreground_labels,
            trainer.label_manager.ignore_label,
            chill=True,
            num_processes=default_num_processes * dist.get_world_size() if trainer.is_ddp else default_num_processes,
        )
        trainer.print_to_log_file("Validation complete", also_print_to_console=True)
        trainer.print_to_log_file(
            "Mean Validation Dice: ", metrics["foreground_mean"]["Dice"], also_print_to_console=True
        )

    return metrics
