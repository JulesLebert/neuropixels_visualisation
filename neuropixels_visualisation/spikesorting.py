from pathlib import Path

import spikeinterface.full as si

def spikeglx_preprocessing(recording):
    recording = si.bandpass_filter(recording, freq_min=300, freq_max=6000)
    bad_channel_ids, channel_labels = si.detect_bad_channels(recording)
    recording = recording.remove_channels(bad_channel_ids)
    recording = si.phase_shift(recording)
    recording = si.common_reference(recording, reference='global', operator='median')

    return recording


def spikesorting_pipeline(recording, working_directory, sorter='kilosort4'):
    # working_directory = Path(output_folder) / 'tempDir' / recording.name

    if (working_directory / 'binary.json').exists():
        recording = si.load_extractor(working_directory)
    else:
        recording = spikeglx_preprocessing(recording)
        job_kwargs = dict(n_jobs=-1, chunk_duration='1s', progress_bar=True)
        recording = recording.save(folder = working_directory, format='binary', **job_kwargs)

    sorting = si.run_sorter(
        sorter_name=sorter, 
        recording=recording, 
        output_folder = working_directory / f'{sorter}_output',
        verbose=True,
        )
    
    return sorting


def spikesorting_postprocessing(sorting, output_folder):
    output_folder.mkdir(exist_ok=True, parents=True)
    rec = sorting._recording
    outDir = output_folder/ sorting.name

    jobs_kwargs = dict(n_jobs=-1, chunk_duration='1s', progress_bar=True)
    sorting = si.remove_duplicated_spikes(sorting, censored_period_ms=2)

    if (outDir / 'waveforms_folder').exists():
        we = si.load_waveforms(
            outDir / 'waveforms_folder', 
            sorting=sorting,
            with_recording=True,
            )

    else:
        we = si.extract_waveforms(rec, sorting, outDir / 'waveforms_folder',
            overwrite=False,
            ms_before=2, 
            ms_after=3., 
            max_spikes_per_unit=300,
            sparse=True,
            num_spikes_for_sparsity=100,
            method="radius",
            radius_um=40,
            **jobs_kwargs,
            )
        
    if not (outDir / 'report').exists():
        metrics = si.compute_quality_metrics(
            we, 
            n_jobs = jobs_kwargs['n_jobs'], 
            verbose=True,
            )
        
        si.export_to_phy(we, outDir / 'phy_folder', 
                        verbose=True, 
                        compute_pc_features=False,
                        copy_binary=False,
                        remove_if_exists=True,
                        **jobs_kwargs,
                        )

        si.export_report(we, outDir / 'report',
                format='png',
                force_computation=True,
                **jobs_kwargs,
                )
