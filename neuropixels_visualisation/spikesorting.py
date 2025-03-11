from pathlib import Path

import spikeinterface.full as si

def spikeglx_preprocessing(recording,doDetectBadChannels =1):

    recording = si.phase_shift(recording) #mandatory for NP recordings because the channels are not sampled at the same time. make
    recording = si.common_reference(recording, reference='global', operator='median')
    recording = si.bandpass_filter(recording, freq_min=300, freq_max=6000)
    if doDetectBadChannels:
        bad_channel_ids, channel_labels = si.detect_bad_channels(recording) # took tghis out temporarily because it was causing an issue... To be investigated on Cuttlefish.
        recording = recording.remove_channels(bad_channel_ids)
    return recording

def spikesorting_pipeline(recording, output_folder, sorter='kilosort4',concatenated=False):
    working_directory = Path(output_folder) / 'tempDir'

    # if (working_directory / 'binary.json').exists():
    #     recording = si.load_extractor(working_directory)
    # else:
    # job_kwargs = dict(n_jobs=-1, chunk_duration='1s', progress_bar=True)
    # recording = recording.save(folder = working_directory, format='binary', **job_kwargs)

    sorting = si.run_sorter(
        sorter_name=sorter, 
        recording=recording, 
        output_folder = working_directory / f'{sorter}_output',
        verbose=True,
        )
    
    return sorting


def spikesorting_postprocessing(sorting, output_folder):
    # output_folder.mkdir(exist_ok=True, parents=True)
    # rec = sorting._recording
    # outDir = output_folder/ sorting.name
    #
    # jobs_kwargs = dict(n_jobs=-1, chunk_duration='1s', progress_bar=True)
    # sorting = si.remove_duplicated_spikes(sorting, censored_period_ms=2)
    #
    # if (outDir / 'waveforms_folder').exists():
    #     we = si.load_waveforms(
    #         outDir / 'waveforms_folder',
    #         sorting=sorting,
    #         with_recording=True,
    #         )
    #
    # else:
    #     we = si.extract_waveforms(rec, sorting, outDir / 'waveforms_folder',
    #         overwrite=False,
    #         ms_before=2,
    #         ms_after=3.,
    #         max_spikes_per_unit=300,
    #         sparse=True,
    #         num_spikes_for_sparsity=100,
    #         method="radius",
    #         radius_um=40,
    #         **jobs_kwargs,
    #         )
    #
    # if not (outDir / 'report').exists():
    #     metrics = si.compute_quality_metrics(
    #         we,
    #         n_jobs = jobs_kwargs['n_jobs'],
    #         verbose=True,
    #         )
    #
    #     si.export_to_phy(we, outDir / 'phy_folder',
    #                     verbose=True,
    #                     compute_pc_features=False,
    #                     copy_binary=False,
    #                     remove_if_exists=True,
    #                     **jobs_kwargs,
    #                     )
    #
    #     si.export_report(we, outDir / 'report',
    #             format='png',
    #             force_computation=True,
    #             **jobs_kwargs,
    #             )
    output_folder.mkdir(exist_ok=True, parents=True)
    rec = sorting._recording
    outDir = output_folder / sorting.name
    censored_period_ms = 0.5 ### Jules had this as 2 but he claimed this was above the default. Using 2 would probably miss inhibitory cells.
    si.set_global_job_kwargs(n_jobs=-1)
    jobs_kwargs = dict(chunk_duration='1s', progress_bar=True)
    sorting = si.remove_duplicated_spikes(sorting, censored_period_ms=censored_period_ms)
    sorting = si.remove_excess_spikes(sorting, rec)

    if (outDir / 'sortings_folder').exists():
        we = si.load_waveforms(
            outDir / 'sortings_folder',
            sorting=sorting,
            with_recording=True,
        )

    else:
        # we = si.create_sorting_analyzer() # need to figure this out later.
        # check https://spikeinterface.readthedocs.io/en/latest/tutorials/waveform_extractor_to_sorting_analyzer.html
        # The reason for the update was to not have to extract waveforms for things that didn't need them.
        #
        we = si.create_sorting_analyzer(recording=rec, sorting=sorting, folder=outDir / 'sortings_folder',
                                        format="binary_folder",
                                        sparse=True
                                        )

        ### I've basically translated the compatibility code below.

        # other_kwargs, new_job_qwargs = split_job_kwargs(**jobs_kwargs)

        if not we.has_extension("random_spikes"):
            we.compute("random_spikes",max_spikes_per_unit=300) ### subselects a number of spikes for downstream analysis.
            print('Marker 3')
        if not we.has_extension("waveforms"):
            we.compute("waveforms",ms_before=2,ms_after=3.) ### has the waveforms corresponding to the subset of spikes selected above, on all relevant channels (with relevancy decided by the "sparsity"). You in fact only need to collect these to calculate the templates, which can also be calculated directly from the random spikes and raw data. But, the waveforms file should be smaller than the raw data I think, and also if you use the raw data directly, apparently you can only calculate mean and standard deviation, and not median or percentile. Anyway, I dont mind calculating waveforms as long as it works.
            print('Marker 4')
        if not we.has_extension("templates"):
            we.compute("templates") ### used for downstream analysis and are useful to look at to see if your results make sense.
            print('Marker 5')
        if not we.has_extension("noise_levels"):
            we.compute("noise_levels") ### I worry how useful this would be for my concatenated data...
            print('Marker 6')
        if not we.has_extension("spike_amplitudes"):
            print('Marker 6.5')
            we.compute("spike_amplitudes")
            print('Marker 7')
        if not we.has_extension("template_similarity"):
            we.compute("template_similarity")
            print('Marker 8')
    if not (outDir / 'report').exists():
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
        metrics = si.compute_quality_metrics( #### this seems defunct. Seems like I want something related to "ComputeQualityMetrics" here.
            we,
            n_jobs=jobs_kwargs['n_jobs'],
            verbose=True,
        )