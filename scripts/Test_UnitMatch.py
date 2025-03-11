import sys
from pathlib import Path

import os
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
import UnitMatchPy.extract_raw_data as erd ### makes UnitMatchPy a requirement obviously.
import numpy as np

from neuropixels_visualisation.spikesorting import spikeglx_preprocessing, spikesorting_pipeline, spikesorting_postprocessing
from neuropixels_visualisation.helpers.helpers_spikesorting_scripts import sort_np_sessions, get_channelmap_names

#### below imports for running UnitMatch

#load_ext autoreload
#autoreload

import UnitMatchPy.bayes_functions as bf
import UnitMatchPy.utils as util
import UnitMatchPy.overlord as ov
import numpy as np
import matplotlib.pyplot as plt
import UnitMatchPy.save_utils as su
import UnitMatchPy.GUI as gui
import UnitMatchPy.assign_unique_id as aid
import UnitMatchPy.default_params as default_params
def main():

# Make list of recordings/sortings to iterate over
    session_path = Path('Z:/Data/Neuropixels/F2302_Challah/') #path to where all relevant sessions are stored
    session_path_KS = Path('D:/Jeffrey/Output/spikesorted/F2302_Challah/')
    ferret = 'F2302_Challah'
    detectBadChannels = 0 # should be 1 unless debugging
    usedStream_id = ['imec0.ap'] ### ['imec0.ap', 'imec1.ap']
    recordingZone = 'ACx_Challah'

    if recordingZone == 'ACx_Challah':
        stream_id = 'imec0.ap'
        channel_map_to_use = 'Challah_top_b1_horizontal_band_ground.imro'
    elif recordingZone == 'PFC_shank0_Challah':
        stream_id = 'imec1.ap'
        # something else also. Need to read metadata

    output_folder = Path('D:/Jeffrey/Output')
    sessionSetLabel = 'TheFirstDay'


    if sessionSetLabel == 'All_ACx_Top':
        sessionString = '[0-9][0-9]*'  ### this actually selects more than just the top
    elif sessionSetLabel == 'Tens_Of_June':
        sessionString = '1[0-9]06*'
    elif sessionSetLabel == 'TheFirstDay':
        sessionString = '1305*'
    elif sessionSetLabel == 'TheFirstSession':
        sessionString = '1305*AM*'

    SessionsInOrder = sort_np_sessions(list(session_path.glob(sessionString)))
    output_folder = Path('D:/Jeffrey/Output/')
    recordings = {}
    KS_dirs = {}
    for i,session in enumerate(SessionsInOrder):
        session_name = session.name
        print(f'Processing {session_name}')
        KS_dirs[i] = output_folder / 'spikeSorted' / ferret / session_name / stream_id / 'kilosort' / 'phy_folder' ### the placement of stream_id here must br wrong. Need to think about it though.
        dp = session_path / session_name
        for stream_id in usedStream_id:
            probeFolder = list(dp.glob('*' + stream_id[:-3]))
            probeFolder = probeFolder[0]
            recording = si.extractors.read_spikeglx(dp, stream_id = stream_id)
            #recording = si.extractors.read_cbin_ibl(probeFolder)  # for compressed, which only exists on myriad
            recording = spikeglx_preprocessing(recording,doDetectBadChannels=detectBadChannels) # do I want to have done the preprocessing for this? Seems like yes, it is in the example, at least the phase shift and filter.
        chan_dict = get_channelmap_names(
            dp)  # almost works but something about the format is different. no "imRoFile" perameter. There is something called an "imRoTable" which is probably also what I want. But let's deal with this later, when we know we need it. Because, honestly, we want something more sophisticated than this eventually.
        chan_map_name = chan_dict[session_name + "_" + stream_id[:-3]]

        if chan_map_name == channel_map_to_use:  # I did this for basically no reason. The reason is because it fits the expected format of Jules and because someday I may want to organize things by the probe map.
            if chan_map_name in recordings:
                recordings[chan_map_name].append(recording)
            else:
                recordings[chan_map_name] = [recording]
    ### Haven't done, but need a list of recordings and sortings... I can grab the recordings-list old-style...


    Sessions_KS = list(session_path_KS.glob(sessionString)) # unsorted, which is a big problem if I need to match these (which I do), but there might be alternate means somewhere.
    sortings = {}
    for i,session in enumerate(Sessions_KS):
        session_name = session.name
        print(f'Processing {session_name}')
        for stream_id in usedStream_id:
            sorter_folder = list(session.glob('*' + stream_id))
            sorter_folder = sorter_folder[0]
            #sorting = ss.read_sorter_folder(sorter_folder) ### this works with the tempDir folder available, the latter requires the phy output somehow.
            sorting = se.read_kilosort(sorter_folder / 'kilosort\phy_folder')

        if i > 0:
            sortings.append(sorting)
        else:
            sortings = [sorting]

    # Will only make average waveforms for good units
    extract_good_units_only = False ### Might need to turn this off since I plan to not spikesort everything. But there's an argument for doing that manual curation for only a few key sessions, then letting UnitMatch decide on which are kept.

    #Getting good units only
    sortings[0].get_property_keys() #lists keys for attached properties if 'quality' is not suitable

    #Good units which will be used in Unit Match
    good_units = []
    units_used = []
    for i, sorting in enumerate(sortings):
        unit_ids_tmp = sorting.get_property('original_cluster_id')
        is_good_tmp = sorting.get_property('quality')
        good_units.append(np.stack((unit_ids_tmp,is_good_tmp), axis = 1))

        units_used.append(unit_ids_tmp)
        if extract_good_units_only is True:
            keep = np.argwhere(is_good_tmp == 'good').squeeze()
            sortings[i] = sorting.select_units(keep)


    # Split each recording/sorting into 2 halves
    for i, sorting in enumerate(sortings):
        split_idx = recordings[chan_map_name][i].get_num_samples() // 2

        split_sorting = []
        split_sorting.append(sorting.frame_slice(start_frame=0, end_frame=split_idx))
        split_sorting.append(sorting.frame_slice(start_frame=split_idx, end_frame=recordings[chan_map_name][i].get_num_samples()))

        sortings[i] = split_sorting ### UnitMatch overwrites it, so so shall we.

    for i, recording in enumerate(recordings[chan_map_name]):
        split_idx = recording.get_num_samples() // 2

        split_recording = []
        split_recording.append(recording.frame_slice(start_frame=0, end_frame=split_idx))
        split_recording.append(recording.frame_slice(start_frame=split_idx, end_frame=recording.get_num_samples()))

        recordings[chan_map_name][i] = split_recording

    print('here is the point to which I have tracked')


    # create sorting analyzer for each pair
    analysers = []
    for i, recording in enumerate(recordings[chan_map_name]):
        split_analysers = []

        split_analysers.append(si.create_sorting_analyzer(sortings[i][0], recordings[chan_map_name][i][0], sparse=False))
        split_analysers.append(si.create_sorting_analyzer(sortings[i][1], recordings[chan_map_name][i][1], sparse=False))
        analysers.append(split_analysers)


    # create the fast template extension for each sorting analyser
    all_waveforms = []
    for i in range(len(analysers)):
        for half in range(2):
            analysers[i][half].compute(
                "random_spikes",
                method="uniform",
                max_spikes_per_unit=500)

            # Analysers[i][half].compute('fast_templates', n_jobs = 0.8,  return_scaled=True)
            #analysers[i][half].compute('fast_templates', n_jobs=0.8)
            analysers[i][half].compute('templates', n_jobs=0.8)

        #templates_first = analysers[i][0].get_extension('fast_templates')
        #templates_second = analysers[i][1].get_extension('fast_templates')
        templates_first = analysers[i][0].get_extension('templates')
        templates_second = analysers[i][1].get_extension('templates')
        t1 = templates_first.get_data()
        t2 = templates_second.get_data()
        all_waveforms.append(np.stack((t1, t2), axis=-1))

    # Make a channel_positions array
    all_positions = []
    for i in range(len(analysers)):
        # positions for first half and second half are the same  #### however, this gives me the potential to compare the ground a ref sessions
        all_positions.append(analysers[i][0].get_channel_locations())

### Save extracted data in a unit match friendly folder
    #UM_input_dir = os.path.join(os.getcwd(), 'UMInputData') ### currently wrong. Need this to be in the phy folder in the kilosort output. Notably also, I think I overwrite the channel_map.npy which is in there natively... So I should take some kind of care with that.

    #Path(UM_input_dir).mkdir(parents=True,exist_ok=True)

    all_session_paths = []
    for i,session in enumerate(SessionsInOrder):
        for j in range(0,2):
            session_x_path = os.path.join(KS_dirs[i], f'Half{j + 1}')  # lets start at 1 ### half may be wrong, there might be supposed to be different sessions?
            Path(session_x_path).mkdir(parents=True, exist_ok=True)

            # save the GoodUnits as a .rsv first column is unit ID,second is 'good' or 'mua'
            good_units_path = os.path.join(session_x_path, 'cluster_group.tsv')
            channel_positions_path = os.path.join(session_x_path, 'channel_positions.npy')
            save_good_units = np.vstack(
                (np.array(('cluster_id', 'group')), good_units[i]))  # Title of colum one is '0000' Not 'cluster_id')
            save_good_units[0, 0] = 0  # need to be int to use np.savetxt
            np.savetxt(good_units_path, save_good_units, fmt=['%d', '%s'], delimiter='\t')
            if extract_good_units_only:
                Units = np.argwhere(good_units[0][:, 1] == 'good')
                erd.save_avg_waveforms(all_waveforms[i], session_x_path, Units, extract_good_units_only=extract_good_units_only)
            else:
                erd.save_avg_waveforms(all_waveforms[i], session_x_path, good_units[i],
                                       extract_good_units_only=extract_good_units_only)
            np.save(channel_positions_path, all_positions[i])

            all_session_paths.append(session_x_path)

### Run UnitMatch

    # get default parameters, can add your own before or after!

    # default of Spikeinterface as by default spike interface extracts waveforms in a different manner.
    param = {'SpikeWidth': 90, 'waveidx': np.arange(20, 50), 'PeakLoc': 35}
    param = default_params.get_default_param()

    param['KS_dirs'] = KS_dirs
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(KS_dirs)

    #read in data and select the good units and exact metadata
    waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(wave_paths, unit_label_paths, param, good_units_only = extract_good_units_only)

    #param['peak_loc'] = #may need to set as a value if the peak location is NOT ~ half the spike width

    # create clus_info, contains all unit id/session related info
    clus_info = {'good_units' : good_units, 'session_switch' : session_switch, 'sessions_id' : session_id,
                'original_ids' : np.concatenate(good_units) }

    #Extract parameters from waveform
    extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info, param)

    #Extract metric scores
    total_score, candidate_pairs, scores_to_include, predictors  = ov.extract_metric_scores(extracted_wave_properties, session_switch, within_session, param, niter  = 2)

    #Probability analysis
    prior_match = 1 - (param['n_expected_matches'] / param['n_units']**2 ) # freedom of choose in prior prob
    priors = np.array((prior_match, 1-prior_match))

    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    score_vector = param['score_vector']
    parameter_kernels = np.full((len(score_vector), len(scores_to_include), len(cond)), np.nan)

    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one = 1)

    probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, cond)

    output_prob_matrix = probability[:,1].reshape(param['n_units'],param['n_units'])

    util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold = 0.75)

    match_threshold = param['match_threshold']
    #match_threshold = try different values here!

    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > match_threshold] = 1

    plt.imshow(output_threshold, cmap = 'Greys')

    amplitude = extracted_wave_properties['amplitude']
    spatial_decay = extracted_wave_properties['spatial_decay']
    avg_centroid = extracted_wave_properties['avg_centroid']
    avg_waveform = extracted_wave_properties['avg_waveform']
    avg_waveform_per_tp = extracted_wave_properties['avg_waveform_per_tp']
    wave_idx = extracted_wave_properties['good_wave_idxs']
    max_site = extracted_wave_properties['max_site']
    max_site_mean = extracted_wave_properties['max_site_mean']
    gui.process_info_for_GUI(output_prob_matrix, match_threshold, scores_to_include, total_score, amplitude, spatial_decay,
                         avg_centroid, avg_waveform, avg_waveform_per_tp, wave_idx, max_site, max_site_mean,
                         waveform, within_session, channel_pos, clus_info, param)

    is_match, not_match, matches_GUI = gui.run_GUI()

    #this function has 2 mode 'And' 'Or', which returns a matches if they appear in both or one cv pair
    #then it will add all the matches selected as IsMaatch, then remove all matches in NotMatch
    matches_curated = util.curate_matches(matches_GUI, is_match, not_match, mode = 'And')

    matches = np.argwhere(match_threshold == 1)
    UIDs = aid.assign_unique_id(output_prob_matrix, param, clus_info)

    save_dir = session_path_KS / 'UnitMatchOutput'
    #NOTE - change to matches to matches_curated if done manual curation with the GUI
    su.save_to_output(save_dir, scores_to_include, matches # matches_curated
                  , output_prob_matrix, avg_centroid, avg_waveform, avg_waveform_per_tp, max_site,
                   total_score, output_threshold, clus_info, param, UIDs = UIDs, matches_curated = None, save_match_table = True)


    print('jaha')
def scratchScript():
    import dill
    dill.load_session('saveState.pkl')
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(KS_dirs)


    param = default_params.get_default_param()
    param['no_shanks'] = 4 #{'SpikeWidth': 90, 'waveidx': np.arange(20, 50), 'PeakLoc': 35} ### I put this afterward; it was originally before loading the defaults but in fact, this line was overwritten. There appears to be some naming cnvention differences; PeakLoc for example is peak_loc.
    param['shank_dist'] = 250 ### assuming that we want micrometers. But who knows.
    param['KS_dirs'] = KS_dirs ### gotta manually do param because despire the saveState param is out of scope?

    # read in data and select the good units and exact metadata
    waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(wave_paths,
                                                                                                       unit_label_paths,
                                                                                                       param,
                                                                                                       good_units_only=extract_good_units_only)

    # param['peak_loc'] = #may need to set as a value if the peak location is NOT ~ half the spike width

    # create clus_info, contains all unit id/session related info
    clus_info = {'good_units': good_units, 'session_switch': session_switch, 'session_id': session_id,
                 'original_ids': np.concatenate(good_units)}

    # Extract parameters from waveform
    extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info, param)

    # Extract metric scores
    total_score, candidate_pairs, scores_to_include, predictors = ov.extract_metric_scores(extracted_wave_properties,
                                                                                           session_switch,
                                                                                           within_session, param,
                                                                                           niter=2)

    # Probability analysis
    prior_match = 1 - (param['n_expected_matches'] / param['n_units'] ** 2)  # freedom of choose in prior prob
    priors = np.array((prior_match, 1 - prior_match))

    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    score_vector = param['score_vector']
    parameter_kernels = np.full((len(score_vector), len(scores_to_include), len(cond)), np.nan)

    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one=1)

    probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, cond)

    output_prob_matrix = probability[:, 1].reshape(param['n_units'], param['n_units'])

    util.evaluate_output(output_prob_matrix, param, within_session, session_switch, match_threshold=0.75)

    match_threshold = param['match_threshold']
    # match_threshold = try different values here!

    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > match_threshold] = 1

    plt.imshow(output_threshold, cmap='Greys')

    amplitude = extracted_wave_properties['amplitude']
    spatial_decay = extracted_wave_properties['spatial_decay']
    avg_centroid = extracted_wave_properties['avg_centroid']
    avg_waveform = extracted_wave_properties['avg_waveform']
    avg_waveform_per_tp = extracted_wave_properties['avg_waveform_per_tp']
    wave_idx = extracted_wave_properties['good_wave_idxs']
    max_site = extracted_wave_properties['max_site']
    max_site_mean = extracted_wave_properties['max_site_mean']
    gui.process_info_for_GUI(output_prob_matrix, match_threshold, scores_to_include, total_score, amplitude,
                             spatial_decay,
                             avg_centroid, avg_waveform, avg_waveform_per_tp, wave_idx, max_site, max_site_mean,
                             waveform, within_session, channel_pos, clus_info, param)  ### If you haven't done manual sorting then you won't have anything in clus_info... If you don't have anything in clus_info then the code breaks.

    is_match, not_match, matches_GUI = gui.run_GUI()

    # this function has 2 mode 'And' 'Or', which returns a matches if they appear in both or one cv pair
    # then it will add all the matches selected as IsMaatch, then remove all matches in NotMatch
    matches_curated = util.curate_matches(matches_GUI, is_match, not_match, mode='And')

    matches = np.argwhere(match_threshold == 1)
    UIDs = aid.assign_unique_id(output_prob_matrix, param, clus_info)

    save_dir = session_path_KS / 'UnitMatchOutput'
    # NOTE - change to matches to matches_curated if done manual curation with the GUI
    su.save_to_output(save_dir, scores_to_include, matches  # matches_curated
                      , output_prob_matrix, avg_centroid, avg_waveform, avg_waveform_per_tp, max_site,
                      total_score, output_threshold, clus_info, param, UIDs=UIDs, matches_curated=None,
                      save_match_table=True)

    print('jaha')
def zero_center_waveform(waveform):
    """
    Centers waveform about zero, by subtracting the mean of the first 15 time points.
    This function is useful for Spike Interface where the waveforms are not centered about 0.

    Arguments:
        waveform - ndarray (nUnits, Time Points, Channels, CV)

    Returns:
        Zero centered waveform
    """
    waveform = waveform -  np.broadcast_to(waveform[:,:15,:,:].mean(axis=1)[:, np.newaxis,:,:], waveform.shape)
    return waveform
if __name__ == '__main__':
    scratchScript()
    #main()