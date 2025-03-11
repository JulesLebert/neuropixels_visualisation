from pathlib import Path
import pandas as pd

import spikeinterface.extractors as se
import spikeinterface.full as si

from neuropixels_visualisation.helpers.helpers_spikesorting_scripts import sort_np_sessions, get_channelmap_names

from neuropixels_visualisation.spikesorting import spikesorting_pipeline, spikesorting_postprocessing, spikeglx_preprocessing
from neuropixels_visualisation.helpers.npyx_metadata_fct import load_meta_file
def main():
    session_path = Path('Z:/Data/Neuropixels/F2302_Challah/') #path to where all relevant sessions are stored
    ferret = 'F2302_Challah'
    recordingZone = 'ACx_Challah'

    if recordingZone == 'ACx_Challah':
        stream_id = 'imec0.ap'
        channel_map_to_use = 'Challah_top_b1_horizontal_band_ground.imro'
    elif recordingZone == 'PFC_shank0_Challah':
        stream_id = 'imec1.ap'
        # something else also. Need to read metadata

    output_folder = Path('D:/Jeffrey/Output')
    bonusName = 'Tens_Of_June'
    #SessionsInOrder = sort_np_sessions(list(session_path.glob('[0-9][0-9]*'))) # start with two numbers and I look into it
    ### below commented is for debugging
    ### pre-code checks
    SessionsInOrder = sort_np_sessions(list(session_path.glob('1[0-9]06*')))# should only do June 10s.
    #SessionsInOrder = sort_np_sessions(list(session_path.glob('[1][0-3]*'))) # a smaller but still disjoint subset for checking drift.
    #SessionsInOrder = sort_np_sessions(list(session_path.glob('[2][0]*')))  # an even smaller but still disjoint subset for checking drift.
    #SessionsInOrder = sort_np_sessions(list(session_path.glob('1305*'))) # for debug, only use two sessions
    # SessionsInOrder = sort_np_sessions(list(session_path.glob('03062024_PM*'))) # for debug, only use two sessions
    # session_name = '021122_trifle_pm3_g0'
    ### Make a file that keeps track of the recording info
    multirec_info = {'name': [],
         'start_time': [],
         # 'stop_time': [],
         'duration': [],
         'fs': [],
         'n_samples': [],
         'multirec_start_sample': [],
         'multirec_stop_sample': []}
    dict_of_recordings = {}
    i = 0
    for session in SessionsInOrder:
        session_name = session.name
        print(f'Processing {session_name}')
        working_dir = output_folder / 'tempDir' / ferret / session_name
        dp = session_path / session_name
        probeFolder = list(dp.glob('*' + stream_id[:-3]))
        probeFolder = probeFolder[0]
        recording = si.read_spikeglx(probeFolder, stream_id=stream_id)
        recording = spikeglx_preprocessing(recording)
        ### do things related to the construction of a file which stores the recording information.
        multirec_info['name'].append(session_name)
        multirec_info['fs'].append(recording.get_sampling_frequency())
        multirec_info['n_samples'].append(recording.get_num_samples())
        multirec_info['duration'].append(recording.get_total_duration())

        meta = load_meta_file(probeFolder / (session_name + '_t0.imec0.ap.meta'))
        multirec_info['start_time'].append(meta['fileCreateTime'])

        if i == 0:
            multirec_info['multirec_start_sample'].append(0)
        else:
            # multirec_info['multirec_start_sample'].append(int(
            #     multirec_info['multirec_start_sample'][i-1] + (multirec_info['duration'][i-1].total_seconds() * multirec_info['fs'][i-1])+1))

            multirec_info['multirec_start_sample'].append(
                multirec_info['multirec_start_sample'][i - 1] + (multirec_info['n_samples'][i - 1]) + 1)

        # multirec_info['multirec_stop_sample'].append(int(multirec_info['multirec_start_sample'][i] + (multirec_info['duration'][i].total_seconds() * multirec_info['fs'][i])))
        multirec_info['multirec_stop_sample'].append(
            multirec_info['multirec_start_sample'][i] + (multirec_info['n_samples'][i]))

        chan_dict = get_channelmap_names(dp)  # almost works but something about the format is different. no "imRoFile" perameter. There is something called an "imRoTable" which is probably also what I want. But let's deal with this later, when we know we need it. Because, honestly, we want something more sophisticated than this eventually.
        chan_map_name = chan_dict[session_name + "_" + stream_id[:-3]]

        if chan_map_name == channel_map_to_use:  # I did this for basically no reason. The reason is because it fits the expected format of Jules and because someday I may want to organize things by the probe map.
            if chan_map_name in dict_of_recordings:
                dict_of_recordings[chan_map_name].append(recording)
            else:
                dict_of_recordings[chan_map_name] = [recording]
        i += 1
    multirecordings = {channel_map: si.concatenate_recordings(dict_of_recordings[channel_map]) for channel_map in
                       dict_of_recordings}
    multirecordings = {channel_map: multirecordings[channel_map].set_probe(dict_of_recordings[channel_map][0].get_probe())
                       for channel_map in multirecordings}
    #recording = si.concatenate_recordings(dict_of_recordings)

    output_folder_temp = output_folder / 'tempDir' / ferret / recordingZone / bonusName
    output_folder_sorted = output_folder / 'spikesorted' / ferret / recordingZone /bonusName
    phy_folder = output_folder_sorted / 'KiloSortSortingExtractor' / 'phy_folder' #probably.
    ### save multirec_info
    phy_folder.mkdir(parents=True, exist_ok=True)
    df_rec = pd.DataFrame(multirec_info)
    df_rec.to_csv(phy_folder / 'multirec_info.csv',
                  index=False)  # this is the earliest phy folder around and may be a problem...

    try:
        sorting = spikesorting_pipeline(
            multirecordings[chan_map_name],
            output_folder=output_folder_temp,
            sorter='kilosort4',
            concatenated=True
            )

        # sorting = si.read_sorter_folder(output_folder / 'tempDir' / 'kilosort4_output')
        output_dir = output_folder_sorted
        sorting = spikesorting_postprocessing(sorting, output_folder=output_dir)
    except Exception as e:
        print(f'Error processing {session_name}: {e}')

if __name__ == '__main__':
    main()
