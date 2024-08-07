from pathlib import Path
import spikeinterface.extractors as se
import spikeinterface.full as si

from neuropixels_visualisation.helpers.helpers_spikesorting_scripts import sort_np_sessions

from neuropixels_visualisation.spikesorting import spikesorting_pipeline, spikesorting_postprocessing, spikeglx_preprocessing, spikeglx_preprocessing_preconcatenation, spikeglx_preprocessing_postconcatenation

def main():
    session_path = Path('Z:/Data/Neuropixels/F2302_Challah/') #path to where all relevant sessions are stored
    ferret = 'F2302_Challah'
    recordingZone = 'ACx_Challah'

    if recordingZone == 'ACx_Challah':
        stream_id = 'imec0.ap'
    elif recordingZone == 'PFC_shank0_Challah':
        stream_id = 'imec1.ap'
        # something else also. Need to read metadata

    output_folder = Path('D:/Jeffrey/Output')
    #SessionsInOrder = sort_np_sessions(list(session_path.glob('[0-9][0-9]*'))) # start with two numbers and I look into it
    ### below commented is for debugging
    #SessionsInOrder = sort_np_sessions(list(session_path.glob('[1][0-3]*'))) # a smaller but still disjoint subset for checking drift.
    #SessionsInOrder = sort_np_sessions(list(session_path.glob('[2][0]*')))  # an even smaller but still disjoint subset for checking drift.
    SessionsInOrder = sort_np_sessions(list(session_path.glob('1305*'))) # for debug, only use two sessions
    # session_name = '021122_trifle_pm3_g0'
    list_of_recordings = []
    for session in SessionsInOrder:
        session_name = session.name
        print(f'Processing {session_name}')
        working_dir = output_folder / 'tempDir' / ferret / session_name
        dp = session_path / session_name
        probeFolder = list(dp.glob('*' + stream_id[:-3]))
        probeFolder = probeFolder[0]
        recording = si.read_spikeglx(probeFolder, stream_id=stream_id)
        recording = spikeglx_preprocessing_preconcatenation(recording)
        list_of_recordings.append(recording)
    recording = si.concatenate_recordings(list_of_recordings)

    output_folder_temp = output_folder / 'tempDir' / ferret / recordingZone
    output_folder_sorted = output_folder / 'spikesorted' / ferret / recordingZone
    try:
        sorting = spikesorting_pipeline(
            recording,
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
