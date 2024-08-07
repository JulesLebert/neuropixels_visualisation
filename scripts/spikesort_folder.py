from pathlib import Path
# import spikeinterface.extractors as se
import spikeinterface.full as si

from neuropixels_visualisation.spikesorting import spikesorting_pipeline, spikesorting_postprocessing

def main():
    session_path = Path('Z:/Data/Neuropixels/F2302_Challah/') #path to where all relevant sessions are stored
    ferret = 'F2302_Challah'
    output_folder = Path('D:/Jeffrey/Output')

    # session_name = '021122_trifle_pm3_g0'
    for session in session_path.glob('*'):
        session_name = session.name
        print(f'Processing {session_name}')
        working_dir = output_folder / 'tempDir' / ferret / session_name
        dp = session_path / session_name

        for stream_id in  ['imec0.ap', 'imec1.ap']:
            recording = si.read_spikeglx(dp, stream_id = stream_id)
            try:
                sorting = spikesorting_pipeline(
                    recording, 
                    output_folder=working_dir / stream_id,
                    sorter='kilosort4'
                    )
                
                # sorting = si.read_sorter_folder(output_folder / 'tempDir' / 'kilosort4_output')

                output_dir = output_folder / 'spikesorted' / ferret / session_name / stream_id
                sorting = spikesorting_postprocessing(sorting, output_folder=output_dir)
            except Exception as e:
                print(f'Error processing {session_name}: {e}')
                continue

if __name__ == '__main__':
    main()