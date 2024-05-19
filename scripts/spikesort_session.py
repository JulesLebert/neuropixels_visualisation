from pathlib import Path
# import spikeinterface.extractors as se
import spikeinterface.full as si

from neuropixels_visualisation.spikesorting import spikesorting_pipeline, spikesorting_postprocessing

def main():
    session_path = Path('/mnt/rockefeller/Data/Neuropixels/')
    ferret = 'F2302_Challah'
    session_name = '13052023_AM_Challah_g0'

    output_folder = Path('/mnt/a/NeuropixelData/spikesorted_single')
    working_dir = output_folder / 'tempDir' / ferret / session_name

    dp = session_path / ferret / session_name

    for stream_id in  ['imec0.ap', 'imec1.ap']:
        recording = si.read_spikeglx(dp, stream_id = stream_id)
        sorting = spikesorting_pipeline(
            recording, 
            working_directory=working_dir / stream_id,
            sorter='kilosort4'
            )
        
        # sorting = si.read_sorter_folder(output_folder / 'tempDir' / 'kilosort4_output')

        output_dir = output_folder / 'spikesorted' / ferret / session_name / stream_id
        sorting = spikesorting_postprocessing(sorting, output_folder=output_dir)

if __name__ == '__main__':
    main()