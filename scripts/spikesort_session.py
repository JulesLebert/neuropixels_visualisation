from pathlib import Path
# import spikeinterface.extractors as se
import spikeinterface.full as si

from neuropixels_visualisation.spikesorting import spikesorting_pipeline, spikesorting_postprocessing

def main():
    session_path = Path('/mnt/a/NeuropixelData/raw')
    ferret = 'F1903_Trifle'
    session_name = '021122_trifle_pm3_g0'

    output_folder = Path('/mnt/a/NeuropixelData/output_jeffrey')

    dp = session_path / ferret / session_name

    recording = si.read_spikeglx(dp, stream_id = 'imec0.ap')
    sorting = spikesorting_pipeline(
        recording, 
        output_folder=output_folder,
        sorter='kilosort4'
        )
    
    # sorting = si.read_sorter_folder(output_folder / 'tempDir' / 'kilosort4_output')

    output_dir = output_folder / 'spikesorted' / ferret / session_name
    sorting = spikesorting_postprocessing(sorting, output_folder=output_dir)

if __name__ == '__main__':
    main()