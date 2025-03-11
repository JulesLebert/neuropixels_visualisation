from pathlib import Path
# import spikeinterface.extractors as se
import spikeinterface.full as si


from neuropixels_visualisation.spikesorting import spikesorting_pipeline, spikesorting_postprocessing, spikeglx_preprocessing

def main():

    session_path = Path('Z:/Data/Neuropixels')
    ferret = 'F2302_Challah'  # chooses the ferret folder.
    #ferret = 'F2105_Clove'  # chooses the ferret folder.
    #ferret = 'F1903_Trifle'  # chooses the ferret folder.
    session_name = '06122024_AM_Challah_g0'  # chooses the session folder. Use this to select specific sessions. You can write something to do this batch later if you want, but seems likely that you'll instead be changing this as you go.
    #session_name = '021122_trifle_am_g0'
    output_folder = Path('D:/Jeffrey/Output/Tones/')

    dp = session_path / ferret / session_name

    recording = si.read_spikeglx(dp, stream_id = 'imec0.ap')
    recording = spikeglx_preprocessing(recording,1)
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