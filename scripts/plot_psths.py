from pathlib import Path

from neuropixels_visualisation.helpers.neural_analysis_helpers import NeuralDataset

def analyse_single_session(session_path, save_dir):
    save_dir.mkdir(exist_ok=True, parents=True) # make the save directory

    dp = session_path / 'phy_folder' #get the phy folder
    dataset = NeuralDataset(dp, datatype = 'neuropixel') # NeuralDataset seems to be designed to allow either neuropixel or warp. Remember: objects have methods and parameters (under those names or others). We have a bunch of methods in this thing. I've taken notes in the relevant function. Running theme though is that it mostly loads information but also contains the "ev alignment" function. ### another note taken way later: pycharm doesn't like to step through these class initialization functions. But, on initialization, all this dataset contains is the "dp" and the neural data type as a string. Real stuff happens in the "load" method.
    dataset.load()

    dataset.create_summary_pdf(save_dir, title=f'summary_data_{session_path.parents[0].name}')


def main(): #basically, I will want to make a new version of this for the concatenated case.
    save_dir = Path('D:/Jeffrey/Output/figures')

    data_path = Path('D:/Jeffrey/Output/Tones/spikesorted') # D:\Jeffrey\Output\Tones\spikesorted\F2302_Challah\06122024_AM_Challah_g0\KiloSortSortingExtractor\KiloSortSortingExtractor
    #data_path = Path('D:/Jeffrey/Output/spikesorted')
    ferret = 'F2302_Challah'
    session_name = '06122024_AM_Challah_g0' #I'll obviously have to change stuff like this for the concatenated version.
    #session_name = '13052023_AM_Challah_g0'
    probe = 'imec0.ap'
    sorter = 'KiloSortSortingExtractor'#'kilosort'
    #sorter = 'kilosort'

    dp = data_path / ferret / session_name / probe / sorter


    analyse_single_session(dp, save_dir=save_dir)

if __name__ == '__main__':
    main()