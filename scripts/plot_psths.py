from pathlib import Path

from neuropixels_visualisation.helpers.neural_analysis_helpers import NeuralDataset

def analyse_single_session(session_path, save_dir):
    save_dir.mkdir(exist_ok=True, parents=True)

    dp = session_path / 'phy_folder'
    dataset = NeuralDataset(dp, datatype = 'neuropixel')
    dataset.load()

    dataset.create_summary_pdf(save_dir, title=f'summary_data_{session_path.parents[0].name}')


def main():
    save_dir = Path('/mnt/a/NeuropixelData/output_jeffrey/figures')

    data_path = Path('/mnt/a/NeuropixelData/spikesorted_single/spikesorted/')
    ferret = 'F2302_Challah'
    session_name = '13052023_AM_Challah_g0'
    stream_id = 'imec1.ap'
    sorter = 'kilosort'

    dp = data_path / ferret / session_name / stream_id / sorter

    analyse_single_session(dp, save_dir=save_dir)

if __name__ == '__main__':
    main()