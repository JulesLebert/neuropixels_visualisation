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

    data_path = Path('/mnt/a/NeuropixelData/output_jeffrey/spikesorted')
    ferret = 'F1903_Trifle'
    session_name = '021122_trifle_pm3_g0'
    sorter = 'kilosort'

    dp = data_path / ferret / session_name / sorter

    analyse_single_session(dp, save_dir=save_dir)

if __name__ == '__main__':
    main()