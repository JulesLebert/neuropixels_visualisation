# Goal with this script is to load the sorting output of any
# particular spikesorting attempt and then apply functions which
# will help me understand it. The scope will be dependent on whatever
# can or cannot be done with the sortings output. So: pre-phy, and potentially
# excluding useful spikesorting analyses which don't involve the sorting output.


from pathlib import Path

import spikeinterface.full as si

def main():

    print('I''ve ended up using an updated version on Myriad, so ignore this.')
    sorter_folder = Path('D:\Jeffrey\Output\spikesorted\F2302_Challah\ACx_Challah\Test_Output_SpikeSorting_AllmostAll\spikesorted\F2302_Challah\ACx_Challah\All_ACx_Through_Julyish\KiloSortSortingExtractor\sortings_folder')
    sorting = si.read_sorter_folder(sorter_folder)

if __name__ == '__main__':
    main()